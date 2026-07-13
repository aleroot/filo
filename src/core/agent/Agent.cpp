#include "Agent.hpp"
#include "PermissionGate.hpp"
#include "ToolCallPlanner.hpp"
#include "ToolCallScheduler.hpp"
#include "ToolOutputHistory.hpp"
#include "SemanticHistoryEditor.hpp"
#include "../budget/BudgetTracker.hpp"
#include "../config/ConfigManager.hpp"
#include "../context/ContextBuilder.hpp"
#include "../hooks/HookManager.hpp"
#include "../session/SessionStats.hpp"
#include "../tools/MemoryTool.hpp"
#include "../tools/ShellTool.hpp"
#include "../tools/ToolNames.hpp"
#include "../tools/ToolPolicy.hpp"
#include "../tools/ToolSchema.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"
#include "../logging/Logger.hpp"
#include "../llm/ModelRegistry.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <future>
#include <format>
#include <ranges>
#include <utility>

namespace core::agent {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

constexpr int kMaxOutputRecoveryLimit = 3;
constexpr std::string_view kMaxOutputRecoveryPrompt =
    "Output token limit hit. Resume directly with the next unfinished action or text. "
    "Do not apologize or recap. If a tool call was cut off, emit the complete tool call again.";

// Returns true if the tool result JSON looks like an error response.
// All Filo tools return {"error": "..."} on failure.
[[nodiscard]] bool is_tool_error(const std::string& result) noexcept {
    return result.find("\"error\"") != std::string::npos;
}

[[nodiscard]] bool is_truncation_stop_reason(std::string_view stop_reason) noexcept {
    return stop_reason == "max_tokens"
        || stop_reason == "model_context_window_exceeded";
}

[[nodiscard]] bool should_recover_truncated_turn(std::string_view stop_reason,
                                                 bool incomplete_tool_call) noexcept {
    return incomplete_tool_call || is_truncation_stop_reason(stop_reason);
}

[[nodiscard]] std::string provider_notice_subject(std::string_view provider_name) {
    std::string lowered;
    lowered.reserve(provider_name.size());
    for (const char ch : provider_name) {
        lowered.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lowered.find("claude") != std::string::npos) {
        return "Claude";
    }
    if (!provider_name.empty()) {
        return "Provider '" + std::string(provider_name) + "'";
    }
    return "Model";
}

[[nodiscard]] std::string empty_response_detail(std::string_view stop_reason,
                                                bool incomplete_tool_call) {
    if (incomplete_tool_call) {
        return " The stream ended before the provider completed a tool call, so no tool was executed.";
    }
    if (stop_reason == "tool_use") {
        return " The provider reported tool_use, but Filo did not receive a complete tool call.";
    }
    if (stop_reason == "pause_turn") {
        return " The provider paused a server-tool turn before producing client-visible output.";
    }
    if (stop_reason == "refusal") {
        return " The provider refused the request without returning displayable text.";
    }
    if (is_truncation_stop_reason(stop_reason)) {
        return " The response hit the configured output limit before Filo received a complete text block or tool call.";
    }
    return {};
}

[[nodiscard]] std::string empty_response_notice(std::string_view stop_reason,
                                                bool incomplete_tool_call,
                                                std::string_view provider_name) {
    const std::string reason = stop_reason.empty()
        ? std::string("unknown")
        : std::string(stop_reason);
    std::string notice =
        std::format("\n[{} ended this turn with no visible text or tool calls (stop_reason={}).",
                    provider_notice_subject(provider_name),
                    reason);
    notice += empty_response_detail(stop_reason, incomplete_tool_call);
    notice += "]";
    return notice;
}

[[nodiscard]] std::string truncation_notice(std::string_view stop_reason,
                                            bool incomplete_tool_call,
                                            std::string_view provider_name) {
    const auto subject = provider_notice_subject(provider_name);
    if (incomplete_tool_call) {
        return std::format("\n\n[{} stopped before completing a tool call; no tool was executed.]",
                           subject);
    }
    if (stop_reason == "max_tokens") {
        return std::format("\n\n[{} stopped because the response reached max_tokens; this answer may be incomplete.]",
                           subject);
    }
    if (stop_reason == "model_context_window_exceeded") {
        return std::format("\n\n[{} stopped because the model context window was exceeded; this answer may be incomplete.]",
                           subject);
    }
    return {};
}

[[nodiscard]] double sanitize_context_utilization_threshold(double value) noexcept {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return std::clamp(value, 0.0, 1.0);
}

struct HookFieldPayload {
    std::string value;
    bool truncated = false;
};

[[nodiscard]] HookFieldPayload clamp_hook_field(std::string_view text) {
    constexpr std::size_t kMaxChars = 4 * 1024;
    constexpr std::size_t kHeadChars = 3 * 1024;
    constexpr std::size_t kTailChars = kMaxChars - kHeadChars;

    HookFieldPayload payload{.value = std::string(text)};
    if (text.size() <= kMaxChars) {
        return payload;
    }

    payload.truncated = true;
    payload.value.clear();
    payload.value.reserve(kMaxChars + 64);
    payload.value.append(text.substr(0, kHeadChars));
    payload.value.append("\n\n[... hook payload truncated ...]\n\n");
    payload.value.append(text.substr(text.size() - kTailChars));
    return payload;
}

[[nodiscard]] std::string build_user_prompt_hook_payload(
    const core::llm::Message& user_message,
    std::string_view mode)
{
    core::utils::JsonWriter writer(1024);
    const auto content =
        clamp_hook_field(core::llm::message_text_for_display(user_message));
    {
        auto object = writer.object();
        writer.kv_str("role", user_message.role.empty() ? "user" : user_message.role).comma()
              .kv_str("mode", mode).comma()
              .kv_str("content", content.value).comma()
              .kv_num("content_parts_count", user_message.content_parts.size());
        if (content.truncated) {
            writer.comma().kv_bool("content_truncated", true);
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string build_tool_hook_payload(const core::llm::ToolCall& tool_call,
                                                  const core::llm::Message* result = nullptr) {
    core::utils::JsonWriter writer(2048);
    const auto arguments = clamp_hook_field(tool_call.function.arguments);
    {
        auto object = writer.object();
        writer.kv_str("tool_name", tool_call.function.name).comma()
              .kv_str("tool_call_id", tool_call.id).comma()
              .kv_str("arguments", arguments.value);
        if (arguments.truncated) {
            writer.comma().kv_bool("arguments_truncated", true);
        }
        if (result != nullptr) {
            const auto result_content = clamp_hook_field(result->content);
            writer.comma().kv_str("result", result_content.value).comma()
                  .kv_bool("success", !is_tool_error(result->content));
            if (result_content.truncated) {
                writer.comma().kv_bool("result_truncated", true);
            }
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] bool tool_is_allowed_for_turn(std::string_view tool_name,
                                            const Agent::TurnCallbacks& turn_callbacks) {
    if (turn_callbacks.allowed_tools.empty()) {
        return true;
    }
    return core::tools::policy::is_tool_allowed(tool_name, turn_callbacks.allowed_tools);
}

[[nodiscard]] std::string collect_active_skill_context(
    const std::vector<core::llm::Message>& messages) {
    std::string collected;
    for (const auto& message : messages) {
        const std::string_view content = message.content;
        std::size_t cursor = 0;
        while (true) {
            const auto start = content.find("<skill_content name=", cursor);
            if (start == std::string_view::npos) break;
            const auto end = content.find("</skill_content>", start);
            if (end == std::string_view::npos) break;
            const auto after_end = end + std::string_view("</skill_content>").size();
            const auto block = content.substr(start, after_end - start);
            if (collected.find(block) == std::string::npos) {
                if (!collected.empty()) collected += "\n\n";
                collected += block;
            }
            cursor = after_end;
        }
    }
    return collected;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Agent::Agent(std::shared_ptr<core::llm::LLMProvider> provider,
             core::tools::ToolManager& skill_manager,
             core::context::SessionContext session_context)
    : provider_(std::move(provider))
    , skill_manager_(skill_manager)
    , session_context_(std::move(session_context))
    , orchestrator_(skill_manager_, &core::config::ConfigManager::get_instance().get_config()) {
    loop_limits_.max_steps_per_turn = sanitize_max_steps_per_turn(loop_limits_.max_steps_per_turn);
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

// ---------------------------------------------------------------------------
// Cancellation support
// ---------------------------------------------------------------------------

void Agent::request_stop() {
    stop_requested_.store(true, std::memory_order_release);
    std::shared_ptr<core::llm::LLMProvider> provider;
    {
        std::lock_guard lock(history_mutex_);
        provider = provider_;
    }
    if (provider) {
        provider->cancel();
    }
    const auto context = session_context_snapshot();
    if (!context.session_id.empty()) {
        core::tools::ShellTool::interrupt_mcp_session(context.session_id);
    }
}

bool Agent::is_stop_requested() const {
    return stop_requested_.load(std::memory_order_acquire);
}

void Agent::clear_stop_request() {
    stop_requested_.store(false, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Mode management
// ---------------------------------------------------------------------------

void Agent::set_mode(const std::string& mode) {
    std::string clean = mode;
    std::erase_if(clean, [](unsigned char c){ return !std::isalpha(c); });
    if (clean.empty()) clean = "BUILD";
    std::ranges::transform(clean, clean.begin(), ::toupper);

    std::lock_guard lock(history_mutex_);
    if (current_mode_ != clean) {
        current_mode_ = clean;

        if (!history_.empty() && history_[0].role == "system") {
            history_.erase(history_.begin());
        }
        mark_stable_prompt_prefix_dirty();
        ensure_system_prompt();
        refresh_context_window_snapshot_unlocked();
    }
}

void Agent::set_session_id(std::string session_id) {
    std::lock_guard lock(history_mutex_);
    if (session_context_.session_id != session_id) {
        session_context_.session_id = std::move(session_id);
        mark_stable_prompt_prefix_dirty();
    }
}

void Agent::set_session_goal(std::optional<core::session::SessionGoal> goal) {
    std::lock_guard lock(history_mutex_);
    session_goal_ = std::move(goal);
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

void Agent::set_memory_thread_policy(core::memory::MemoryThreadPolicy policy) {
    std::lock_guard lock(history_mutex_);
    session_context_.memory_policy = policy;
    mark_stable_prompt_prefix_dirty();
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

core::memory::MemoryThreadPolicy Agent::memory_thread_policy() const {
    std::lock_guard lock(history_mutex_);
    return session_context_.memory_policy;
}

void Agent::run_memory_review_async(std::function<void(std::string)> status_callback) {
    auto input = [&]() {
        std::lock_guard lock(history_mutex_);
        auto history = history_;
        std::erase_if(history, [](const core::llm::Message& msg) {
            return msg.role == "system";
        });
        return core::memory::MemoryReviewInput{
            .history = std::move(history),
            .session_context = session_context_,
            .thread_policy = session_context_.memory_policy,
            .rate_limit = {},
        };
    }();
    if (provider_) {
        input.rate_limit = provider_->get_last_rate_limit_info();
    }
    auto weak_self = weak_from_this();
    core::memory::MemoryBackgroundService{}.review_async(
        std::move(input),
        [weak_self, status_callback = std::move(status_callback)](
            core::memory::MemoryReviewResult result) {
            if ((result.memories_stored > 0 || result.memories_cleaned > 0)
                && !weak_self.expired()) {
                if (auto self = weak_self.lock()) {
                    self->refresh_system_prompt();
                }
            }
            if (status_callback && !result.message.empty()) {
                status_callback(std::move(result.message));
            }
        });
}

void Agent::run_memory_background_review(
    core::llm::protocols::RateLimitInfo rate_limit,
    std::function<void(const std::string&)> status_log_callback) {
    auto input = [&]() {
        std::lock_guard lock(history_mutex_);
        auto history = history_;
        std::erase_if(history, [](const core::llm::Message& msg) {
            return msg.role == "system";
        });
        return core::memory::MemoryReviewInput{
            .history = std::move(history),
            .session_context = session_context_,
            .thread_policy = session_context_.memory_policy,
            .rate_limit = {},
        };
    }();
    input.rate_limit = std::move(rate_limit);
    auto weak_self = weak_from_this();
    core::memory::MemoryBackgroundService{}.review_async(
        std::move(input),
        [weak_self, status_log_callback = std::move(status_log_callback)](
            core::memory::MemoryReviewResult result) {
            if ((result.memories_stored > 0 || result.memories_cleaned > 0)
                && !weak_self.expired()) {
                if (auto self = weak_self.lock()) {
                    self->refresh_system_prompt();
                }
            }
            if (status_log_callback && !result.message.empty()) {
                status_log_callback("\n[" + result.message + "]\n");
            }
        });
}

void Agent::refresh_system_prompt() {
    std::lock_guard lock(history_mutex_);
    mark_stable_prompt_prefix_dirty();
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

core::context::SessionContext Agent::session_context_snapshot() const {
    std::lock_guard lock(history_mutex_);
    return session_context_;
}

void Agent::reload_subagent_profiles(const core::config::AppConfig& app_config) {
    std::lock_guard lock(history_mutex_);
    orchestrator_.reload_profiles(app_config);
}

int Agent::sanitize_max_steps_per_turn(int value) noexcept {
    return std::clamp(
        value,
        LoopLimits::kMinMaxStepsPerTurn,
        LoopLimits::kMaxMaxStepsPerTurn);
}

std::string Agent::build_stable_prompt_prefix() const {
    return core::context::ContextBuilder(session_context_)
        .with_mode(current_mode_)
        .build();
}

std::string Agent::build_dynamic_prompt_suffix() const {
    std::string suffix = core::session::GoalManager::prompt_context(session_goal_);
    if (!context_summary_.empty()) {
        suffix += "\n\nSummary of earlier conversation context:\n" + context_summary_;
    }
    return suffix;
}

void Agent::refresh_stable_prompt_prefix_unlocked() {
    if (!stable_prompt_prefix_dirty_ && !stable_prompt_prefix_.empty()) {
        return;
    }
    stable_prompt_prefix_ = build_stable_prompt_prefix();
    stable_prompt_prefix_tokens_ = stable_prompt_prefix_.empty()
        ? 0
        : core::context::ContextWindowTracker::estimate_tokens({
            core::llm::Message{
                .role = "system",
                .content = stable_prompt_prefix_,
            },
        });
    stable_prompt_prefix_dirty_ = false;
}

void Agent::ensure_system_prompt() {
    refresh_stable_prompt_prefix_unlocked();
    std::string prompt = stable_prompt_prefix_;
    prompt += build_dynamic_prompt_suffix();

    if (history_.empty() || history_[0].role != "system") {
        history_.insert(history_.begin(), {"system", prompt, "", "", {}});
    } else {
        history_[0].content = prompt;
    }
}

void Agent::refresh_context_window_snapshot_unlocked() noexcept {
    context_window_snapshot_ = core::context::ContextWindowTracker::snapshot(
        history_,
        provider_,
        active_model_,
        stable_prompt_prefix_tokens_);
}

// ---------------------------------------------------------------------------
// History manipulation
// ---------------------------------------------------------------------------

void Agent::clear_history() {
    std::lock_guard lock(history_mutex_);
    history_.clear();
    context_summary_.clear();
    consecutive_failure_rounds_ = 0;
    orchestrator_.clear_sessions();
    reset_efficiency_tracking_unlocked();
    if (provider_) {
        provider_->reset_conversation_state();
    }
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

void Agent::compact_history(std::string summary) {
    std::lock_guard lock(history_mutex_);
    const auto active_skill_context = collect_active_skill_context(history_);
    context_summary_ = core::utils::str::trim_ascii_copy(summary);
    if (!active_skill_context.empty()) {
        if (!context_summary_.empty()) context_summary_ += "\n\n";
        context_summary_ += "Active Agent Skill instructions preserved from earlier context:\n";
        context_summary_ += active_skill_context;
    }
    history_.clear();
    consecutive_failure_rounds_ = 0;
    orchestrator_.clear_sessions();
    reset_efficiency_tracking_unlocked();
    if (provider_) {
        provider_->reset_conversation_state();
    }
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

void Agent::compact_history_async(
    std::function<void(const std::string&)> text_callback,
    std::function<void()> done_callback,
    HistoryCompactionReason reason) {

    std::vector<core::llm::Message> history_copy;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string model;
    {
        std::lock_guard lock(history_mutex_);
        history_copy = history_;
        provider = provider_;
        model = active_model_;
    }

    auto self = shared_from_this();
    history_compactor_.compact_async(
        HistoryCompactionRequest{
            .history = std::move(history_copy),
            .provider = std::move(provider),
            .model = std::move(model),
            .reason = reason,
        },
        HistoryCompactionCallbacks{
            .on_status = std::move(text_callback),
            .on_summary = [self, done_callback = std::move(done_callback)](std::string summary) {
                self->compact_history(std::move(summary));
                if (done_callback) {
                    done_callback();
                }
            },
        });
}

void Agent::undo_last() {
    std::lock_guard lock(history_mutex_);
    if (!history_.empty() && history_.back().role == "assistant") history_.pop_back();
    while (!history_.empty() && history_.back().role == "tool") history_.pop_back();
    if (history_.size() > 1 && history_.back().role == "user")  history_.pop_back();
    refresh_context_window_snapshot_unlocked();
}

std::string Agent::last_user_message() {
    if (const auto last = last_user_turn(); last.has_value()) {
        return core::llm::message_text_for_display(*last);
    }
    return {};
}

bool Agent::has_user_turn() const {
    return last_user_turn().has_value();
}

std::optional<core::llm::Message> Agent::last_user_turn() const {
    std::lock_guard lock(history_mutex_);
    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
        if (it->role == "user") {
            return *it;
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Permission helper
// ---------------------------------------------------------------------------

bool Agent::check_permission(const std::string& tool_name, const std::string& args) {
    PermissionProfile profile;
    {
        std::lock_guard lock(history_mutex_);
        profile = permission_profile_;
    }

    bool permission_required = needs_permission(tool_name, profile, args);
    const bool profile_intentionally_allows_tool =
        profile == PermissionProfile::Standard
        && core::tools::names::is_file_modification_tool(tool_name);
    if (!permission_required
        && profile != PermissionProfile::Autonomous
        && !profile_intentionally_allows_tool) {
        if (const auto def = skill_manager_.get_tool_definition(tool_name); def.has_value()) {
            const auto& ann = def->annotations;
            permission_required = ann.destructive_hint || ann.open_world_hint;
        }
    }

    if (!permission_required) {
        return true;
    }

    std::function<bool(std::string_view, std::string_view)> fn;
    {
        std::lock_guard lock(history_mutex_);
        fn = permission_fn_;
    }
    if (!fn) {
        return true;   // no gate registered → headless mode, allow all
    }

    return fn(tool_name, args);
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void Agent::send_message(const std::string& user_message,
                         std::function<void(const std::string&)> text_callback,
                         std::function<void(const std::string&, const std::string&)> tool_callback,
                         std::function<void()> done_callback) {
    send_message(
        user_message,
        std::move(text_callback),
        std::move(tool_callback),
        std::move(done_callback),
        TurnCallbacks{});
}

void Agent::send_message(const std::string& user_message,
                         std::function<void(const std::string&)> text_callback,
                         std::function<void(const std::string&, const std::string&)> tool_callback,
                         std::function<void()> done_callback,
                         TurnCallbacks turn_callbacks) {
    send_message(
        core::llm::Message{
            .role = "user",
            .content = user_message,
        },
        std::move(text_callback),
        std::move(tool_callback),
        std::move(done_callback),
        std::move(turn_callbacks));
}

void Agent::send_message(core::llm::Message user_message,
                         std::function<void(const std::string&)> text_callback,
                         std::function<void(const std::string&, const std::string&)> tool_callback,
                         std::function<void()> done_callback) {
    send_message(
        std::move(user_message),
        std::move(text_callback),
        std::move(tool_callback),
        std::move(done_callback),
        TurnCallbacks{});
}

void Agent::send_message(core::llm::Message user_message,
                         std::function<void(const std::string&)> text_callback,
                         std::function<void(const std::string&, const std::string&)> tool_callback,
                         std::function<void()> done_callback,
                         TurnCallbacks turn_callbacks) {
    bool expected_idle = false;
    if (!turn_in_progress_.compare_exchange_strong(
            expected_idle,
            true,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        text_callback("\n[Error: another agent turn is already running. Wait for it to finish or stop it before sending a new message.]\n");
        done_callback();
        return;
    }

    auto done_once = std::make_shared<std::atomic<bool>>(false);
    auto finish_turn = [this,
                        done_callback = std::move(done_callback),
                        done_once]() mutable {
        bool expected = false;
        if (!done_once->compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return;
        }
        turn_in_progress_.store(false, std::memory_order_release);
        done_callback();
    };

    clear_stop_request();  // Reset cancellation flag for new turn
    auto turn_state = std::make_shared<TurnState>();
    if (user_message.role.empty()) {
        user_message.role = "user";
    }
    const core::llm::Message hook_message = user_message;
    std::string mode_snapshot;
    const auto session_context = session_context_snapshot();
    {
        std::lock_guard lock(history_mutex_);
        ensure_system_prompt();
        mode_snapshot = current_mode_;
        history_.push_back(std::move(user_message));
        refresh_context_window_snapshot_unlocked();
        consecutive_failure_rounds_ = 0;  // reset loop breaker on new user input
        turn_state->max_steps = sanitize_max_steps_per_turn(loop_limits_.max_steps_per_turn);
    }
    core::hooks::dispatch(
        core::hooks::HookEvent::UserPromptSubmit,
        build_user_prompt_hook_payload(hook_message, mode_snapshot),
        session_context);
    step(
        std::move(text_callback),
        std::move(tool_callback),
        std::move(finish_turn),
        std::move(turn_callbacks),
        std::move(turn_state));
}

// ---------------------------------------------------------------------------
// Core agentic loop step
// ---------------------------------------------------------------------------

void Agent::step(std::function<void(const std::string&)> text_callback,
                 std::function<void(const std::string&, const std::string&)> tool_callback,
                 std::function<void()> done_callback,
                 TurnCallbacks turn_callbacks,
                 std::shared_ptr<TurnState> turn_state) {

    if (!turn_state) {
        turn_state = std::make_shared<TurnState>();
        std::lock_guard lock(history_mutex_);
        turn_state->max_steps = sanitize_max_steps_per_turn(loop_limits_.max_steps_per_turn);
    }

    if (turn_state->transport_turn_id.empty()) {
        const auto sequence = next_transport_turn_id_.fetch_add(1, std::memory_order_relaxed);
        turn_state->transport_turn_id = std::format("{}:{}", session_context_snapshot().session_id, sequence);
    }

    if (turn_state->max_steps > 0 && turn_state->steps_taken >= turn_state->max_steps) {
        const std::string message = std::format(
            "Stopped after reaching the per-turn step limit ({} model steps) without a final response.",
            turn_state->max_steps);
        {
            std::lock_guard lock(history_mutex_);
            history_.push_back({"assistant", message, "", "", {}});
            refresh_context_window_snapshot_unlocked();
        }
        text_callback(message);
        check_auto_compact(turn_callbacks.on_status_log
                               ? turn_callbacks.on_status_log
                               : text_callback);
        done_callback();
        return;
    }
    ++turn_state->steps_taken;

    auto self = shared_from_this();
    const auto step_session_context = session_context_snapshot();
    core::llm::ChatRequest request;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string mode_snapshot;
    std::string provider_name_snapshot;
    std::string dynamic_prompt_suffix;
    {
        std::lock_guard lock(history_mutex_);
        request.messages = history_;
        request.model = turn_callbacks.model_override.empty()
            ? active_model_
            : turn_callbacks.model_override;
        request.effort = effort_level_;
        if (!turn_callbacks.effort_override.empty()) {
            request.effort = turn_callbacks.effort_override;
        }
        if (turn_callbacks.max_tokens_override.has_value()) {
            request.max_tokens = turn_callbacks.max_tokens_override;
        }
        if (turn_callbacks.response_format_override.has_value()) {
            request.response_format = *turn_callbacks.response_format_override;
        }
        request.session_id = step_session_context.session_id;
        request.transport_turn_id = turn_state->transport_turn_id;
        provider = turn_callbacks.provider_override ? turn_callbacks.provider_override : provider_;
        mode_snapshot = current_mode_;
        dynamic_prompt_suffix = build_dynamic_prompt_suffix();
        provider_name_snapshot = turn_callbacks.provider_name_override.empty()
            ? active_provider_name_
            : turn_callbacks.provider_name_override;
    }
    if (!provider) {
        text_callback("\n[Error: no active provider configured]\n");
        done_callback();
        return;
    }
    if (!turn_state->prompt_plan.has_value()) {
        auto plan = core::context::ContextBuilder(step_session_context)
            .with_mode(mode_snapshot)
            .build_plan();
        plan.append(core::context::ContextLayer{
            .kind = core::context::ContextLayerKind::ConversationState,
            .stability = core::context::PromptStability::Dynamic,
            .name = "conversation_state",
            .content = std::move(dynamic_prompt_suffix),
        });
        turn_state->prompt_plan = std::move(plan);
    }
    request.prompt_plan = *turn_state->prompt_plan;
    const auto context_before_edit = core::context::ContextWindowTracker::snapshot(
        request.messages, provider, request.model);
    if (context_before_edit.max_context_tokens > 0
        && context_before_edit.estimated_context_tokens * 100
            >= static_cast<std::size_t>(context_before_edit.max_context_tokens) * 65) {
        auto edited = SemanticHistoryEditor::edit(request.messages);
        if (edited.superseded_results > 0) {
            core::logging::debug(
                "[Agent] Semantic context edit removed {} chars from {} superseded tool results",
                edited.characters_removed,
                edited.superseded_results);
            request.messages = std::move(edited.messages);
        }
    }
    if (provider->capabilities().supports_tool_calls) {
        request.tools = skill_manager_.get_all_tools();
        if (!turn_callbacks.allowed_tools.empty()) {
            std::erase_if(request.tools, [&](const core::llm::Tool& tool) {
                return !tool_is_allowed_for_turn(tool.function.name, turn_callbacks);
            });
        }
        if (tool_is_allowed_for_turn(SubagentOrchestrator::kTaskToolName, turn_callbacks)) {
            request.tools.push_back(orchestrator_.task_tool_definition());
        }
    }

    if (turn_callbacks.on_step_begin) {
        turn_callbacks.on_step_begin();
    }

    // In PLAN/RESEARCH mode strip write-destructive tools
    if (mode_snapshot == "PLAN" || mode_snapshot == "RESEARCH") {
        std::erase_if(request.tools, [](const core::llm::Tool& t) {
            return core::tools::names::is_write_destructive_tool(t.function.name);
        });
    }

    auto assistant_response   = std::make_shared<std::string>();
    auto tool_calls_accum     = std::make_shared<std::vector<core::llm::ToolCall>>();
    auto tool_cost_attribution =
        std::make_shared<std::vector<std::pair<int32_t, int64_t>>>();
    auto reasoning_accum      = std::make_shared<std::string>();  // For Kimi thinking mode
    auto continuation_accum   =
        std::make_shared<std::vector<core::llm::ContinuationItem>>();
    auto already_stopped      = std::make_shared<std::atomic<bool>>(false);

    auto on_stream_chunk =
        [self, provider, assistant_response, tool_calls_accum, reasoning_accum,
         continuation_accum,
         tool_cost_attribution, text_callback, tool_callback, done_callback, turn_callbacks,
         already_stopped, turn_state, step_session_context, provider_name_snapshot](
             const core::llm::StreamChunk& chunk) {

        if (already_stopped->load(std::memory_order_acquire)) {
            return;
        }

        auto finish_stopped_turn = [&]() {
            if (already_stopped->exchange(true, std::memory_order_acq_rel)) {
                return;
            }
            if (!assistant_response->empty() || !reasoning_accum->empty()
                || !continuation_accum->empty()) {
                core::llm::Message stopped_msg;
                stopped_msg.role = "assistant";
                stopped_msg.content = *assistant_response;
                stopped_msg.reasoning_content = *reasoning_accum;
                stopped_msg.continuation_items = *continuation_accum;
                {
                    std::lock_guard lock(self->history_mutex_);
                    self->history_.push_back(std::move(stopped_msg));
                    self->refresh_context_window_snapshot_unlocked();
                }
            }
            text_callback("\n\n[Generation stopped by user]\n");
            done_callback();
        };

        // Check before processing final chunks too: some transports abort the
        // stream by emitting only a final marker after cancellation.
        if (self->is_stop_requested()) {
            finish_stopped_turn();
            return;
        }

        // Accumulate actual response content
        if (!chunk.content.empty()) {
            *assistant_response += chunk.content;
            text_callback(chunk.content);
        }
        
        // Accumulate reasoning content (Kimi K2.5 thinking mode)
        // This is required to be sent back with tool_calls messages
        if (!chunk.reasoning_content.empty()) {
            core::logging::debug("[Agent] Accumulating reasoning_content: '{}'", chunk.reasoning_content.substr(0, 50));
            *reasoning_accum += chunk.reasoning_content;
        }
        continuation_accum->insert(
            continuation_accum->end(),
            chunk.continuation_items.begin(),
            chunk.continuation_items.end());

        // Accumulate streamed tool-call fragments (OpenAI-style delta streaming)
        for (const auto& t : chunk.tools) {
            bool found = false;
            for (auto& acc : *tool_calls_accum) {
                bool same = (t.index != -1 && acc.index == t.index)
                         || (t.index == -1 && !t.id.empty() && acc.id == t.id)
                         || (t.index == -1 && t.id.empty() && tool_calls_accum->size() == 1);
                if (same) {
                    if (!t.id.empty())             acc.id             = t.id;
                    if (!t.type.empty())            acc.type           = t.type;
                    if (!t.function.name.empty())   acc.function.name  = t.function.name;
                    acc.function.arguments += t.function.arguments;
                    found = true;
                    break;
                }
            }
            if (!found) tool_calls_accum->push_back(t);
        }

        if (!chunk.is_final) return;

        // ── Final chunk ──────────────────────────────────────────────────
        // Record API call outcome (is_error is true for HTTP 4XX/5XX or connection errors)
        core::session::SessionStats::get_instance().record_api_call(!chunk.is_error);

        // Record token usage in the ledger and session stats.
        {
            auto usage = provider->get_last_usage();
            std::string model = provider->get_last_model();
            const bool should_estimate_cost = provider->should_estimate_cost();
            if (model.empty()) {
                std::lock_guard lock(self->history_mutex_);
                model = self->active_model_;
            }
            const std::string ledger_actor = turn_callbacks.ledger_actor.empty()
                ? std::string("agent")
                : turn_callbacks.ledger_actor;
            if (usage.has_data()) {
                core::budget::BudgetTracker::get_instance().record_event({
                    .kind = core::budget::TokenLedgerEventKind::Actual,
                    .source = ledger_actor.starts_with("subagent:")
                        ? core::budget::TokenLedgerSource::Subagent
                        : core::budget::TokenLedgerSource::ModelCall,
                    .session_id = step_session_context.session_id,
                    .actor = ledger_actor,
                    .model = model,
                    .usage = usage,
                    .should_estimate_cost = should_estimate_cost,
                    .billable = should_estimate_cost,
                });
            }
            core::session::SessionStats::get_instance().record_turn(
                model, usage, should_estimate_cost);

            if (!tool_calls_accum->empty()) {
                const std::size_t count = tool_calls_accum->size();
                const int64_t completion_cost_micro =
                    core::session::SessionStats::estimate_completion_cost_micro_usd(
                        model, usage.completion_tokens, should_estimate_cost);
                tool_cost_attribution->assign(count, {});
                for (std::size_t i = 0; i < count; ++i) {
                    (*tool_cost_attribution)[i] = {
                        core::session::SessionStats::split_integer_share<int32_t>(
                            usage.completion_tokens, count, i),
                        core::session::SessionStats::split_integer_share<int64_t>(
                            completion_cost_micro, count, i),
                    };
                }
            }

            core::context::ContextWindowSnapshot context_window;
            {
                std::lock_guard lock(self->history_mutex_);
                context_window = core::context::ContextWindowTracker::snapshot(
                    self->history_,
                    provider,
                    model);
                self->efficiency_controller_.record_turn({
                    .prompt_tokens = usage.prompt_tokens,
                    .completion_tokens = usage.completion_tokens,
                    .estimated_history_tokens =
                        context_window.estimated_context_tokens,
                    .max_context_tokens = context_window.max_context_tokens,
                    .provider_is_local = provider ? provider->capabilities().is_local : false,
                    .provider_supports_prompt_caching =
                        !model.empty() && core::llm::ModelRegistry::instance().supports(
                            model,
                            core::llm::ModelCapability::PromptCaching),
                });
            }
        }

        // Persist the assistant message (with accumulated tool calls)
        core::llm::Message asst_msg;
        asst_msg.role               = "assistant";
        asst_msg.content            = *assistant_response;
        asst_msg.tool_calls         = self->is_stop_requested()
            ? std::vector<core::llm::ToolCall>{}
            : *tool_calls_accum;
        asst_msg.reasoning_content  = *reasoning_accum;  // Required for Kimi thinking mode
        asst_msg.continuation_items = *continuation_accum;

        if (!chunk.is_error
            && asst_msg.tool_calls.empty()
            && should_recover_truncated_turn(chunk.stop_reason, chunk.incomplete_tool_call)
            && turn_state->max_output_recovery_count < kMaxOutputRecoveryLimit
            && !self->is_stop_requested()) {
            ++turn_state->max_output_recovery_count;

            const auto status = std::format(
                "\n[{} hit an output limit; continuing automatically ({}/{}).]\n",
                provider_notice_subject(provider_name_snapshot),
                turn_state->max_output_recovery_count,
                kMaxOutputRecoveryLimit);
            if (turn_callbacks.on_status_log) {
                turn_callbacks.on_status_log(status);
            } else {
                text_callback(status);
            }

            {
                std::lock_guard lock(self->history_mutex_);
                if (!asst_msg.content.empty() || !asst_msg.reasoning_content.empty()
                    || !asst_msg.continuation_items.empty()) {
                    self->history_.push_back(asst_msg);
                }
                self->history_.push_back(core::llm::Message{
                    .role = "user",
                    .content = std::string(kMaxOutputRecoveryPrompt),
                    .synthetic = true,
                });
                self->refresh_context_window_snapshot_unlocked();
            }

            self->step(text_callback, tool_callback, done_callback, turn_callbacks, turn_state);
            return;
        }

        if (!chunk.is_error && asst_msg.tool_calls.empty()) {
            if (asst_msg.content.empty()) {
                asst_msg.content = empty_response_notice(
                    chunk.stop_reason,
                    chunk.incomplete_tool_call,
                    provider_name_snapshot);
                text_callback(asst_msg.content);
            } else if (const std::string notice = truncation_notice(
                           chunk.stop_reason,
                           chunk.incomplete_tool_call,
                           provider_name_snapshot);
                       !notice.empty()) {
                asst_msg.content += notice;
                text_callback(notice);
            }
        }

        core::logging::debug("[Agent] Persisting assistant message: content_len={}, reasoning_len={}, tool_calls_count={}", 
                            asst_msg.content.size(), asst_msg.reasoning_content.size(), asst_msg.tool_calls.size());
        const bool should_persist_assistant =
            !self->is_stop_requested()
            || !asst_msg.content.empty()
            || !asst_msg.tool_calls.empty();
        if (should_persist_assistant) {
            std::lock_guard lock(self->history_mutex_);
            self->history_.push_back(asst_msg);
            self->refresh_context_window_snapshot_unlocked();
        }

        // Check if we were stopped - if so, don't proceed to tool execution
        if (self->is_stop_requested()) {
            done_callback();
            return;
        }

        if (tool_calls_accum->empty()) {
            // Pure text response — we're done
            if (turn_callbacks.allow_efficiency_rotation) {
                self->run_efficiency_rotation_if_needed(
                    turn_callbacks.min_context_utilization_for_rotation);
            }
            self->run_memory_background_review(
                provider ? provider->get_last_rate_limit_info()
                         : core::llm::protocols::RateLimitInfo{},
                turn_callbacks.on_status_log);
            self->check_auto_compact(turn_callbacks.on_status_log
                                          ? turn_callbacks.on_status_log
                                          : text_callback);
            done_callback();
            return;
        }

        // ── Execute tool calls ───────────────────────────────────────────
        std::thread([self, tool_calls_accum, tool_cost_attribution, text_callback,
                     tool_callback, done_callback, turn_callbacks, turn_state,
                     step_session_context, provider_name_snapshot]() {
            try {

            // 1. Permission checks (sequential — only one prompt at a time)
            enum class DeniedReason {
                None,
                InvalidArguments,
                BlockedByTurnAllowList,
                BlockedByHook,
                UserDenied,
                SkippedAfterEarlierDenial,
            };
            std::vector<bool> approved(tool_calls_accum->size());
            std::vector<DeniedReason> denied_reasons(tool_calls_accum->size(), DeniedReason::None);
            std::vector<std::string> hook_denial_reasons(tool_calls_accum->size());
            std::vector<std::string> argument_errors(tool_calls_accum->size());
            bool denied_any = false;
            for (size_t i = 0; i < tool_calls_accum->size(); ++i) {
                if (self->is_stop_requested()) {
                    done_callback();
                    return;
                }
                auto& tc = (*tool_calls_accum)[i];
                if (turn_callbacks.on_tool_start) {
                    turn_callbacks.on_tool_start(tc);
                }
                if (!tool_is_allowed_for_turn(tc.function.name, turn_callbacks)) {
                    approved[i] = false;
                    denied_reasons[i] = DeniedReason::BlockedByTurnAllowList;
                    tool_callback(tc.function.name, "[blocked by turn tool allow-list]");
                    continue;
                }
                if (denied_any) {
                    approved[i] = false;
                    denied_reasons[i] = DeniedReason::SkippedAfterEarlierDenial;
                    tool_callback(tc.function.name, "[skipped after earlier denial]");
                    continue;
                }
                const auto hook_decision = core::hooks::run_pre_tool_use(
                    build_tool_hook_payload(tc),
                    step_session_context);
                if (!hook_decision.allowed) {
                    approved[i] = false;
                    denied_reasons[i] = DeniedReason::BlockedByHook;
                    hook_denial_reasons[i] = hook_decision.reason;
                    denied_any = true;
                    tool_callback(
                        tc.function.name,
                        hook_decision.reason.empty()
                            ? "[blocked by PreToolUse hook]"
                            : "[blocked by PreToolUse hook: " + hook_decision.reason + "]");
                    continue;
                }
                approved[i] = hook_decision.approved
                    || self->check_permission(tc.function.name, tc.function.arguments);
                if (!approved[i]) {
                    denied_reasons[i] = DeniedReason::UserDenied;
                    denied_any = true;
                    tool_callback(tc.function.name, "[denied by user]");
                    continue;
                }

                // Validate arguments against the authoritative schema only for
                // calls that passed every approval gate. Gating (allow-list,
                // hooks, user permission) must not depend on tool registration,
                // and its denial precedence must be preserved.
                std::optional<core::tools::ToolDefinition> definition;
                if (tc.function.name == SubagentOrchestrator::kTaskToolName) {
                    definition = self->orchestrator_.task_tool_definition().function;
                } else {
                    definition = self->skill_manager_.get_tool_definition(tc.function.name);
                }
                if (!definition.has_value()) {
                    approved[i] = false;
                    denied_reasons[i] = DeniedReason::InvalidArguments;
                    argument_errors[i] = "tool not found";
                    tool_callback(tc.function.name, "[invalid tool call: tool not found]");
                    continue;
                }
                auto normalized = core::tools::schema::normalize_arguments(
                    *definition,
                    tc.function.arguments);
                if (!normalized.has_value()) {
                    approved[i] = false;
                    denied_reasons[i] = DeniedReason::InvalidArguments;
                    argument_errors[i] = normalized.error();
                    tool_callback(
                        tc.function.name,
                        "[invalid tool arguments: " + normalized.error() + "]");
                    continue;
                }
                tc.function.arguments = std::move(*normalized);
            }

            if (self->is_stop_requested()) {
                done_callback();
                return;
            }

            // 2. Execute approved calls through a resource-aware scheduler.
            std::shared_ptr<core::llm::LLMProvider> provider_for_task;
            std::string active_provider_name_for_task;
            std::string active_model_for_task;
            std::string parent_mode_for_task;
            {
                std::lock_guard lock(self->history_mutex_);
                provider_for_task = turn_callbacks.provider_override
                    ? turn_callbacks.provider_override
                    : self->provider_;
                active_provider_name_for_task = turn_callbacks.provider_name_override.empty()
                    ? self->active_provider_name_
                    : turn_callbacks.provider_name_override;
                if (active_provider_name_for_task.empty()) {
                    active_provider_name_for_task = provider_name_snapshot;
                }
                active_model_for_task = turn_callbacks.model_override.empty()
                    ? self->active_model_
                    : turn_callbacks.model_override;
                parent_mode_for_task = self->current_mode_;
            }

            turn_state->deduplicator.begin_step();
            auto dedup_stop_requested = std::make_shared<std::atomic<bool>>(false);
            std::vector<ScheduledToolTask<core::llm::Message>> scheduled_tasks;
            scheduled_tasks.reserve(tool_calls_accum->size());
            std::vector<std::size_t> original_task_index_by_dedup_index;
            original_task_index_by_dedup_index.reserve(tool_calls_accum->size());
            for (size_t i = 0; i < tool_calls_accum->size(); ++i) {
                const auto& tc = (*tool_calls_accum)[i];
                if (!approved[i]) {
                    const bool invalid_arguments =
                        denied_reasons[i] == DeniedReason::InvalidArguments;
                    const bool skipped_after_denial =
                        denied_reasons[i] == DeniedReason::SkippedAfterEarlierDenial;
                    const bool blocked_by_turn_allow_list =
                        denied_reasons[i] == DeniedReason::BlockedByTurnAllowList;
                    const bool blocked_by_hook =
                        denied_reasons[i] == DeniedReason::BlockedByHook;
                    const std::string error_payload = invalid_arguments
                        ? std::format(
                            R"({{"error":"Invalid tool arguments: {}"}})",
                            core::utils::escape_json_string(argument_errors[i]))
                        : skipped_after_denial
                        ? R"({"error":"Tool call skipped after a previous denial in this step."})"
                        : blocked_by_turn_allow_list
                            ? R"({"error":"Tool call blocked by the turn tool allow-list."})"
                            : blocked_by_hook
                                ? std::format(
                                    R"({{"error":"Tool call blocked by PreToolUse hook: {}"}})",
                                    core::utils::escape_json_string(hook_denial_reasons[i]))
                                : R"({"error":"Tool call denied by user."})";
                    scheduled_tasks.push_back({
                        .accesses = no_tool_access(),
                        .run = [tc, error_payload]() {
                            return core::llm::Message{
                            .role        = "tool",
                            .content     = error_payload,
                            .name        = tc.function.name,
                            .tool_call_id = tc.id,
                            .tool_calls   = {}
                            };
                        },
                    });
                    continue;
                }

                const auto dedup_decision = turn_state->deduplicator.register_call(tc);
                if (dedup_decision.duplicate_in_step) {
                    const std::size_t original_task_index =
                        dedup_decision.original_index < original_task_index_by_dedup_index.size()
                            ? original_task_index_by_dedup_index[dedup_decision.original_index]
                            : i;
                    scheduled_tasks.push_back({
                        .accesses = no_tool_access(),
                        .after = {original_task_index},
                        .run = [tc,
                                original_index = dedup_decision.original_index,
                                deduplicator = &turn_state->deduplicator]() {
                            auto result = deduplicator->duplicate_result(original_index)
                                .value_or(R"({"error":"Tool call deduplicated before the original result was available."})");
                            return core::llm::Message{
                                .role         = "tool",
                                .content      = std::move(result),
                                .name         = tc.function.name,
                                .tool_call_id = tc.id,
                                .tool_calls   = {}
                            };
                        },
                    });
                    continue;
                }

                if (dedup_decision.original_index >= original_task_index_by_dedup_index.size()) {
                    original_task_index_by_dedup_index.resize(dedup_decision.original_index + 1);
                }
                original_task_index_by_dedup_index[dedup_decision.original_index] =
                    scheduled_tasks.size();
                auto planned = plan_tool_call(tc, step_session_context);
                scheduled_tasks.push_back({
                    .accesses = std::move(planned.accesses),
                    .run = [self, tc, &tool_callback,
                     dedup_index = dedup_decision.original_index,
                     deduplicator = &turn_state->deduplicator,
                     dedup_stop_requested,
                     session_context = step_session_context,
                     provider_for_task,
                     active_provider_name_for_task,
                     active_model_for_task,
                     parent_mode_for_task,
                     on_subagent_event = turn_callbacks.on_subagent_event]() -> core::llm::Message {
                        tool_callback(tc.function.name, tc.function.arguments);
                        std::string result;
                        if (tc.function.name == SubagentOrchestrator::kTaskToolName) {
                            SubagentOrchestrator::RunContext run_context{
                                .active_provider_name = active_provider_name_for_task,
                                .active_model = active_model_for_task,
                                .parent_mode = parent_mode_for_task,
                                .session_context = session_context,
                                .permission_check = [self](const std::string& tool_name,
                                                           const std::string& args) {
                                    return self->check_permission(tool_name, args);
                                },
                                .parent_tool_call_id = tc.id,
                                .on_subagent_event = on_subagent_event,
                                // Propagate the parent's stop request (Esc /
                                // Ctrl+C) to the delegated worker so cancelling
                                // a turn also cancels its running subagents.
                                .cancellation_requested = [self]() {
                                    return self->is_stop_requested();
                                },
                            };

                            result = self->orchestrator_.execute_task(
                                tc.function.arguments,
                                provider_for_task,
                                run_context);
                        } else {
                            result = self->skill_manager_.execute_tool(
                                tc.function.name,
                                tc.function.arguments,
                                core::tools::ToolInvocationContext{
                                    .session_context = session_context,
                                    .tool_call_id = tc.id,
                                    .provider_name = active_provider_name_for_task,
                                    .model_name = active_model_for_task,
                                    .provider = provider_for_task,
                                });
                        }
                        if (core::tools::MemoryTool::committed_mutation(
                                tc.function.name,
                                tc.function.arguments,
                                result)) {
                            self->refresh_system_prompt();
                        }
                        result = tool_output_history::clamp_for_history(
                            tc.function.name,
                            result,
                            core::config::ConfigManager::get_instance()
                                .get_config()
                                .context_compression,
                            core::agent::tool_output_history::Context{
                                .tool_arguments = tc.function.arguments,
                                .session_id = session_context.session_id,
                            });
                        auto dedup_final = deduplicator->finalize_for_model(
                            dedup_index,
                            std::move(result));
                        if (dedup_final.stop_turn) {
                            dedup_stop_requested->store(true, std::memory_order_release);
                        }
                        result = std::move(dedup_final.result);
                        deduplicator->complete_original(dedup_index, result);
                        core::llm::Message message{
                            .role         = "tool",
                            .content      = result,
                            .name         = tc.function.name,
                            .tool_call_id = tc.id,
                            .tool_calls   = {}
                        };
                        core::hooks::dispatch(
                            core::hooks::HookEvent::PostToolUse,
                            build_tool_hook_payload(tc, &message),
                            session_context);
                        return message;
                    },
                });
            }

            ToolCallScheduler<core::llm::Message> scheduler;
            auto tool_messages = scheduler.run(std::move(scheduled_tasks));
            turn_state->deduplicator.end_step();

            const bool stop_requested_after_tools = self->is_stop_requested();

            // 3. Collect results and assess failures
            int failure_count = 0;
            const bool dedup_requested_stop =
                dedup_stop_requested->load(std::memory_order_acquire);
            for (std::size_t i = 0; i < tool_messages.size(); ++i) {
                auto msg = std::move(tool_messages[i]);
                const bool tool_ok = !is_tool_error(msg.content);
                if (!tool_ok) ++failure_count;
                const auto& tool_call = (*tool_calls_accum)[i];
                const int32_t argument_tokens =
                    core::session::SessionStats::estimate_payload_tokens(
                        tool_call.function.arguments);
                const int32_t result_tokens =
                    core::session::SessionStats::estimate_payload_tokens(msg.content);
                core::session::SessionStats::get_instance().record_tool_call(
                    tool_call.function.name,
                    tool_ok,
                    argument_tokens,
                    result_tokens,
                    i < tool_cost_attribution->size()
                        ? (*tool_cost_attribution)[i].first
                        : 0,
                    i < tool_cost_attribution->size()
                        ? (*tool_cost_attribution)[i].second
                        : 0);
                const std::string ledger_actor = turn_callbacks.ledger_actor.empty()
                    ? std::string("agent")
                    : turn_callbacks.ledger_actor;
                core::budget::BudgetTracker::get_instance().record_event({
                    .kind = core::budget::TokenLedgerEventKind::Estimate,
                    .source = core::budget::TokenLedgerSource::ToolPayload,
                    .session_id = step_session_context.session_id,
                    .actor = ledger_actor,
                    .tool_name = tool_call.function.name,
                    .note = tool_ok ? std::string{} : std::string("tool_error"),
                    .usage = core::llm::TokenUsage{
                        .prompt_tokens = argument_tokens,
                        .completion_tokens = result_tokens,
                        .total_tokens = argument_tokens + result_tokens,
                    },
                    .should_estimate_cost = false,
                    .billable = false,
                    .cost_micro_usd = 0,
                });
                {
                    std::lock_guard lock(self->history_mutex_);
                    self->history_.push_back(msg);
                    self->refresh_context_window_snapshot_unlocked();
                }
                if (turn_callbacks.on_tool_finish) {
                    turn_callbacks.on_tool_finish(tool_call, msg);
                }
            }

            // 4. Loop-breaker: stop if all tools failed for N rounds in a row
            const bool all_failed = (failure_count == static_cast<int>(tool_messages.size()))
                                 && !tool_messages.empty();
            {
                std::lock_guard lock(self->history_mutex_);
                if (all_failed) {
                    ++self->consecutive_failure_rounds_;
                } else {
                    self->consecutive_failure_rounds_ = 0;
                }
            }

            int rounds;
            {
                std::lock_guard lock(self->history_mutex_);
                rounds = self->consecutive_failure_rounds_;
            }

            if (rounds >= kLoopBreakerThreshold) {
                // Notify the TUI and stop recursing
                std::function<void(int)> break_fn;
                {
                    std::lock_guard lock(self->history_mutex_);
                    break_fn = self->on_loop_break_;
                    self->consecutive_failure_rounds_ = 0;
                }
                if (break_fn) break_fn(rounds);
                done_callback();
                return;
            }

            if (stop_requested_after_tools || self->is_stop_requested()) {
                done_callback();
                return;
            }

            if (dedup_requested_stop) {
                if (!turn_state->final_response_after_repeat_stop_requested) {
                    turn_state->final_response_after_repeat_stop_requested = true;
                    auto final_callbacks = turn_callbacks;
                    final_callbacks.allowed_tools = {"__filo_no_tools__"};
                    self->step(
                        text_callback,
                        tool_callback,
                        done_callback,
                        std::move(final_callbacks),
                        turn_state);
                    return;
                }
                done_callback();
                return;
            }

            // A "No, suggest something" denial should pause the current loop so
            // the user can provide guidance without new permission prompts.
            if (denied_any) {
                done_callback();
                return;
            }

            // 5. Continue the agentic loop
            // A transparent rotation is safe here because the current step's
            // tool results are already in history and the next request can
            // continue from the handoff summary with a leaner working set.
            if (turn_callbacks.allow_efficiency_rotation) {
                self->run_efficiency_rotation_if_needed(
                    turn_callbacks.min_context_utilization_for_rotation);
            }
            self->step(text_callback, tool_callback, done_callback, turn_callbacks, turn_state);
            } catch (const std::exception& e) {
                core::logging::error("Agent tool loop crashed: {}", e.what());
                text_callback(std::string("\n[Internal tool execution error: ") + e.what() + "]");
                done_callback();
            } catch (...) {
                core::logging::error("Agent tool loop crashed: unknown exception");
                text_callback("\n[Internal tool execution error: unknown exception]");
                done_callback();
            }

        }).detach();
    };

    try {
        provider->stream_response(request, on_stream_chunk);
    } catch (const std::exception& e) {
        core::logging::error("Provider threw before streaming started: {}", e.what());
        on_stream_chunk(core::llm::StreamChunk::make_error(
            std::string("\n[Provider startup error: ") + e.what() + "]"));
    } catch (...) {
        core::logging::error("Provider threw before streaming started: unknown exception");
        on_stream_chunk(core::llm::StreamChunk::make_error(
            "\n[Provider startup error: unknown exception]"));
    }
}

// ---------------------------------------------------------------------------
// Session persistence helpers
// ---------------------------------------------------------------------------

std::vector<core::llm::Message> Agent::get_history() const {
    std::lock_guard lock(history_mutex_);
    std::vector<core::llm::Message> result;
    result.reserve(history_.size());
    for (const auto& msg : history_) {
        if (msg.role != "system") result.push_back(msg);
    }
    return result;
}

void Agent::append_history_message(core::llm::Message message) {
    std::lock_guard lock(history_mutex_);
    ensure_system_prompt();
    if (message.role.empty()) {
        message.role = "user";
    }
    history_.push_back(std::move(message));
    mark_stable_prompt_prefix_dirty();
    refresh_context_window_snapshot_unlocked();
}

void Agent::load_history(std::vector<core::llm::Message> messages,
                          const std::string& context_summary,
                          const std::string& mode) {
    std::lock_guard lock(history_mutex_);
    std::string clean_mode = mode;
    std::erase_if(clean_mode, [](unsigned char c){ return !std::isalpha(c); });
    if (clean_mode.empty()) clean_mode = "BUILD";
    std::ranges::transform(clean_mode, clean_mode.begin(), ::toupper);

    current_mode_ = clean_mode;
    context_summary_ = context_summary;
    consecutive_failure_rounds_ = 0;
    orchestrator_.clear_sessions();
    reset_efficiency_tracking_unlocked();
    if (provider_) {
        provider_->reset_conversation_state();
    }

    // Remove any stale system message — ensure_system_prompt() inserts a fresh one.
    std::erase_if(messages, [](const core::llm::Message& m){ return m.role == "system"; });
    history_ = std::move(messages);
    mark_stable_prompt_prefix_dirty();
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

std::string Agent::get_mode() const {
    std::lock_guard lock(history_mutex_);
    return current_mode_;
}

std::string Agent::get_context_summary() const {
    std::lock_guard lock(history_mutex_);
    return context_summary_;
}

core::context::ContextWindowSnapshot Agent::context_window_snapshot() const {
    std::lock_guard lock(history_mutex_);
    return context_window_snapshot_;
}

std::string Agent::get_active_model_name() const {
    std::lock_guard lock(history_mutex_);
    return active_model_;
}

void Agent::set_effort_level(std::string effort) {
    std::lock_guard lock(history_mutex_);
    std::erase_if(effort, [](unsigned char ch) {
        return std::isspace(ch);
    });
    std::ranges::transform(effort, effort.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    effort_level_ = std::move(effort);
}

std::string Agent::get_effort_level() const {
    std::lock_guard lock(history_mutex_);
    return effort_level_;
}

core::session::SessionEfficiencyDecision Agent::current_efficiency_decision_unlocked() const {
    return efficiency_controller_.current_decision();
}

void Agent::reset_efficiency_tracking_unlocked() {
    efficiency_controller_.reset();
}

void Agent::run_efficiency_rotation_if_needed(double min_context_utilization_for_rotation) {
    std::function<void(const core::session::SessionEfficiencyDecision&)> efficiency_fn;
    core::session::SessionEfficiencyDecision efficiency_decision;
    {
        std::lock_guard lock(history_mutex_);
        efficiency_decision = current_efficiency_decision_unlocked();
        efficiency_fn = efficiency_decision_fn_;
    }
    const double min_context_utilization = sanitize_context_utilization_threshold(
        min_context_utilization_for_rotation);
    if (efficiency_decision.context_utilization < min_context_utilization) {
        return;
    }
    if (efficiency_decision.action == core::session::SessionEfficiencyDecision::Action::Rotate
        && efficiency_fn) {
        efficiency_fn(efficiency_decision);
    }
}

void Agent::check_auto_compact(std::function<void(const std::string&)> status_log_callback) {
    // Only check compaction when turn is complete (not in the middle of multi-step execution)
    // This is called from:
    // 1. After assistant text response (no tool calls) - OK to check
    // 2. After tool execution before next step - NOT OK, let the loop continue
    // The caller must ensure we're at a natural conversation boundary.
    
    int threshold = 0;
    std::vector<core::llm::Message> history_copy;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string model;
    core::context::CompactionDecision compaction;
    {
        std::lock_guard lock(history_mutex_);
        threshold = auto_compact_threshold_;
        const bool use_model_aware_default_threshold =
            auto_compact_uses_model_window_default_;
        history_copy = history_;
        provider = provider_;
        model = active_model_;
        compaction = core::context::ContextWindowTracker::compaction_decision(
            history_copy,
            provider,
            model,
            core::context::CompactionTriggerPolicy{
                .configured_token_threshold = threshold,
                .use_model_aware_default_threshold =
                    use_model_aware_default_threshold,
            });
    }

    // Auto-compact disabled
    if (threshold <= 0) return;

    if (!compaction.should_compact) return;

    auto self = shared_from_this();
    history_compactor_.compact_async(
        HistoryCompactionRequest{
            .history = std::move(history_copy),
            .provider = std::move(provider),
            .model = std::move(model),
            .reason = HistoryCompactionReason::Auto,
        },
        HistoryCompactionCallbacks{
            .on_status = std::move(status_log_callback),
            .on_summary = [self](std::string summary) {
                self->compact_history(std::move(summary));
            },
        });
}

} // namespace core::agent
