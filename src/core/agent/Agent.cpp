#include "Agent.hpp"
#include "PermissionGate.hpp"
#include "../budget/BudgetTracker.hpp"
#include "../config/ConfigManager.hpp"
#include "../session/SessionStats.hpp"
#include "../scm/ScmFactory.hpp"
#include "../utils/FileSystemUtils.hpp"
#include "../logging/Logger.hpp"
#include "../llm/ModelRegistry.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include <algorithm>
#include <future>
#include <ranges>

namespace core::agent {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Returns true if the tool result JSON looks like an error response.
// All Filo tools return {"error": "..."} on failure.
[[nodiscard]] bool is_tool_error(const std::string& result) noexcept {
    return result.find("\"error\"") != std::string::npos;
}

[[nodiscard]] std::string trim_copy(std::string value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Agent::Agent(std::shared_ptr<core::llm::LLMProvider> provider,
             core::tools::ToolManager& skill_manager)
    : provider_(std::move(provider))
    , skill_manager_(skill_manager)
    , orchestrator_(skill_manager_, &core::config::ConfigManager::get_instance().get_config()) {
    ensure_system_prompt();
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
        
        // Auto-configure permissions based on mode intention
        if (current_mode_ == "EXECUTE") {
            permission_profile_ = PermissionProfile::Autonomous;
        } else if (current_mode_ == "PLAN" || current_mode_ == "RESEARCH") {
            permission_profile_ = PermissionProfile::Restricted;
        } else {
            // BUILD / DEBUG / Default
            permission_profile_ = PermissionProfile::Interactive;
        }

        if (!history_.empty() && history_[0].role == "system") {
            history_.erase(history_.begin());
        }
        ensure_system_prompt();
    }
}

void Agent::ensure_system_prompt() {
    std::string prompt =
        "You are Filo, an advanced AI coding assistant running in " + current_mode_ + " mode.\n\n";
    if (current_mode_ == "PLAN" || current_mode_ == "RESEARCH") {
        prompt += "Analyse, research, and plan. Do NOT modify files (avoid apply_patch / write_file). "
                  "Use read and search tools to understand the codebase, then propose a plan.";
    } else if (current_mode_ == "EXECUTE") {
        prompt += "Execute instructions autonomously. Use tools to modify files and run commands "
                  "immediately without asking permission.";
    } else { // BUILD (default)
        prompt += "Build software methodically. Search, read, edit, and run commands. "
                  "Verify your changes where possible. Ask clarifying questions only when truly needed.";
    }

    prompt += "\n\nYou can delegate complex background work via the `task` tool.";
    prompt += " Use the `subagent_type` values listed in the task tool schema/description.";
    prompt += " Default profiles are `general` (broad multi-step work) and";
    prompt += " `explore` (fast read-only codebase search).";
    prompt += " If the user asks with `@general` or `@explore`, map that request to a `task` call.";

    if (current_mode_ == "DEBUG") {
        prompt += "\nRun a tight reproduce -> inspect -> fix -> verify loop. "
                  "Collect concrete diagnostics before editing, prefer the smallest fix that resolves the root cause, "
                  "and finish by rerunning the failing command or test.";
    }

    // ─── Project Context Injection ──────────────────────────────────────────
    try {
        // Detect SCM (Git, etc.) or fallback to NoOp
        auto scm = core::scm::ScmFactory::create(std::filesystem::current_path());
        
        std::string context_section = "\n\n[Project Context]\n";
        bool has_context = false;

        // 1. SCM Status
        std::string status = scm->get_status_summary();
        if (!status.empty()) {
            context_section += "Status:\n" + status + "\n";
            has_context = true;
        }

        // 2. File Tree
        // Limit depth to 2 levels to avoid token bloat, but give high-level map
        std::string tree = core::utils::get_file_tree(std::filesystem::current_path(), *scm, 2);
        if (!tree.empty()) {
            context_section += "Structure:\n" + tree + "\n";
            has_context = true;
        }
        
        if (has_context) {
            prompt += context_section;
        }

    } catch (const std::exception& e) {
        // Fallback: ignore context injection errors, don't crash agent init
        // std::cerr << "Context injection failed: " << e.what() << "\n";
    }
    // ────────────────────────────────────────────────────────────────────────

    if (!context_summary_.empty()) {
        prompt += "\n\nSummary of earlier conversation context:\n";
        prompt += context_summary_;
    }

    if (history_.empty() || history_[0].role != "system") {
        history_.insert(history_.begin(), {"system", prompt, "", "", {}});
    } else {
        history_[0].content = prompt;
    }
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
    ensure_system_prompt();
}

void Agent::compact_history(std::string summary) {
    std::lock_guard lock(history_mutex_);
    context_summary_ = trim_copy(std::move(summary));
    history_.clear();
    consecutive_failure_rounds_ = 0;
    orchestrator_.clear_sessions();
    ensure_system_prompt();
}

void Agent::undo_last() {
    std::lock_guard lock(history_mutex_);
    if (!history_.empty() && history_.back().role == "assistant") history_.pop_back();
    while (!history_.empty() && history_.back().role == "tool") history_.pop_back();
    if (history_.size() > 1 && history_.back().role == "user")  history_.pop_back();
}

std::string Agent::last_user_message() {
    std::lock_guard lock(history_mutex_);
    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
        if (it->role == "user") return it->content;
    }
    return {};
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

    if (!needs_permission(tool_name, profile, args)) {
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
                         std::function<void()> done_callback,
                         TurnCallbacks turn_callbacks) {
    {
        std::lock_guard lock(history_mutex_);
        ensure_system_prompt();
        history_.push_back({"user", user_message, "", "", {}});
        consecutive_failure_rounds_ = 0;  // reset loop breaker on new user input
    }
    step(text_callback, tool_callback, done_callback, std::move(turn_callbacks));
}

// ---------------------------------------------------------------------------
// Core agentic loop step
// ---------------------------------------------------------------------------

void Agent::step(std::function<void(const std::string&)> text_callback,
                 std::function<void(const std::string&, const std::string&)> tool_callback,
                 std::function<void()> done_callback,
                 TurnCallbacks turn_callbacks) {

    auto self = shared_from_this();
    core::llm::ChatRequest request;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string mode_snapshot;
    {
        std::lock_guard lock(history_mutex_);
        request.messages = history_;
        request.model = active_model_;
        provider = provider_;
        mode_snapshot = current_mode_;
    }
    if (!provider) {
        text_callback("\n[Error: no active provider configured]\n");
        done_callback();
        return;
    }
    if (provider->capabilities().supports_tool_calls) {
        request.tools = skill_manager_.get_all_tools();
        request.tools.push_back(orchestrator_.task_tool_definition());
    }

    if (turn_callbacks.on_step_begin) {
        turn_callbacks.on_step_begin();
    }

    // In PLAN/RESEARCH mode strip write-destructive tools
    if (mode_snapshot == "PLAN" || mode_snapshot == "RESEARCH") {
        std::erase_if(request.tools, [](const core::llm::Tool& t) {
            return t.function.name == "apply_patch"
                || t.function.name == "write_file"
                || t.function.name == "replace_in_file"
                || t.function.name == "delete_file"
                || t.function.name == "move_file";
        });
    }

    auto assistant_response   = std::make_shared<std::string>();
    auto tool_calls_accum     = std::make_shared<std::vector<core::llm::ToolCall>>();
    auto reasoning_accum      = std::make_shared<std::string>();  // For Kimi thinking mode

    provider->stream_response(request,
        [self, provider, assistant_response, tool_calls_accum, reasoning_accum,
         text_callback, tool_callback, done_callback, turn_callbacks](
             const core::llm::StreamChunk& chunk) {

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

        // Record token usage in the budget tracker and session stats.
        {
            auto usage = provider->get_last_usage();
            std::string model = provider->get_last_model();
            const bool should_estimate_cost = provider->should_estimate_cost();
            if (model.empty()) {
                std::lock_guard lock(self->history_mutex_);
                model = self->active_model_;
            }
            core::budget::BudgetTracker::get_instance().record(usage, model, should_estimate_cost);
            core::session::SessionStats::get_instance().record_turn(
                model, usage, should_estimate_cost);
        }

        // Persist the assistant message (with accumulated tool calls)
        core::llm::Message asst_msg;
        asst_msg.role               = "assistant";
        asst_msg.content            = *assistant_response;
        asst_msg.tool_calls         = *tool_calls_accum;
        asst_msg.reasoning_content  = *reasoning_accum;  // Required for Kimi thinking mode
        core::logging::debug("[Agent] Persisting assistant message: content_len={}, reasoning_len={}, tool_calls_count={}", 
                            asst_msg.content.size(), asst_msg.reasoning_content.size(), asst_msg.tool_calls.size());
        {
            std::lock_guard lock(self->history_mutex_);
            self->history_.push_back(asst_msg);
        }

        if (tool_calls_accum->empty()) {
            // Pure text response — we're done
            self->check_auto_compact(text_callback);
            done_callback();
            return;
        }

        // ── Execute tool calls ───────────────────────────────────────────
        std::thread([self, tool_calls_accum, text_callback, tool_callback, done_callback, turn_callbacks]() {

            // 1. Permission checks (sequential — only one prompt at a time)
            enum class DeniedReason {
                None,
                UserDenied,
                SkippedAfterEarlierDenial,
            };
            std::vector<bool> approved(tool_calls_accum->size());
            std::vector<DeniedReason> denied_reasons(tool_calls_accum->size(), DeniedReason::None);
            bool denied_any = false;
            for (size_t i = 0; i < tool_calls_accum->size(); ++i) {
                const auto& tc = (*tool_calls_accum)[i];
                if (turn_callbacks.on_tool_start) {
                    turn_callbacks.on_tool_start(tc);
                }
                if (denied_any) {
                    approved[i] = false;
                    denied_reasons[i] = DeniedReason::SkippedAfterEarlierDenial;
                    tool_callback(tc.function.name, "[skipped after earlier denial]");
                    continue;
                }
                approved[i] = self->check_permission(tc.function.name, tc.function.arguments);
                if (!approved[i]) {
                    denied_reasons[i] = DeniedReason::UserDenied;
                    denied_any = true;
                    tool_callback(tc.function.name, "[denied by user]");
                }
            }

            // 2. Execute approved calls in parallel
            std::shared_ptr<core::llm::LLMProvider> provider_for_task;
            std::string active_model_for_task;
            std::string parent_mode_for_task;
            {
                std::lock_guard lock(self->history_mutex_);
                provider_for_task = self->provider_;
                active_model_for_task = self->active_model_;
                parent_mode_for_task = self->current_mode_;
            }

            std::vector<std::future<core::llm::Message>> futures;
            for (size_t i = 0; i < tool_calls_accum->size(); ++i) {
                const auto& tc = (*tool_calls_accum)[i];
                if (!approved[i]) {
                    const bool skipped_after_denial =
                        denied_reasons[i] == DeniedReason::SkippedAfterEarlierDenial;
                    const std::string error_payload = skipped_after_denial
                        ? R"({"error":"Tool call skipped after a previous denial in this step."})"
                        : R"({"error":"Tool call denied by user."})";
                    // Inject a denied message directly
                    futures.push_back(std::async(std::launch::deferred, [tc, error_payload]() {
                        return core::llm::Message{
                            .role        = "tool",
                            .content     = error_payload,
                            .name        = tc.function.name,
                            .tool_call_id = tc.id,
                            .tool_calls   = {}
                        };
                    }));
                    continue;
                }
                    futures.push_back(std::async(std::launch::async,
                    [self, tc, &tool_callback,
                     provider_for_task,
                     active_model_for_task,
                     parent_mode_for_task]() -> core::llm::Message {
                        tool_callback(tc.function.name, tc.function.arguments);
                        std::string result;
                        if (tc.function.name == SubagentOrchestrator::kTaskToolName) {
                            SubagentOrchestrator::RunContext run_context{
                                .active_model = active_model_for_task,
                                .parent_mode = parent_mode_for_task,
                                .permission_check = [self](const std::string& tool_name,
                                                           const std::string& args) {
                                    return self->check_permission(tool_name, args);
                                },
                            };

                            result = self->orchestrator_.execute_task(
                                tc.function.arguments,
                                provider_for_task,
                                run_context);
                        } else {
                            result = self->skill_manager_.execute_tool(
                                tc.function.name,
                                tc.function.arguments);
                        }
                        return core::llm::Message{
                            .role         = "tool",
                            .content      = result,
                            .name         = tc.function.name,
                            .tool_call_id = tc.id,
                            .tool_calls   = {}
                        };
                    }));
            }

            // 3. Collect results and assess failures
            int failure_count = 0;
            for (std::size_t i = 0; i < futures.size(); ++i) {
                auto& f = futures[i];
                auto msg = f.get();
                const bool tool_ok = !is_tool_error(msg.content);
                if (!tool_ok) ++failure_count;
                core::session::SessionStats::get_instance().record_tool_call(tool_ok);
                {
                    std::lock_guard lock(self->history_mutex_);
                    self->history_.push_back(msg);
                }
                if (turn_callbacks.on_tool_finish) {
                    turn_callbacks.on_tool_finish((*tool_calls_accum)[i], msg);
                }
            }

            // 4. Loop-breaker: stop if all tools failed for N rounds in a row
            const bool all_failed = (failure_count == static_cast<int>(futures.size()))
                                 && !futures.empty();
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

            // A "No, suggest something" denial should pause the current loop so
            // the user can provide guidance without new permission prompts.
            if (denied_any) {
                done_callback();
                return;
            }

            // 5. Continue the agentic loop
            // Note: We don't check auto-compact here because we're in the middle
            // of a multi-step execution. Compaction should only happen at natural
            // conversation boundaries (pure text response, not during tool loops).
            self->step(text_callback, tool_callback, done_callback, turn_callbacks);

        }).detach();
    });
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

    // Remove any stale system message — ensure_system_prompt() inserts a fresh one.
    std::erase_if(messages, [](const core::llm::Message& m){ return m.role == "system"; });
    history_ = std::move(messages);
    ensure_system_prompt();
}

std::string Agent::get_mode() const {
    std::lock_guard lock(history_mutex_);
    return current_mode_;
}

std::string Agent::get_context_summary() const {
    std::lock_guard lock(history_mutex_);
    return context_summary_;
}

std::string Agent::get_active_model_name() const {
    std::lock_guard lock(history_mutex_);
    return active_model_;
}

void Agent::check_auto_compact(std::function<void(const std::string&)> text_callback) {
    // Only check compaction when turn is complete (not in the middle of multi-step execution)
    // This is called from:
    // 1. After assistant text response (no tool calls) - OK to check
    // 2. After tool execution before next step - NOT OK, let the loop continue
    // The caller must ensure we're at a natural conversation boundary.
    
    int threshold = 0;
    std::vector<core::llm::Message> history_copy;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string model;
    int max_context_size = 0;
    {
        std::lock_guard lock(history_mutex_);
        threshold = auto_compact_threshold_;
        history_copy = history_;
        provider = provider_;
        model = active_model_;
        // Get model-specific context window size
        max_context_size = provider_->max_context_size();
        
        // Fallback to ModelRegistry if the provider (e.g. Router) doesn't know context size
        if (max_context_size == 0 && !model.empty()) {
            max_context_size = core::llm::get_max_context_size(model);
        }
    }

    // Auto-compact disabled
    if (threshold <= 0) return;
    
    // Estimate total tokens (4 chars per token)
    size_t total_chars = 0;
    for (const auto& msg : history_copy) {
        total_chars += msg.content.size();
        for (const auto& tc : msg.tool_calls) {
            total_chars += tc.function.name.size() + tc.function.arguments.size();
        }
    }
    int estimated_tokens = static_cast<int>(total_chars / 4);

    // Use model-aware ratio-based threshold if context size is known, otherwise fallback to fixed threshold
    constexpr float kCompactionTriggerRatio = 0.75f;  // Compact at 75% of context window
    int effective_threshold = threshold;
    
    if (max_context_size > 0) {
        // Model-aware: trigger at ratio of context window (reference: kimi-cli behavior)
        int ratio_threshold = static_cast<int>(max_context_size * kCompactionTriggerRatio);
        // Use the more conservative of ratio-based or user-configured threshold
        effective_threshold = std::min(threshold, ratio_threshold);
    }

    if (estimated_tokens < effective_threshold) return;

    // Trigger background compaction
    auto self = shared_from_this();
    std::thread([self, provider, model, history_copy, text_callback]() {
        text_callback("\n\n\xe2\x9a\x99  Auto-compacting history...\n");

        core::llm::ChatRequest req;
        req.model = model;
        req.messages = history_copy;
        req.messages.push_back({
            "user",
            "Summarise the conversation so far in a single, dense paragraph. "
            "Focus on the technical context and the state of the task we are working on. "
            "Maintain all critical facts, paths, and requirements."
        });

        auto summary = std::make_shared<std::string>();
        provider->stream_response(req, [self, summary, text_callback](const core::llm::StreamChunk& chunk) {
            if (!chunk.content.empty()) *summary += chunk.content;
            if (!chunk.is_final) return;
            
            if (chunk.is_error) {
                text_callback("\xe2\x9c\x97  History compaction failed.\n\n");
                return;
            }
            
            self->compact_history(*summary);
            text_callback("\xe2\x9c\x93  History compacted.\n\n");
        });
    }).detach();
}

} // namespace core::agent
