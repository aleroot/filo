#include "Agent.hpp"
#include "AgentCapabilities.hpp"
#include "../budget/BudgetTracker.hpp"
#include "../config/ConfigManager.hpp"
#include "../context/ContextBuilder.hpp"
#include "../hooks/HookManager.hpp"
#include "../llm/ModelRegistry.hpp"
#include "../llm/OneShotCompletion.hpp"
#include "../logging/Logger.hpp"
#include "../memory/MemorySystem.hpp"
#include "../scm/EphemeralWorktree.hpp"
#include "../session/SessionStats.hpp"
#include "../session/SessionStore.hpp"
#include "../tools/MemoryTool.hpp"
#include "../tools/ShellTool.hpp"
#include "../tools/ToolNames.hpp"
#include "../tools/ToolPolicy.hpp"
#include "../tools/read/ReadTypes.hpp"
#include "../tools/ToolSchema.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"
#include "PermissionGate.hpp"
#include "RepositoryContextMessage.hpp"
#include "SemanticHistoryEditor.hpp"
#include "ToolCallPlanner.hpp"
#include "ToolCallScheduler.hpp"
#include "ToolOutputHistory.hpp"
#include "ToolRecovery.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <ranges>
#include <thread>
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

class TurnCompletionState final {
public:
    explicit TurnCompletionState(
        std::shared_ptr<core::power::SleepInhibitionLease> inhibition) noexcept
        : sleep_inhibition_(std::move(inhibition)) {}

    [[nodiscard]] std::optional<std::shared_ptr<core::power::SleepInhibitionLease>>
    claim_completion() noexcept {
        if (completed_.exchange(true, std::memory_order_acq_rel)) {
            return std::nullopt;
        }
        return std::move(sleep_inhibition_);
    }

private:
    std::atomic_bool completed_{false};
    std::shared_ptr<core::power::SleepInhibitionLease> sleep_inhibition_;
};

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

[[nodiscard]] std::string build_tool_batch_hook_payload(
    std::span<const core::llm::ToolCall> tool_calls,
    std::span<const core::llm::Message> results) {
    std::string tool_names;
    for (const auto& tool_call : tool_calls) {
        if (!tool_names.empty()) {
            tool_names += ',';
        }
        tool_names += tool_call.function.name;
    }
    const auto names = clamp_hook_field(tool_names);
    const auto failed = std::ranges::count_if(results, [](const auto& result) {
        return is_tool_error(result.content);
    });
    core::utils::JsonWriter writer(512);
    {
        auto object = writer.object();
        writer.kv_num("tool_count", tool_calls.size()).comma()
              .kv_num("failure_count", failed).comma()
              .kv_str("tool_names", names.value);
        if (names.truncated) {
            writer.comma().kv_bool("tool_names_truncated", true);
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string build_stop_hook_payload(
    const core::llm::Message& assistant_message,
    std::string_view mode,
    bool mutation_observed) {
    const auto response =
        clamp_hook_field(core::llm::message_text_for_display(assistant_message));
    core::utils::JsonWriter writer(1024);
    {
        auto object = writer.object();
        writer.kv_str("mode", mode).comma()
              .kv_str("response", response.value).comma()
              .kv_bool("mutation_observed", mutation_observed);
        if (response.truncated) {
            writer.comma().kv_bool("response_truncated", true);
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

void append_compaction_section(
    std::string& summary,
    std::string_view title,
    std::string_view content) {
    const auto trimmed = core::utils::str::trim_ascii_view(content);
    if (trimmed.empty()) return;
    if (!summary.empty()) summary += "\n\n";
    summary += title;
    summary += ":\n";
    summary += trimmed;
}

[[nodiscard]] HistoryCompactionPolicy compaction_policy(
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    std::string_view model) noexcept {
    return HistoryCompactionPlanner::policy_for_context_window(
        core::context::ContextWindowTracker::resolve_max_context_tokens(
            provider,
            model));
}

struct InvalidToolHistory {
    std::size_t index = 0;
    std::string reason;
};

[[nodiscard]] std::optional<InvalidToolHistory> invalid_tool_history(
    const std::vector<core::llm::Message>& messages) {
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& assistant = messages[i];
        if (assistant.role != "assistant" || assistant.tool_calls.empty()) {
            continue;
        }

        std::vector<bool> matched(assistant.tool_calls.size(), false);
        std::size_t result_count = 0;
        std::size_t cursor = i + 1;
        while (cursor < messages.size() && messages[cursor].role == "tool") {
            const auto& result = messages[cursor];
            const auto call = std::ranges::find_if(
                assistant.tool_calls,
                [&](const core::llm::ToolCall& tool_call) {
                    return tool_call.id == result.tool_call_id;
                });
            if (call == assistant.tool_calls.end()) {
                return InvalidToolHistory{
                    .index = i,
                    .reason = std::format(
                        "tool result '{}' does not belong to the preceding assistant message at history index {}",
                        result.tool_call_id,
                        i),
                };
            }
            const auto match_index = static_cast<std::size_t>(
                std::distance(assistant.tool_calls.begin(), call));
            if (matched[match_index]) {
                return InvalidToolHistory{
                    .index = i,
                    .reason = std::format(
                        "tool call '{}' has more than one result after history index {}",
                        result.tool_call_id,
                        i),
                };
            }
            matched[match_index] = true;
            ++result_count;
            ++cursor;
        }

        if (result_count != assistant.tool_calls.size()) {
            std::string missing;
            for (std::size_t call_index = 0;
                 call_index < assistant.tool_calls.size();
                 ++call_index) {
                if (matched[call_index]) continue;
                if (!missing.empty()) missing += ", ";
                missing += assistant.tool_calls[call_index].id;
            }
            return InvalidToolHistory{
                .index = i,
                .reason = std::format(
                    "assistant tool calls at history index {} are not immediately followed by all results; missing: {}",
                    i,
                    missing),
            };
        }
    }
    return std::nullopt;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Agent::Agent(std::shared_ptr<core::llm::LLMProvider> provider,
             core::tools::ToolManager& skill_manager,
             core::context::SessionContext session_context,
             std::filesystem::path tool_result_root,
             std::shared_ptr<core::power::SleepInhibitor> sleep_inhibitor,
             std::shared_ptr<core::session::SessionStatsRegistry> session_stats_registry,
             core::budget::BudgetTracker* budget_tracker,
             std::shared_ptr<core::memory::MemorySystem> memory_system,
             std::shared_ptr<core::scm::WorkspaceLeaseRegistry>
                 workspace_leases)
    : provider_(std::move(provider))
    , sleep_inhibitor_(sleep_inhibitor
          ? std::move(sleep_inhibitor)
          : core::power::make_sleep_inhibitor())
    , skill_manager_(skill_manager)
    , session_context_(std::move(session_context))
    // Dependency injection: the execution root shares one registry across all
    // of its agents; the process-shared registry preserves legacy callers.
    , session_stats_registry_(session_stats_registry
          ? std::move(session_stats_registry)
          : core::session::SessionStatsRegistry::shared_instance())
    , workspace_leases_(workspace_leases
          ? std::move(workspace_leases)
          : std::make_shared<core::scm::WorkspaceLeaseRegistry>())
    , budget_tracker_(budget_tracker
          ? budget_tracker
          : &core::budget::BudgetTracker::get_instance())
    , orchestrator_(skill_manager_,
                    &core::config::ConfigManager::get_instance().get_config(),
                    session_stats_registry_,
                    workspace_leases_)
    , todo_manager_(&core::session::SessionStore::now_iso8601)
    , todo_tool_(todo_manager_)
    , tool_result_store_(std::move(tool_result_root))
    , read_tool_result_tool_(tool_result_store_)
    // Memory is optional: execution roots inject the MemorySystem they own.
    // Tests and embedders that omit it get a private inert system (null
    // recovery, default semantic store) — never a process-global instance.
    , memory_system_(memory_system
          ? std::move(memory_system)
          : std::make_shared<core::memory::MemorySystem>())
    , auto_turn_coordinator_(workspace_leases_) {
    loop_limits_.max_steps_per_turn = sanitize_max_steps_per_turn(loop_limits_.max_steps_per_turn);
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

Agent::~Agent() = default;

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

bool Agent::last_turn_failed() const {
    return turn_failed_.load(std::memory_order_acquire);
}

bool Agent::turn_in_progress() const noexcept {
    return turn_in_progress_.load(std::memory_order_acquire);
}

std::string Agent::session_id() const {
    return session_context_snapshot().session_id;
}

std::expected<core::verification::Receipt, std::string>
Agent::run_verification_recipe(std::string_view recipe_id,
                               const std::filesystem::path &root) {
  std::shared_ptr<core::llm::LLMProvider> provider;
  std::string provider_name;
  std::string model_name;
  {
    std::lock_guard lock(history_mutex_);
    provider = provider_;
    provider_name = active_provider_name_;
    model_name = active_model_;
  }
  // An explicit root runs the same recipe inside an isolated copy of the
  // workspace (Boost baseline attribution). Rebinding the primary keeps every
  // path check and tool resolution scoped to that copy.
  auto context = session_context_snapshot();
  if (!root.empty() && !context.set_workspace_primary(root)) {
    return std::unexpected(
        "Could not scope verification to " + root.string());
  }
  return capabilities::execute_verification(
      skill_manager_,
      capabilities::VerificationRequest{
          .recipe_id = std::string(recipe_id),
          .session_context = std::move(context),
          .tool_call_id = std::format(
              "goal-verification-{}",
              next_transport_turn_id_.fetch_add(1, std::memory_order_relaxed)),
          .provider_name = std::move(provider_name),
          .model_name = std::move(model_name),
          .provider = std::move(provider),
          .permission_check =
              [this](const std::string &tool,
                     const std::string &arguments) {
                return check_permission(tool, arguments);
              },
      });
}

std::expected<std::string, std::string>
Agent::run_read_only_goal_task(std::string_view description,
                               std::string_view prompt) {
  std::shared_ptr<core::llm::LLMProvider> provider;
  std::shared_ptr<core::memory::MemorySystem> memory_system;
  std::string provider_name;
  std::string model_name;
  std::string parent_mode;
  {
    std::lock_guard lock(history_mutex_);
    provider = provider_;
    memory_system = memory_system_;
    provider_name = active_provider_name_;
    model_name = active_model_;
    parent_mode = std::string(to_string(current_mode_));
  }
  return capabilities::execute_exploration(
      orchestrator_,
      capabilities::ExplorationRequest{
          .description = std::string(description),
          .prompt = std::string(prompt),
          .provider = std::move(provider),
          .provider_name = std::move(provider_name),
          .model_name = std::move(model_name),
          .parent_mode = std::move(parent_mode),
          .session_context = session_context_snapshot(),
          .tool_call_id = std::format(
              "goal-explore-{}",
              next_transport_turn_id_.fetch_add(1, std::memory_order_relaxed)),
          .permission_check =
              [this](const std::string &tool, const std::string &arguments) {
                return check_permission(tool, arguments);
              },
          .cancellation_requested = [this] { return is_stop_requested(); },
          .memory_system = std::move(memory_system),
      });
}

bool Agent::is_turn_current(const std::shared_ptr<TurnState>& turn_state) const {
    std::lock_guard lock(history_mutex_);
    return turn_state->conversation_generation == conversation_generation_;
}

void Agent::capture_turn_provider_snapshot_unlocked(
    TurnState& turn_state, const TurnCallbacks& turn_callbacks) {
    turn_state.conversation_generation = conversation_generation_;
    turn_state.provider = turn_callbacks.provider_override
        ? turn_callbacks.provider_override
        : provider_;
    turn_state.provider_name = turn_callbacks.provider_name_override.empty()
        ? active_provider_name_
        : turn_callbacks.provider_name_override;
    turn_state.model = turn_callbacks.model_override.empty()
        ? active_model_
        : turn_callbacks.model_override;
}

AutoGraphOrchestrator::Hooks Agent::make_auto_graph_hooks(
    std::shared_ptr<core::llm::LLMProvider> provider,
    std::string provider_name,
    std::string model,
    core::context::SessionContext session_context,
    std::function<void(const SubagentEvent&)> on_subagent_event,
    bool boost, std::vector<std::string> allowed_tools, std::string effort) {
  AutoGraphOrchestrator::Hooks hooks;
  hooks.cancellation_requested = [this] { return is_stop_requested(); };
  if (!provider) {
    return hooks;
  }

  const std::string boost_effort(BoostPipeline::reasoning_effort(effort));
  hooks.complete = [provider, model, boost, boost_effort, this](std::string_view prompt) {
    return core::llm::complete_once(
        provider, model, prompt,
        [this] { return is_stop_requested(); }, boost ? boost_effort : "");
  };
  if (!provider->capabilities().supports_tool_calls ||
      (!allowed_tools.empty() && !core::tools::policy::is_tool_allowed(
          SubagentOrchestrator::kTaskToolName, allowed_tools)))
    return hooks;
  auto memory_system = memory_system_;
  // Implementation workstreams inherit the user's selected mode; they are only
  // enabled when that mode permits writes at all, and when the workspace has
  // opted in (they cost two cold gate runs, which large compiled projects
  // should not pay by surprise).
  std::string worker_mode = get_mode();
  const bool writable_workers =
      boost && !is_read_only_mode(agent_mode_from_string(worker_mode)) &&
      BoostPipeline::candidates_enabled(
          session_context.workspace_view().primary());
  hooks.explore =
      [this, provider, provider_name, model, session_context, memory_system,
       on_subagent_event, boost, boost_effort, allowed_tools](
          const core::goal::Node &node,
          std::string_view retry_context) {
        std::string prompt =
            node.directive.empty() ? node.name : node.directive;
        prompt += retry_context;
        const auto evidence = capabilities::execute_exploration(
            orchestrator_,
            capabilities::ExplorationRequest{
                .description = node.name,
                .prompt = std::move(prompt),
                .provider = provider,
                .provider_name = provider_name,
                .model_name = model,
                .parent_mode = std::string(to_string(AgentMode::Research)),
                .session_context = session_context,
                .tool_call_id = std::format("auto-graph:{}", node.name),
                .permission_check =
                    [this, allowed_tools](const std::string &tool,
                           const std::string &arguments) {
                      return (allowed_tools.empty() || core::tools::policy::is_tool_allowed(tool, allowed_tools)) &&
                             check_permission(tool, arguments);
                    },
                .on_subagent_event = on_subagent_event,
                .cancellation_requested =
                    [this] { return is_stop_requested(); },
                .memory_system = memory_system,
                .effort = boost || node.name.starts_with("BOOST") ? boost_effort : "",
            });
        if (!evidence.has_value()) {
          return core::goal::WorkOutcome{
              .ok = false,
              .output = evidence.error(),
          };
        }
        return core::goal::WorkOutcome{
            .ok = true,
            .output = *evidence,
        };
      };

  // Phase 2 implementation workstreams. Each candidate gets its own throwaway
  // Git worktree and is a writer only *there*; the user's checkout keeps a
  // single writer. Without Git, or in a read-only mode, Boost simply runs with
  // investigations alone rather than degrading the isolation guarantee.
  if (writable_workers) {
    hooks.implement =
        [this, provider = std::move(provider),
         provider_name = std::move(provider_name), model = std::move(model),
         session_context = std::move(session_context),
         memory_system = std::move(memory_system),
         on_subagent_event = std::move(on_subagent_event), boost_effort,
         worker_mode = std::move(worker_mode),
         allowed_tools = std::move(allowed_tools)](
            const core::goal::Node &node,
            std::string_view label) -> std::optional<BoostCandidate> {
          if (is_stop_requested()) {
            return std::nullopt;
          }
          auto worktree = core::scm::EphemeralWorktree::create(
              session_context.workspace_view().primary(), label);
          if (!worktree.has_value()) {
            return std::nullopt;
          }
          auto isolated = session_context;
          if (!isolated.set_workspace_primary(worktree->root())) {
            return std::nullopt;
          }

          BoostCandidate candidate;
          candidate.name = node.name;
          const auto evidence = capabilities::execute_exploration(
              orchestrator_,
              capabilities::ExplorationRequest{
                  .description = node.name,
                  .prompt = node.directive,
                  .provider = provider,
                  .provider_name = provider_name,
                  .model_name = model,
                  .parent_mode = worker_mode,
                  .session_context = isolated,
                  .tool_call_id = std::format("boost-candidate:{}", node.name),
                  .permission_check =
                      [this, allowed_tools](const std::string &tool,
                                            const std::string &arguments) {
                        return (allowed_tools.empty() ||
                                core::tools::policy::is_tool_allowed(
                                    tool, allowed_tools)) &&
                               check_permission(tool, arguments);
                      },
                  .on_subagent_event = on_subagent_event,
                  .cancellation_requested =
                      [this] { return is_stop_requested(); },
                  .memory_system = memory_system,
                  .effort = boost_effort,
                  .read_only = false,
              });
          candidate.ok = evidence.has_value();
          candidate.evidence =
              candidate.ok ? *evidence : "Worker failed: " + evidence.error();
          if (const auto patch = worktree->patch()) {
            candidate.patch = *patch;
          }

          // The harness re-runs the repository gate inside the candidate's own
          // worktree. A candidate is "verified" only on receipts Filo produced
          // itself: a worker asserting success in prose proves nothing.
          if (candidate.ok && !candidate.patch.empty() && !is_stop_requested()) {
            const core::verification::Catalog catalog;
            const auto discovered = catalog.discover(worktree->root());
            const auto gate = core::verification::Catalog::default_quality_gate(
                discovered.recipes);
            bool all_passed = !gate.empty();
            for (const auto &recipe_id : gate) {
              if (is_stop_requested()) {
                all_passed = false;
                break;
              }
              const auto receipt =
                  run_verification_recipe(recipe_id, worktree->root());
              if (!receipt.has_value()) {
                all_passed = false;
                candidate.evidence += std::format(
                    "\n[candidate check] {}: {}", recipe_id, receipt.error());
                continue;
              }
              all_passed = all_passed && receipt->passed();
              candidate.evidence += std::format(
                  "\n[candidate check] {} (exit {}): {}\n{}", receipt->recipe_id,
                  receipt->exit_code, receipt->command,
                  receipt->evidence.substr(0, 2048));
            }
            candidate.verified = all_passed;
          }
          return candidate;
        };
  }
  return hooks;
}

// ---------------------------------------------------------------------------
// Mode management
// ---------------------------------------------------------------------------

void Agent::set_mode(const std::string &mode) {
  const AgentMode parsed = agent_mode_from_string(mode);

  std::lock_guard lock(history_mutex_);
  if (current_mode_ != parsed) {
    current_mode_ = parsed;

    if (!history_.empty() && history_[0].role == "system") {
      history_.erase(history_.begin());
    }
    refresh_stable_prompt_state_unlocked();
  }
}

void Agent::set_session_id(std::string session_id) {
    std::lock_guard lock(history_mutex_);
    if (session_context_.session_id != session_id) {
        session_context_.session_id = std::move(session_id);
        mark_stable_prompt_prefix_dirty();
        ++history_revision_;
    }
}

std::size_t Agent::grant_workspace_paths(
    const std::vector<std::filesystem::path>& paths) {
    std::lock_guard lock(history_mutex_);
    const auto added = session_context_.extend_workspace(paths);
    if (added > 0) {
        refresh_stable_prompt_state_unlocked();
    }
    return added;
}

bool Agent::change_workspace_root(const std::filesystem::path& new_primary) {
    std::lock_guard lock(history_mutex_);
    const bool changed = session_context_.set_workspace_primary(new_primary);
    if (changed) {
        refresh_stable_prompt_state_unlocked();
    }
    return changed;
}

core::workspace::SessionWorkspace Agent::workspace_snapshot() const {
    return session_context_snapshot().workspace_view();
}

void Agent::set_session_goal(std::optional<core::session::SessionGoal> goal) {
    std::lock_guard lock(history_mutex_);
    session_goal_ = std::move(goal);
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

void Agent::set_goal_graph_context_fn(std::function<std::string()> fn) {
    std::lock_guard lock(history_mutex_);
    goal_graph_context_fn_ = std::move(fn);
    ensure_system_prompt();
    refresh_context_window_snapshot_unlocked();
}

void Agent::restore_todos(std::vector<core::session::SessionTodoItem> todos) {
    todo_manager_.restore(std::move(todos));
}

std::vector<core::session::SessionTodoItem> Agent::get_todos() const {
    return todo_manager_.current();
}

std::expected<core::session::SessionTodoItem, std::string>
Agent::add_todo(std::string_view text) {
    return todo_manager_.add(text);
}

std::expected<core::session::SessionTodoItem, std::string>
Agent::set_todo_status(std::string_view selector, core::session::TodoStatus status) {
    return todo_manager_.set_status(selector, status);
}

std::expected<core::session::SessionTodoItem, std::string>
Agent::remove_todo(std::string_view selector) {
    return todo_manager_.remove(selector);
}

std::size_t Agent::clear_completed_todos() {
    return todo_manager_.clear_completed();
}

void Agent::clear_todos() {
    todo_manager_.clear();
}

void Agent::set_memory_thread_policy(core::memory::MemoryThreadPolicy policy) {
    std::lock_guard lock(history_mutex_);
    session_context_.memory_policy = policy;
    refresh_stable_prompt_state_unlocked();
}

core::memory::MemoryThreadPolicy Agent::memory_thread_policy() const {
    std::lock_guard lock(history_mutex_);
    return session_context_.memory_policy;
}

std::shared_ptr<core::memory::MemorySystem> Agent::memory_system() const {
    return memory_system_;
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
    memory_system_->background().review_async(
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
    memory_system_->background().review_async(
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
    refresh_stable_prompt_state_unlocked();
}

void Agent::refresh_stable_prompt_state_unlocked() {
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

std::string Agent::build_dynamic_prompt_suffix() const {
    std::string suffix = core::session::GoalManager::prompt_context(session_goal_);
    // The graph engine renders its own (empty when no goal is active) block,
    // so a plain session goal and a running DAG compose without duplication.
    if (goal_graph_context_fn_) {
        suffix += goal_graph_context_fn_();
    }
    suffix += todo_manager_.prompt_context();
    if (!context_summary_.empty()) {
        suffix += "\n\nSummary of earlier conversation context:\n" + context_summary_;
    }
    return suffix;
}

void Agent::append_project_facts_update_unlocked(
    std::optional<core::context::ProjectFactsSnapshot> current) {
    if (!current.has_value()) {
        return;
    }

    std::string content;
    if (!project_facts_snapshot_.has_value()) {
        content = core::context::render_project_facts(*current);
        if (!content.empty()) {
            content.insert(
                0,
                "Repository snapshot captured at the start of this turn. Later "
                "[Project Context Update] messages supersede only the sections they include.");
        }
    } else {
        content = core::context::render_project_facts_update(
            *project_facts_snapshot_, *current);
    }
    project_facts_snapshot_ = std::move(current);

    if (!content.empty()) {
        for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
            if (!is_repository_context_message(*it)) continue;
            it->input_text.clear();
            break;
        }
        history_.push_back(core::llm::Message{
            .role = "user",
            .content = std::move(content),
            .name = std::string(kRepositoryContextMessageName),
            .input_text = encode_repository_context_state(
                project_facts_snapshot_->status,
                project_facts_snapshot_->tree),
            .synthetic = true,
        });
    }
}

void Agent::refresh_stable_prompt_prefix_unlocked() {
    if (!stable_prompt_prefix_dirty_ && !stable_prompt_prefix_.empty()) {
        return;
    }
    stable_prompt_plan_ =
        core::context::ContextBuilder(session_context_)
            .with_mode(to_string(current_mode_))
            .with_memory_prompt(memory_system_->semantic_prompt_block(
                session_context_))
            .include_project_facts(false)
            .build_plan();
    stable_prompt_prefix_ = stable_prompt_plan_.render();
    if (stable_prompt_prefix_.empty()) {
        stable_prompt_prefix_tokens_ = 0;
    } else {
        // Named, so the probe vector outlives the call unambiguously: GCC
        // reports a dangling temporary when this is spelled inline inside a
        // conditional expression.
        const std::vector<core::llm::Message> prefix_probe{
            core::llm::Message{
                .role = "system",
                .content = stable_prompt_prefix_,
            },
        };
        stable_prompt_prefix_tokens_ =
            core::context::ContextWindowTracker::estimate_tokens(prefix_probe);
    }
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
    ++history_revision_;
}

// ---------------------------------------------------------------------------
// History manipulation
// ---------------------------------------------------------------------------

void Agent::clear_history() {
    std::lock_guard lock(history_mutex_);
    ++conversation_generation_;
    history_.clear();
    context_summary_.clear();
    project_facts_snapshot_.reset();
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
    auto plan = HistoryCompactionPlanner::plan(
        history_,
        context_summary_,
        compaction_policy(provider_, active_model_));
    const auto active_skill_context = collect_active_skill_context(history_);
    apply_compaction_unlocked(
        std::move(summary),
        std::move(plan),
        active_skill_context);
}

void Agent::apply_compaction_unlocked(
    std::string summary,
    HistoryCompactionPlan plan,
    std::string active_skill_context) {
    context_summary_ = core::utils::str::trim_ascii_copy(summary);
    append_compaction_section(
        context_summary_,
        "Recent untrusted execution record preserved by Filo",
        plan.execution_checkpoint);
    append_compaction_section(
        context_summary_,
        "Active Agent Skill instructions preserved from earlier context",
        active_skill_context);

    history_.clear();
    project_facts_snapshot_.reset();
    // Efficiency samples describe the old prompt shape and must start a fresh
    // baseline. Execution-owned state (loop breaker, todos, goals, resumable
    // subagents, and tool-result storage) deliberately survives compaction.
    reset_efficiency_tracking_unlocked();
    if (provider_) {
        // Every provider must discard transport continuation handles after a
        // history rewrite. This does not affect authentication or session-owned
        // execution state.
        provider_->reset_conversation_state();
    }
    ensure_system_prompt();
    history_.insert(
        history_.end(),
        std::make_move_iterator(plan.retained_history.begin()),
        std::make_move_iterator(plan.retained_history.end()));
    refresh_context_window_snapshot_unlocked();
}

void Agent::compact_history_async(
    std::function<void(const std::string&)> text_callback,
    std::function<void()> done_callback,
    HistoryCompactionReason reason) {

    HistoryCompactionPlan plan;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string active_skill_context;
    std::string model;
    std::uint64_t base_revision = 0;
    bool already_compacting = false;
    {
        std::lock_guard lock(history_mutex_);
        already_compacting = compaction_in_progress_;
        if (!already_compacting) {
            plan = HistoryCompactionPlanner::plan(
                history_,
                context_summary_,
                compaction_policy(provider_, active_model_));
            provider = provider_;
            model = active_model_;
            active_skill_context = collect_active_skill_context(history_);
            base_revision = history_revision_;
            compaction_in_progress_ = true;
        }
    }

    if (already_compacting) {
        if (text_callback) {
            text_callback("\n\xe2\x84\xb9  A history compaction is already in progress.\n");
        }
        return;
    }

    launch_compaction_transaction(
        std::move(plan),
        base_revision,
        std::move(active_skill_context),
        std::move(provider),
        std::move(model),
        reason,
        std::move(text_callback),
        std::move(done_callback));
}

void Agent::launch_compaction_transaction(
    HistoryCompactionPlan plan,
    std::uint64_t base_revision,
    std::string active_skill_context,
    std::shared_ptr<core::llm::LLMProvider> provider,
    std::string model,
    HistoryCompactionReason reason,
    std::function<void(const std::string&)> status_log_callback,
    std::function<void()> done_callback) {
    auto self = shared_from_this();
    auto summary_history = plan.summary_history;
    history_compactor_.compact_async(
        HistoryCompactionRequest{
            .history = std::move(summary_history),
            .provider = std::move(provider),
            .model = std::move(model),
            .reason = reason,
        },
        HistoryCompactionCallbacks{
            .on_status = std::move(status_log_callback),
            .on_summary =
                [self,
                 base_revision,
                 plan = std::move(plan),
                 active_skill_context = std::move(active_skill_context),
                 done_callback = std::move(done_callback)](std::string summary) mutable {
                bool applied = false;
                {
                    std::lock_guard lock(self->history_mutex_);
                    if (self->history_revision_ == base_revision) {
                        self->apply_compaction_unlocked(
                            std::move(summary),
                            std::move(plan),
                            std::move(active_skill_context));
                        applied = true;
                    }
                }
                if (applied && done_callback) {
                    done_callback();
                }
                return applied
                    ? HistoryCompactionApplyStatus::Applied
                    : HistoryCompactionApplyStatus::Stale;
            },
            .on_finished = [self]() {
                std::lock_guard lock(self->history_mutex_);
                self->compaction_in_progress_ = false;
            },
        });
}

void Agent::undo_last() {
    std::lock_guard lock(history_mutex_);
    if (turn_in_progress_.load(std::memory_order_acquire)) {
        return;
    }
    const auto last_user = std::find_if(
        history_.rbegin(),
        history_.rend(),
        [](const core::llm::Message& message) {
            return message.role == "user" && !message.synthetic;
        });
    if (last_user == history_.rend()) {
        return;
    }
    const auto erase_from = std::prev(last_user.base());
    history_.erase(erase_from, history_.end());
    ++conversation_generation_;
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
        if (it->role == "user" && !it->synthetic) {
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
            // `read` can fetch HTTP(S), so it advertises openWorldHint. Local
            // files and result:// snapshots must stay auto-approved as plain
            // reads always were; only network URLs inherit the open-world gate.
            if (permission_required
                && !ann.destructive_hint
                && core::tools::names::is_read_tool(tool_name)) {
                const auto options = core::tools::read::parse_options(args);
                const bool network = options.has_value()
                    && std::ranges::any_of(options->paths, [](const std::string& path) {
                        return path.starts_with("http://") || path.starts_with("https://");
                    });
                if (!network) permission_required = false;
            }
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

    // PermissionGate exposes one pending prompt at a time. Parallel read-only
    // workers may still request open-world access, so keep those prompts ordered.
    std::lock_guard permission_lock(permission_check_mutex_);
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
        turn_failed_.store(true, std::memory_order_release);
        done_callback();
        return;
    }

    std::shared_ptr<core::power::SleepInhibitionLease> sleep_inhibition;
    try {
        sleep_inhibition = sleep_inhibitor_->inhibit("Filo is working on an agent turn");
    } catch (const std::exception& error) {
        core::logging::warn("Could not prevent idle system sleep: {}", error.what());
    } catch (...) {
        core::logging::warn("Could not prevent idle system sleep: unknown error");
    }

    auto completion = std::make_shared<TurnCompletionState>(
        std::move(sleep_inhibition));
    auto finish_turn = [this,
                        done_callback = std::move(done_callback),
                        completion = std::move(completion)]() mutable {
        auto sleep_inhibition = completion->claim_completion();
        if (!sleep_inhibition.has_value()) {
            return;
        }

        // The winning callback now owns the assertion through completion.
        // Any callback copies that outlive the turn retain only empty state.
        turn_in_progress_.store(false, std::memory_order_release);
        done_callback();
    };

    clear_stop_request();  // Reset cancellation flag for new turn
    turn_failed_.store(false, std::memory_order_release);  // Reset outcome for new turn
    auto turn_state = std::make_shared<TurnState>();
    if (user_message.role.empty()) {
        user_message.role = "user";
    }
    const core::llm::Message hook_message = user_message;
    const std::string submitted_text = !user_message.input_text.empty()
        ? user_message.input_text : core::llm::message_text_for_display(user_message);
    const auto boost_task = BoostPipeline::command_task(submitted_text);
    if (boost_task && boost_task->empty()) {
      text_callback("Usage: /boost <task> — deep reasoning and independent verification for one turn.");
      finish_turn();
      return;
    }
    const std::string objective = boost_task ? std::string(*boost_task)
        : core::llm::message_text_for_display(user_message);
    std::string mode_snapshot;
    const auto session_context = session_context_snapshot();
    auto project_facts = core::context::capture_project_facts(session_context);
    const std::string auto_repository_context =
        project_facts.has_value()
            ? core::context::render_project_facts(*project_facts)
            : std::string{};
    std::optional<AutoModeContext> auto_context;
    {
        std::lock_guard lock(history_mutex_);
        ensure_system_prompt();
        append_project_facts_update_unlocked(std::move(project_facts));
        mode_snapshot = std::string(to_string(current_mode_));
        capture_turn_provider_snapshot_unlocked(*turn_state, turn_callbacks);
        if (is_auto_mode(current_mode_) || boost_task.has_value()) {
          const auto user_count = std::ranges::count_if(
              history_, [](const core::llm::Message &message) {
                return message.role == "user" && !message.synthetic;
              });
          const bool has_tool_history = std::ranges::any_of(
              history_, [](const core::llm::Message &message) {
                return message.role == "tool";
              });
          auto_context = AutoModeContext{
              .history_tokens = context_window_snapshot_.estimated_context_tokens,
              .turn_count = static_cast<int>(user_count),
              .has_tool_history = has_tool_history,
              .boost_requested = boost_task.has_value(),
          };
        }
        history_.push_back(std::move(user_message));
        refresh_context_window_snapshot_unlocked();
        consecutive_failure_rounds_ = 0;  // reset loop breaker on new user input
        turn_state->max_steps = sanitize_max_steps_per_turn(loop_limits_.max_steps_per_turn);
    }
    if (auto_context.has_value()) {
      turn_state->auto_turn = auto_turn_coordinator_.start(
          objective,
          *auto_context,
          session_context.workspace_view().primary());
    }
    core::hooks::dispatch(
        core::hooks::HookEvent::UserPromptSubmit,
        build_user_prompt_hook_payload(hook_message, mode_snapshot),
        session_context);
    if (turn_state->auto_turn &&
        turn_callbacks.on_status_log) {
      turn_callbacks.on_status_log(std::format(
          "\n[AUTO · {}]\n",
          auto_turn_coordinator_.decision(*turn_state->auto_turn).reason));
    }
    if (turn_state->auto_turn) {
      const bool orchestrated =
          auto_turn_coordinator_.decision(*turn_state->auto_turn).path ==
          AutoExecutionPath::Orchestrated;
      if (orchestrated && turn_state->provider &&
          turn_callbacks.on_status_log) {
        turn_callbacks.on_status_log(
            "\n[AUTO · compiling goal graph and running read-only frontier]\n");
      }
      auto provider = turn_state->provider;
      const std::string provider_name = turn_state->provider_name;
      const std::string model = turn_state->model;

      const auto findings = auto_turn_coordinator_.prepare(
          *turn_state->auto_turn,
          objective,
          auto_repository_context,
          session_context.workspace_view().primary(),
          make_auto_graph_hooks(
              std::move(provider), provider_name, model, session_context,
              turn_callbacks.on_subagent_event,
              auto_turn_coordinator_.decision(*turn_state->auto_turn).boost,
              turn_callbacks.allowed_tools,
              turn_callbacks.effort_override.empty() ? get_effort_level() : turn_callbacks.effort_override));
      if (orchestrated && turn_state->provider &&
          turn_callbacks.on_status_log) {
        const auto &decision =
            auto_turn_coordinator_.decision(*turn_state->auto_turn);
        turn_callbacks.on_status_log(
            decision.boost
                ? std::format("\n[BOOST · {} · {} investigation(s) and {} isolated "
                              "candidate(s) completed]\n",
                              decision.reason, findings,
                              auto_turn_coordinator_.candidate_count(
                                  *turn_state->auto_turn))
                : std::format(
                      "\n[AUTO · graph ready; {} read-only node(s) completed]\n",
                      findings));
      }
    }
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
        capture_turn_provider_snapshot_unlocked(*turn_state, turn_callbacks);
    }

    if (!is_turn_current(turn_state) || is_stop_requested()) {
        done_callback();
        return;
    }

    if (turn_state->transport_turn_id.empty()) {
        const auto sequence = next_transport_turn_id_.fetch_add(1, std::memory_order_relaxed);
        turn_state->transport_turn_id = std::format("{}:{}", session_context_snapshot().session_id, sequence);
    }

    if (turn_state->auto_turn &&
        auto_turn_coordinator_.decision(*turn_state->auto_turn).boost) {
      turn_state->max_steps = turn_state->max_steps > 0
          ? std::min(turn_state->max_steps, BoostPipeline::kMaxSteps)
          : BoostPipeline::kMaxSteps;
    }
    if (turn_state->max_steps > 0 && turn_state->steps_taken >= turn_state->max_steps) {
        turn_failed_.store(true, std::memory_order_release);
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
    bool stale_turn = false;
    {
        std::lock_guard lock(history_mutex_);
        if (turn_state->conversation_generation != conversation_generation_) {
            stale_turn = true;
        } else {
            request.messages = history_;
            request.model = turn_state->model;
            request.effort = effort_level_;
            if (!turn_callbacks.effort_override.empty()) {
                request.effort = turn_callbacks.effort_override;
            }
            if (turn_state->auto_turn &&
                auto_turn_coordinator_.decision(*turn_state->auto_turn).boost) {
              // Copy first: the returned view can reference request.effort's
              // own buffer when the selected effort already exceeds "high".
              request.effort =
                  std::string(BoostPipeline::reasoning_effort(request.effort));
            }
            if (turn_callbacks.max_tokens_override.has_value()) {
                request.max_tokens = turn_callbacks.max_tokens_override;
            }
            if (turn_callbacks.response_format_override.has_value()) {
                request.response_format = *turn_callbacks.response_format_override;
            }
            request.session_id = step_session_context.session_id;
            request.transport_turn_id = turn_state->transport_turn_id;
            provider = turn_state->provider;
            mode_snapshot = std::string(to_string(current_mode_));
            dynamic_prompt_suffix = build_dynamic_prompt_suffix();
            if (turn_state->auto_turn) {
              dynamic_prompt_suffix +=
                  self->auto_turn_coordinator_.prompt_suffix(
                      *turn_state->auto_turn);
            }
            if (!turn_state->prompt_plan.has_value()) {
                auto plan = stable_prompt_plan_;
                plan.append(core::context::ContextLayer{
                    .kind = core::context::ContextLayerKind::ConversationState,
                    .stability = core::context::PromptStability::Dynamic,
                    .name = "conversation_state",
                    .content = std::move(dynamic_prompt_suffix),
                });
                turn_state->prompt_plan = std::move(plan);
            }
            request.prompt_plan = *turn_state->prompt_plan;
            provider_name_snapshot = turn_state->provider_name;
        }
    }
    if (stale_turn) {
        done_callback();
        return;
    }
    if (const auto invalid = invalid_tool_history(request.messages);
        invalid.has_value()) {
        text_callback(std::format(
            "\n[Internal history error: Filo blocked a malformed tool transcript before it reached the provider: {}]\n",
            invalid->reason));
        turn_failed_.store(true, std::memory_order_release);
        done_callback();
        return;
    }
    if (!provider) {
        text_callback("\n[Error: no active provider configured]\n");
        turn_failed_.store(true, std::memory_order_release);
        done_callback();
        return;
    }
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
        if (tool_is_allowed_for_turn(core::tools::names::kWriteTodos, turn_callbacks)) {
            request.tools.push_back(core::llm::Tool{
                .function = todo_tool_.get_definition(),
            });
        }
        if (tool_is_allowed_for_turn(core::tools::names::kReadToolResult, turn_callbacks)) {
            request.tools.push_back(core::llm::Tool{
                .function = read_tool_result_tool_.get_definition(),
            });
        }
    }

    if (turn_callbacks.on_step_begin) {
        turn_callbacks.on_step_begin();
    }

    // In PLAN/RESEARCH mode strip write-destructive tools
    if (is_read_only_mode(agent_mode_from_string(mode_snapshot))) {
      std::erase_if(request.tools, [](const core::llm::Tool &t) {
        return core::tools::names::is_write_destructive_tool(t.function.name);
      });
    }

    auto assistant_response   = std::make_shared<std::string>();
    auto tool_calls_accum     = std::make_shared<std::vector<core::llm::ToolCall>>();
    auto tool_cost_attribution =
        std::make_shared<std::vector<std::pair<int32_t, int64_t>>>();
    auto reasoning_accum      = std::make_shared<std::string>();
    auto reasoning_protocol_accum = std::make_shared<std::string>();
    auto continuation_accum   =
        std::make_shared<std::vector<core::llm::ContinuationItem>>();
    auto already_stopped      = std::make_shared<std::atomic<bool>>(false);

    auto on_stream_chunk =
        [self, provider, assistant_response, tool_calls_accum, reasoning_accum,
         reasoning_protocol_accum,
         continuation_accum,
         tool_cost_attribution, text_callback, tool_callback, done_callback, turn_callbacks,
         already_stopped, turn_state,
         step_session_context, provider_name_snapshot](
             const core::llm::StreamChunk& chunk) {

        if (already_stopped->load(std::memory_order_acquire)) {
            return;
        }
        if (!self->is_turn_current(turn_state)) {
            if (!already_stopped->exchange(true, std::memory_order_acq_rel)) {
                done_callback();
            }
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
                stopped_msg.reasoning_protocol = *reasoning_protocol_accum;
                stopped_msg.continuation_items = *continuation_accum;
                self->apply_to_history_if_turn_current(
                    turn_state,
                    [&](std::vector<core::llm::Message>& history) {
                        history.push_back(std::move(stopped_msg));
                    });
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
        
        // Accumulate provider reasoning separately from visible answer text.
        // Protocol provenance controls whether it may be replayed upstream.
        if (!chunk.reasoning_content.empty()) {
            core::logging::debug("[Agent] Accumulating reasoning_content: '{}'", chunk.reasoning_content.substr(0, 50));
            *reasoning_accum += chunk.reasoning_content;
            if (reasoning_protocol_accum->empty()) {
                *reasoning_protocol_accum = chunk.reasoning_protocol;
            } else if (!chunk.reasoning_protocol.empty()
                       && *reasoning_protocol_accum != chunk.reasoning_protocol) {
                // A single assistant response must come from one wire protocol.
                // If a custom transport violates that invariant, retain the
                // reasoning for display but disable upstream replay.
                *reasoning_protocol_accum = "mixed";
            }
            if (turn_callbacks.on_reasoning) {
                turn_callbacks.on_reasoning(chunk.reasoning_content);
            }
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
        // A transport must have exactly one terminal event. Some cancellation
        // and retry paths can race a final marker with an error marker; only
        // the first one may persist history or launch tools.
        if (already_stopped->exchange(true, std::memory_order_acq_rel)) {
            return;
        }

        // ── Final chunk ──────────────────────────────────────────────────
        // Record API call outcome (is_error is true for HTTP 4XX/5XX or connection errors)
        self->session_stats_registry_->record_api_call(
            step_session_context.session_id,
            !chunk.is_error);
        if (chunk.is_error) {
            self->turn_failed_.store(true, std::memory_order_release);
        }
        if (chunk.authentication_recovery.has_value()
            && turn_callbacks.on_authentication_required) {
            turn_callbacks.on_authentication_required(
                *chunk.authentication_recovery);
        }

        // Record token usage in the ledger and session stats.
        {
            auto usage = provider->get_last_usage();
            std::string model = provider->get_last_model();
            const bool should_estimate_cost = provider->should_estimate_cost();
            if (model.empty()) {
                model = turn_state->model;
            }
            const std::string ledger_actor = turn_callbacks.ledger_actor.empty()
                ? std::string("agent")
                : turn_callbacks.ledger_actor;
            if (usage.has_data()) {
                self->budget_tracker_->record_event({
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
            self->session_stats_registry_->record_turn(
                step_session_context.session_id,
                model,
                usage,
                should_estimate_cost);

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
        // A stopped or failed attempt cannot commit provisional tool calls or
        // opaque continuation state. Replaying either would make the next turn
        // depend on a model response that never completed successfully.
        const bool discard_uncommitted_state =
            self->is_stop_requested() || chunk.is_error;
        asst_msg.tool_calls         = discard_uncommitted_state
            ? std::vector<core::llm::ToolCall>{}
            : *tool_calls_accum;
        asst_msg.reasoning_content  = *reasoning_accum;
        asst_msg.reasoning_protocol = *reasoning_protocol_accum;
        asst_msg.continuation_items = discard_uncommitted_state
            ? std::vector<core::llm::ContinuationItem>{}
            : *continuation_accum;

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

            if (!self->apply_to_history_if_turn_current(
                    turn_state,
                    [&](std::vector<core::llm::Message>& history) {
                        if (!asst_msg.content.empty()
                            || !asst_msg.reasoning_content.empty()
                            || !asst_msg.continuation_items.empty()) {
                            history.push_back(asst_msg);
                        }
                        history.push_back(core::llm::Message{
                            .role = "user",
                            .content = std::string(kMaxOutputRecoveryPrompt),
                            .synthetic = true,
                        });
                    })) {
                done_callback();
                return;
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
        if (should_persist_assistant
            && !self->apply_to_history_if_turn_current(
                turn_state,
                [&](std::vector<core::llm::Message>& history) {
                    history.push_back(asst_msg);
                })) {
            done_callback();
            return;
        }

        // Check if we were stopped - if so, don't proceed to tool execution
        if (self->is_stop_requested()) {
            done_callback();
            return;
        }

        if (chunk.is_error) {
            done_callback();
            return;
        }

        if (tool_calls_accum->empty()) {
          const auto continue_with = [&](std::string_view status,
                                         std::string message) {
            const std::string rendered_status =
                std::format("\n[{}]\n", status);
            if (turn_callbacks.on_status_log) {
              turn_callbacks.on_status_log(rendered_status);
            } else {
              text_callback(rendered_status);
            }
            if (!self->apply_to_history_if_turn_current(
                    turn_state,
                    [&](std::vector<core::llm::Message> &history) {
                      history.push_back(core::llm::Message{
                          .role = "user",
                          .content = std::move(message),
                          .synthetic = true,
                      });
                    })) {
              done_callback();
              return;
            }
            self->step(text_callback, tool_callback, done_callback,
                       turn_callbacks, turn_state);
          };
          const auto fail_with = [&](std::string message) {
            const std::string warning = std::format("\n\n[{}]", message);
            self->turn_failed_.store(true, std::memory_order_release);
            text_callback(warning);
            static_cast<void>(self->apply_to_history_if_turn_current(
                turn_state, [&](std::vector<core::llm::Message> &history) {
                  history.push_back(core::llm::Message{
                      .role = "assistant",
                      .content = warning,
                      .synthetic = true,
                  });
                }));
          };

          const bool mutation_observed = turn_state->auto_turn &&
              self->auto_turn_coordinator_.mutation_observed(
                  *turn_state->auto_turn);
          const auto context = self->session_context_snapshot();
          const auto run_completion_hooks = [&]() {
            return core::hooks::evaluate_completion(
                build_stop_hook_payload(
                    asst_msg, self->get_mode(), mutation_observed),
                context, turn_state->completion_hooks);
          };

          if (turn_state->auto_turn &&
              self->auto_turn_coordinator_.decision(*turn_state->auto_turn).boost &&
              turn_callbacks.on_status_log)
            turn_callbacks.on_status_log("\n[BOOST · checking evidence and independent verification]\n");
          const auto completion = turn_state->auto_turn
              ? self->auto_turn_coordinator_.evaluate_completion(
                    *turn_state->auto_turn,
                    asst_msg.content,
                    context.workspace_view().primary(),
                    run_completion_hooks,
                    [self, turn_callbacks](std::string_view recipe_id,
                                           const std::filesystem::path &root)
                        -> std::expected<core::verification::Receipt, std::string> {
                      if (!tool_is_allowed_for_turn(core::tools::names::kRunVerification, turn_callbacks))
                        return std::unexpected("Verification is excluded by this turn's tool allowlist.");
                      return self->run_verification_recipe(recipe_id, root);
                    })
              : run_completion_hooks();
          if (!completion.status.empty() &&
              completion.action ==
                  core::session::TurnCompletionAction::Complete &&
              turn_callbacks.on_status_log) {
            turn_callbacks.on_status_log(
                std::format("\n[{}]\n", completion.status));
          }
          if (completion.action ==
              core::session::TurnCompletionAction::Continue) {
            turn_state->prompt_plan.reset();
            continue_with(completion.status, completion.message);
            return;
          }
          if (completion.action ==
              core::session::TurnCompletionAction::Fail) {
            fail_with(completion.message);
            done_callback();
            return;
          }
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
                     step_session_context,
                     provider_name_snapshot]() {
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
            std::vector<bool> read_only_tasks(tool_calls_accum->size(), false);
            std::vector<DeniedReason> denied_reasons(tool_calls_accum->size(), DeniedReason::None);
            std::vector<std::string> hook_denial_reasons(tool_calls_accum->size());
            std::vector<std::string> argument_errors(tool_calls_accum->size());
            // Structured argument-issue bookkeeping for recovery hints.
            std::vector<std::string> argument_parameters(tool_calls_accum->size());
            std::vector<std::string> argument_recovery_hints(tool_calls_accum->size());
            // Resolved definitions reused by the scheduler pass for recovery
            // bookkeeping (empty where the tool was never found).
            std::vector<std::optional<core::tools::ToolDefinition>> resolved_definitions(
                tool_calls_accum->size());
            std::string permission_mode;
            {
                std::lock_guard lock(self->history_mutex_);
                permission_mode = std::string(to_string(self->current_mode_));
            }
            bool denied_any = false;
            for (size_t i = 0; i < tool_calls_accum->size(); ++i) {
                if (self->is_stop_requested() || !self->is_turn_current(turn_state)) {
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
                read_only_tasks[i] =
                    tc.function.name == SubagentOrchestrator::kTaskToolName
                    && self->orchestrator_.task_is_read_only(
                        tc.function.arguments,
                        permission_mode);
                approved[i] = hook_decision.approved
                    || read_only_tasks[i]
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
                } else if (tc.function.name == core::tools::names::kWriteTodos) {
                    definition = self->todo_tool_.get_definition();
                } else if (tc.function.name == core::tools::names::kReadToolResult) {
                    definition = self->read_tool_result_tool_.get_definition();
                } else {
                    definition = self->skill_manager_.get_tool_definition(tc.function.name);
                }
                resolved_definitions[i] = definition;
                if (!definition.has_value()) {
                    approved[i] = false;
                    denied_reasons[i] = DeniedReason::InvalidArguments;
                    argument_errors[i] = "tool not found";
                    tool_callback(tc.function.name, "[invalid tool call: tool not found]");
                    continue;
                }
                auto normalized = core::tools::schema::validate_arguments(
                    *definition,
                    tc.function.arguments);
                if (!normalized.has_value()) {
                    approved[i] = false;
                    denied_reasons[i] = DeniedReason::InvalidArguments;
                    argument_errors[i] = normalized.error().message;
                    argument_parameters[i] = normalized.error().parameter;
                    if (auto hint = self->note_argument_validation_failure(
                            tc.function.name,
                            *definition,
                            normalized.error(),
                            tc.function.arguments,
                            *turn_state)) {
                        argument_recovery_hints[i] = std::move(*hint);
                    }
                    tool_callback(
                        tc.function.name,
                        "[invalid tool arguments: " + normalized.error().message + "]");
                    continue;
                }
                tc.function.arguments = std::move(*normalized);
            }

            if (self->is_stop_requested() || !self->is_turn_current(turn_state)) {
                done_callback();
                return;
            }

            if (turn_state->auto_turn) {
              std::vector<AutoToolIntent> intents;
              intents.reserve(tool_calls_accum->size());
              for (std::size_t i = 0; i < tool_calls_accum->size(); ++i) {
                const auto &call = (*tool_calls_accum)[i];
                intents.push_back(AutoToolIntent{
                    .name = call.function.name,
                    .arguments = call.function.arguments,
                    .approved = approved[i],
                    .read_only_subagent = read_only_tasks[i],
                    .destructive_hint =
                        resolved_definitions[i].has_value() &&
                        resolved_definitions[i]->annotations.destructive_hint,
                });
              }
              const auto writer =
                  self->auto_turn_coordinator_.prepare_tool_batch(
                      *turn_state->auto_turn,
                      step_session_context.workspace_view().primary(),
                      intents,
                      [self] { return self->is_stop_requested(); });
              if (turn_callbacks.on_status_log) {
                switch (writer) {
                case WorkspaceWriterState::Acquired:
                  turn_callbacks.on_status_log(
                      "\n[AUTO · exclusive repository writer lease "
                      "acquired]\n");
                  break;
                case WorkspaceWriterState::Unavailable:
                  // The batch still runs — the lease is a coordination aid,
                  // not an authorization check — but the user must never be
                  // told exclusion was held when it was not.
                  turn_callbacks.on_status_log(
                      "\n[AUTO · warning: mutating step is running without an "
                      "exclusive repository lease]\n");
                  break;
                case WorkspaceWriterState::NotRequired:
                  break;
                }
              }
            }

            // 2. Execute approved calls through a resource-aware scheduler.
            std::shared_ptr<core::llm::LLMProvider> provider_for_task;
            std::string active_provider_name_for_task;
            std::string active_model_for_task;
            std::string parent_mode_for_task;
            {
                std::lock_guard lock(self->history_mutex_);
                provider_for_task = turn_state->provider;
                active_provider_name_for_task = turn_state->provider_name;
                if (active_provider_name_for_task.empty()) {
                    active_provider_name_for_task = provider_name_snapshot;
                }
                active_model_for_task = turn_state->model;
                parent_mode_for_task =
                    std::string(to_string(self->current_mode_));
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
                    std::string error_payload;
                    if (invalid_arguments) {
                        // Structured rejection payload: the offending parameter
                        // and a recovery hint give the model an immediate,
                        // actionable correction path.
                        core::utils::JsonWriter payload_writer(96);
                        {
                            auto payload = payload_writer.object();
                            payload_writer.kv_str(
                                "error",
                                "Invalid tool arguments: " + argument_errors[i]);
                            if (!argument_parameters[i].empty()) {
                                payload_writer.comma().kv_str(
                                    "parameter", argument_parameters[i]);
                            }
                            if (!argument_recovery_hints[i].empty()) {
                                payload_writer.comma().kv_str(
                                    "recovery_hint", argument_recovery_hints[i]);
                            }
                        }
                        error_payload = std::move(payload_writer).take();
                    } else if (skipped_after_denial) {
                        error_payload =
                            R"({"error":"Tool call skipped after a previous denial in this step."})";
                    } else if (blocked_by_turn_allow_list) {
                        error_payload =
                            R"({"error":"Tool call blocked by the turn tool allow-list."})";
                    } else if (blocked_by_hook) {
                        error_payload = std::format(
                            R"({{"error":"Tool call blocked by PreToolUse hook: {}"}})",
                            core::utils::escape_json_string(hook_denial_reasons[i]));
                    } else {
                        error_payload = R"({"error":"Tool call denied by user."})";
                    }
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
                if (read_only_tasks[i]) {
                    planned.accesses = read_all_tool_access();
                }
                scheduled_tasks.push_back({
                    .accesses = std::move(planned.accesses),
                    .run = [self, tc, &tool_callback,
                     dedup_index = dedup_decision.original_index,
                     deduplicator = &turn_state->deduplicator,
                     dedup_stop_requested,
                     recovery_definition = resolved_definitions[i],
                     recovery_turn_state = turn_state.get(),
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
                                .memory_system = self->memory_system_,
                            };

                            result = self->orchestrator_.execute_task(
                                tc.function.arguments,
                                provider_for_task,
                                run_context);
                        } else if (tc.function.name == core::tools::names::kWriteTodos) {
                            result = self->todo_tool_.execute(
                                tc.function.arguments,
                                session_context);
                        } else if (tc.function.name == core::tools::names::kReadToolResult) {
                            result = self->read_tool_result_tool_.execute(
                                tc.function.arguments,
                                session_context);
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
                                    .cancellation_requested = [self] { return self->is_stop_requested(); },
                                    .session_stats = self->session_stats_registry_,
                                });
                        }
                        const bool raw_tool_ok =
                            !recovery::result_indicates_error(result);
                        if (recovery_turn_state->auto_turn) {
                          self->auto_turn_coordinator_.observe_tool(
                              *recovery_turn_state->auto_turn,
                              AutoToolObservation{
                                  .name = tc.function.name,
                                  .arguments = tc.function.arguments,
                                  .result = result,
                                  .succeeded = raw_tool_ok,
                                  .mutation_hint =
                                      recovery_definition.has_value() &&
                                      recovery_definition->annotations
                                          .destructive_hint,
                                  .trusted_verification_receipts =
                                      recovery_definition.has_value() &&
                                      recovery_definition
                                          ->trusted_verification_receipts,
                              });
                        }
                        if (recovery_definition.has_value()) {
                          self->apply_tool_recovery_outcome(
                              tc, *recovery_definition, *recovery_turn_state,
                              raw_tool_ok, result);
                        }
                        if (core::tools::MemoryTool::committed_mutation(
                                tc.function.name,
                                tc.function.arguments,
                                result)) {
                            self->refresh_system_prompt();
                        }
                        const auto history_limits = tool_output_history::limits_for_tool(
                            tc.function.name,
                            static_cast<std::size_t>(std::max(
                                0,
                                core::config::ConfigManager::get_instance()
                                    .get_config()
                                    .tool_output_token_limit)));
                        auto compact_result = tool_output_history::clamp_for_history(
                            tc.function.name,
                            result,
                            history_limits,
                            core::config::ConfigManager::get_instance()
                                .get_config()
                                .context_compression,
                            core::agent::tool_output_history::Context{
                                .tool_arguments = tc.function.arguments,
                                .session_id = session_context.session_id,
                            });
                        if (result.size() > history_limits.max_chars) {
                            if (auto stored = self->tool_result_store_.store(
                                    session_context.session_id,
                                    tc.id,
                                    result);
                                stored.has_value()) {
                                compact_result = ToolResultStore::attach_reference(
                                    std::move(compact_result),
                                    *stored);
                            } else {
                                core::logging::warn(
                                    "[Agent] Could not offload oversized {} result: {}",
                                    tc.function.name,
                                    stored.error());
                            }
                        }
                        result = std::move(compact_result);
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
            core::hooks::dispatch(
                core::hooks::HookEvent::PostToolBatch,
                build_tool_batch_hook_payload(*tool_calls_accum, tool_messages),
                step_session_context);

            const bool stop_requested_after_tools = self->is_stop_requested();
            if (!self->is_turn_current(turn_state)) {
                done_callback();
                return;
            }

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
                self->session_stats_registry_->record_tool_call(
                    step_session_context.session_id,
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
                self->budget_tracker_->record_event({
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
                if (!self->apply_to_history_if_turn_current(
                        turn_state,
                        [&](std::vector<core::llm::Message>& history) {
                            history.push_back(msg);
                        })) {
                    done_callback();
                    return;
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
    // Session snapshots must always be replayable. While tools are running,
    // the live history temporarily ends at an assistant tool-use message; save
    // only the last complete prefix. The same rule prevents legacy corruption
    // from being persisted again.
    if (const auto invalid = invalid_tool_history(result);
        invalid.has_value()) {
        result.resize(invalid->index);
    }
    return result;
}

void Agent::append_history_message(core::llm::Message message) {
    std::lock_guard lock(history_mutex_);
    if (turn_in_progress_.load(std::memory_order_acquire)) {
        core::logging::warn(
            "Ignored out-of-band history append while an agent turn was active.");
        return;
    }
    ensure_system_prompt();
    if (message.role.empty()) {
        message.role = "user";
    }
    history_.push_back(std::move(message));
    mark_stable_prompt_prefix_dirty();
    refresh_context_window_snapshot_unlocked();
}

void Agent::load_history(std::vector<core::llm::Message> messages,
                         const std::string &context_summary,
                         const std::string &mode) {
  std::lock_guard lock(history_mutex_);
  ++conversation_generation_;
  current_mode_ = agent_mode_from_string(mode);
  context_summary_ = context_summary;
  project_facts_snapshot_.reset();
  for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
    if (!is_repository_context_message(*it))
      continue;
    if (const auto state = decode_repository_context_state(it->input_text)) {
      project_facts_snapshot_ = core::context::ProjectFactsSnapshot{
          .status = state->first,
          .tree = state->second,
      };
    }
    break;
  }
  consecutive_failure_rounds_ = 0;
  orchestrator_.clear_sessions();
  reset_efficiency_tracking_unlocked();
  if (provider_) {
    provider_->reset_conversation_state();
  }

  // Remove any stale system message — ensure_system_prompt() inserts a fresh
  // one.
  std::erase_if(messages,
                [](const core::llm::Message &m) { return m.role == "system"; });
  if (const auto invalid = invalid_tool_history(messages);
      invalid.has_value()) {
    core::logging::warn(
        "Truncated malformed saved tool transcript at history index {}: {}",
        invalid->index, invalid->reason);
    messages.resize(invalid->index);
  }
  history_ = std::move(messages);
  refresh_stable_prompt_state_unlocked();
}

std::string Agent::get_mode() const {
    std::lock_guard lock(history_mutex_);
    return std::string(to_string(current_mode_));
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

std::optional<std::string> Agent::note_argument_validation_failure(
    const std::string& tool_name,
    const core::tools::ToolDefinition& definition,
    const core::tools::schema::ArgumentIssue& issue,
    const std::string& failed_arguments,
    TurnState& turn_state) {
    // Bookkeeping first: a second distinct failure for the same tool within
    // one step makes the correction ambiguous and suppresses learning.
    {
        std::lock_guard lock(turn_state.recovery_mutex);
        auto& pending = turn_state.pending_validation_recoveries;
        const auto it = pending.find(tool_name);
        if (it != pending.end() && it->second.step == turn_state.steps_taken) {
            it->second.ambiguous = true;
        } else {
            pending.insert_or_assign(tool_name, PendingValidationRecovery{
                .issue = issue,
                .failed_arguments = failed_arguments,
                .step = turn_state.steps_taken,
                .ambiguous = false,
            });
        }
    }

    // A proven cross-session lesson outranks on-the-spot deduction.
    const auto key = recovery::make_key(tool_name, definition, issue);
    if (auto remembered = memory_system_->tool_recovery().recall(key)) {
        return remembered;
    }
    if (auto deduced = recovery::advise(issue, definition)) {
        return deduced->instruction;
    }
    return std::nullopt;
}

void Agent::apply_tool_recovery_outcome(const core::llm::ToolCall& call,
                                        const core::tools::ToolDefinition& definition,
                                        TurnState& turn_state,
                                        bool executed_ok,
                                        std::string& result) {
    if (!executed_ok) {
        // Surface lessons already proven for this tool contract so the model
        // gets them with the failing result itself.
        const std::vector<std::string> hints =
            memory_system_->tool_recovery().recall_runtime_hints(
                call.function.name,
                recovery::schema_fingerprint(definition));
        if (!hints.empty()) {
            std::string joined;
            for (std::size_t i = 0; i < hints.size(); ++i) {
                if (i > 0) joined += ' ';
                joined += hints[i];
            }
            result = recovery::augment_error_payload(result, joined);
        }
        std::lock_guard lock(turn_state.recovery_mutex);
        auto& pending = turn_state.pending_runtime_recoveries;
        const auto it = pending.find(call.function.name);
        if (it != pending.end() && it->second.step == turn_state.steps_taken) {
            // Another call to this tool already failed in this model step.
            // The last finisher is not a privileged candidate.
            it->second.ambiguous = true;
        } else {
            pending.insert_or_assign(
                call.function.name,
                PendingRuntimeRecovery{
                    .failed_arguments = call.function.arguments,
                    .step = turn_state.steps_taken,
                    .ambiguous = false,
                });
        }
        return;
    }

    // Success closes the fail→success observation loop for this tool. An
    // observation only counts while it is recent: pairing a failure with a
    // success many steps later would attribute an unrelated call's arguments
    // to the earlier mistake and manufacture a bogus lesson.
    const int current_step = turn_state.steps_taken;
    const auto is_fresh = [current_step](int observed_step) {
        return current_step - observed_step
            <= recovery::kMaxObservationStepDistance;
    };

    std::optional<PendingValidationRecovery> pending_validation;
    std::optional<PendingRuntimeRecovery> pending_runtime;
    {
        std::lock_guard lock(turn_state.recovery_mutex);
        if (const auto it =
                turn_state.pending_validation_recoveries.find(call.function.name);
            it != turn_state.pending_validation_recoveries.end()) {
            if (is_fresh(it->second.step)) {
                pending_validation = it->second;
            }
            turn_state.pending_validation_recoveries.erase(it);
        }
        if (const auto it =
                turn_state.pending_runtime_recoveries.find(call.function.name);
            it != turn_state.pending_runtime_recoveries.end()) {
            // A same-step success is a sibling, not a correction. Leave the
            // failure so a later model step can close the loop; drop only
            // stale observations.
            if (is_fresh(it->second.step) && current_step > it->second.step) {
                pending_runtime = it->second;
                turn_state.pending_runtime_recoveries.erase(it);
            } else if (!is_fresh(it->second.step)) {
                turn_state.pending_runtime_recoveries.erase(it);
            }
        }
    }

    if (pending_validation.has_value() && !pending_validation->ambiguous) {
        if (auto lesson = recovery::derive_validation_lesson(
                call.function.name,
                definition,
                pending_validation->issue,
                pending_validation->failed_arguments,
                call.function.arguments)) {
            memory_system_->tool_recovery().record(*lesson);
        }
    }
    if (pending_runtime.has_value() && !pending_runtime->ambiguous) {
        if (auto lesson = recovery::derive_runtime_lesson(
                call.function.name,
                definition,
                pending_runtime->failed_arguments,
                call.function.arguments)) {
            memory_system_->tool_recovery().record(*lesson);
        }
    }
}

void Agent::check_auto_compact(std::function<void(const std::string&)> status_log_callback) {
    // Only check compaction when turn is complete (not in the middle of multi-step execution)
    // This is called from:
    // 1. After assistant text response (no tool calls) - OK to check
    // 2. After tool execution before next step - NOT OK, let the loop continue
    // The caller must ensure we're at a natural conversation boundary.
    
    int threshold = 0;
    HistoryCompactionPlan plan;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string active_skill_context;
    std::string model;
    std::uint64_t base_revision = 0;
    core::context::CompactionDecision compaction;
    {
        std::lock_guard lock(history_mutex_);
        if (compaction_in_progress_) return;
        threshold = auto_compact_threshold_;
        const bool use_model_aware_default_threshold =
            auto_compact_uses_model_window_default_;
        provider = provider_;
        model = active_model_;
        compaction = core::context::ContextWindowTracker::compaction_decision(
            history_,
            provider,
            model,
            core::context::CompactionTriggerPolicy{
                .configured_token_threshold = threshold,
                .use_model_aware_default_threshold =
                    use_model_aware_default_threshold,
            });
        if (threshold > 0 && compaction.should_compact) {
            plan = HistoryCompactionPlanner::plan(
                history_,
                context_summary_,
                compaction_policy(provider_, active_model_));
            active_skill_context = collect_active_skill_context(history_);
            base_revision = history_revision_;
            compaction_in_progress_ = true;
        }
    }

    // Auto-compact disabled
    if (threshold <= 0) return;

    if (!compaction.should_compact) return;

    launch_compaction_transaction(
        std::move(plan),
        base_revision,
        std::move(active_skill_context),
        std::move(provider),
        std::move(model),
        HistoryCompactionReason::Auto,
        std::move(status_log_callback));
}

void Agent::update_session_context(std::function<void(core::context::SessionContext&)> modifier) {
    if (!modifier) {
        return;
    }
    std::lock_guard lock(history_mutex_);
    modifier(session_context_);
    mark_stable_prompt_prefix_dirty();
}

} // namespace core::agent
