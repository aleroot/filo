#include "DelegatedAgentRunner.hpp"

#include "AgentMode.hpp"
#include "ToolRecovery.hpp"
#include "../config/ConfigManager.hpp"
#include "../memory/MemorySystem.hpp"
#include "../utils/DetachedWorkTracker.hpp"

#include <atomic>
#include <condition_variable>
#include <thread>
#include <mutex>

namespace core::agent {

namespace {

struct RunState {
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
    bool turn_returned = false;
    bool settled_signalled = false;
    std::promise<void> settled;
    std::shared_ptr<void> execution_lease;

    void finish(bool returned) {
        std::lock_guard lock(done_mutex);
        if (returned) turn_returned = true;
        else done = true;
        if (done && turn_returned && !settled_signalled) {
            settled_signalled = true;
            execution_lease.reset();
            settled.set_value();
        }
        done_cv.notify_all();
    }

    std::mutex streamed_mutex;
    std::string streamed_text;
    std::string final_text;
    bool has_final_response = false;
    bool threw = false;

    std::mutex event_mutex;
    bool events_closed = false;

    std::atomic<int> steps{0};
    std::atomic<int> tool_calls_total{0};
    std::atomic<int> tool_calls_success{0};
    std::atomic<int> failed_tool_calls{0};
    std::atomic<bool> budget_exhausted{false};
    std::atomic<bool> hard_step_limit{false};
    std::atomic<bool> stop_requested{false};
};

[[nodiscard]] std::string build_delegated_prompt(const DelegatedAgentRunner::Request& request) {
    if (request.max_steps <= 0 && request.worker_name.empty()
        && request.worker_description.empty()
        && request.worker_prompt.empty()) {
        return request.prompt;
    }

    std::string prompt;
    prompt.reserve(
        request.prompt.size()
        + request.worker_name.size()
        + request.worker_description.size()
        + request.worker_prompt.size()
        + 128);

    prompt += "[Delegated worker]\n";
    if (!request.worker_name.empty()) {
        prompt += "Profile: @";
        prompt += request.worker_name;
        prompt += '\n';
    }
    if (!request.worker_description.empty()) {
        prompt += "Description: ";
        prompt += request.worker_description;
        prompt += '\n';
    }
    if (!request.worker_prompt.empty()) {
        prompt += "Instructions:\n";
        prompt += request.worker_prompt;
        prompt += "\n\n";
    }
    if (request.max_steps > 0) {
        prompt += "Budget: " + std::to_string(request.max_steps)
            + " model steps, including a final step reserved for your summary. "
              "Batch independent searches and reads. Prioritize the task's main question "
              "and return partial findings if the budget is insufficient.\n\n";
    }
    prompt += "[Task]\n";
    prompt += request.prompt;
    return prompt;
}

} // namespace

DelegatedAgentRunner::Result DelegatedAgentRunner::run(Request request) {
    Result result;

    auto agent = std::make_shared<core::agent::Agent>(
        request.provider,
        request.tool_manager,
        request.session_context,
        ToolResultStore::default_root(),
        std::shared_ptr<core::power::SleepInhibitor>{},
        request.session_stats_registry,
        nullptr,
        // Prefer the parent's memory substrate so subagents learn into the
        // same stores. Standalone callers (TaskService) get a fresh system
        // from config; the file-backed ports are transactional, so they merge.
        request.memory_system
            ? request.memory_system
            : core::memory::make_memory_system(core::memory::MemoryConfig{
                  .tool_recovery = core::config::ConfigManager::get_instance()
                                       .get_config()
                                       .tool_recovery}),
        request.workspace_leases,
        request.change_tracker);
    agent->set_execution_journal(request.execution_journal,
        request.task_id.empty() ? "worker_" + core::session::SessionStore::generate_id() : request.task_id,
        request.parent_tool_call_id);
    agent->set_active_provider_name(request.provider_name);
    // Last line of defence: AUTO is a parent-transaction mode. If a caller
    // still forwards it, coerce to BUILD so the child cannot re-enter leases.
    agent->set_mode(is_auto_mode(agent_mode_from_string(request.mode))
                        ? std::string(to_string(AgentMode::Build))
                        : request.mode);
    if (!request.model.empty()) {
        agent->set_active_model(request.model);
    }
    if (request.permission_check) {
        agent->set_permission_fn(std::move(request.permission_check));
    }
    agent->set_loop_limits({
        .max_steps_per_turn = request.max_steps,
    });

    if (request.resume_state.has_value()) {
        agent->load_history(
            request.resume_state->messages,
            request.resume_state->context_summary,
            request.resume_state->mode);
    }
    if (request.on_agent_ready) {
        request.on_agent_ready(agent);
    }

    auto run_state = std::make_shared<RunState>();
    run_state->execution_lease = std::move(request.execution_lease);
    result.settled = run_state->settled.get_future().share();
    auto worker_cancel_requested = [run_state, check = request.cancellation_requested]() {
        return run_state->stop_requested.load(std::memory_order_acquire) || (check && check());
    };
    const std::string delegated_prompt = build_delegated_prompt(request);
    std::string ledger_actor = request.worker_name.empty() ? std::string("subagent") : std::string("subagent:") + request.worker_name;

    auto emit_event = [run_state, callback = request.on_event,
                       parent_tool_call_id = request.parent_tool_call_id,
                       task_id = request.task_id,
                       worker_name = request.worker_name,
                       description = request.task_description,
                       provider_name = request.provider_name,
                       model_name = request.model](SubagentEvent event, bool terminal = false) {
        std::lock_guard lock(run_state->event_mutex);
        if (run_state->events_closed) return;
        if (terminal) run_state->events_closed = true;
        if (!callback) return;
        event.parent_tool_call_id = parent_tool_call_id;
        event.task_id = task_id;
        event.worker_name = worker_name;
        event.description = description;
        event.provider_name = provider_name;
        event.model_name = model_name;
        callback(event);
    };

    emit_event(SubagentEvent{.kind = SubagentEvent::Kind::Started});

    std::thread turn_thread(
        [agent,
         run_state,
         delegated_prompt,
         provider = request.provider,
         provider_name = std::move(request.provider_name),
         model = request.model,
         effort = request.effort,
         response_format = std::move(request.response_format),
         allowed_tools = std::move(request.allowed_tools),
         cancellation_requested = std::move(worker_cancel_requested),
         ledger_actor = std::move(ledger_actor),
         emit_event]() mutable {
          // Counted because this thread is detached below whenever a cancellation
          // or a timeout cannot wait for it, so a caller that is about to destroy
          // shared state has a bounded way to let it finish first.
          const core::utils::DetachedWorkTracker::Guard detached_guard{
              core::utils::DetachedWorkTracker::instance()};
          try {
            agent->send_message(
                delegated_prompt,
                [run_state, emit_event](const std::string& chunk) {
                    {
                        std::lock_guard lock(run_state->streamed_mutex);
                        run_state->streamed_text += chunk;
                    }
                    emit_event(SubagentEvent{
                        .kind = SubagentEvent::Kind::TextDelta,
                        .text_delta = chunk,
                        .steps = run_state->steps.load(std::memory_order_relaxed),
                        .tool_calls = run_state->tool_calls_total.load(std::memory_order_relaxed),
                        .failed_tool_calls = run_state->failed_tool_calls.load(std::memory_order_relaxed),
                    });
                },
                [](const std::string&, const std::string&) {},
                [run_state]() {
                    run_state->finish(false);
                },
                core::agent::Agent::TurnCallbacks{
                    .on_step_begin = [run_state, emit_event]() {
                        const int steps = run_state->steps.fetch_add(1, std::memory_order_relaxed) + 1;
                        emit_event(SubagentEvent{
                            .kind = SubagentEvent::Kind::Progress,
                            .steps = steps,
                            .tool_calls = run_state->tool_calls_total.load(std::memory_order_relaxed),
                            .failed_tool_calls = run_state->failed_tool_calls.load(std::memory_order_relaxed),
                        });
                    },
                    .on_tool_start = [run_state, emit_event](const core::llm::ToolCall& tool_call) {
                        emit_event(SubagentEvent{
                            .kind = SubagentEvent::Kind::ToolStarted,
                            .tool_call_id = tool_call.id,
                            .tool_name = tool_call.function.name,
                            .tool_arguments = tool_call.function.arguments,
                            .steps = run_state->steps.load(std::memory_order_relaxed),
                            .tool_calls = run_state->tool_calls_total.load(std::memory_order_relaxed),
                            .failed_tool_calls = run_state->failed_tool_calls.load(std::memory_order_relaxed),
                        });
                    },
                    .on_tool_finish = [run_state, emit_event](const core::llm::ToolCall& tool_call,
                                                  const core::llm::Message& tool_result) {
                        const int total = run_state->tool_calls_total.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (!recovery::result_indicates_error(tool_result.content)) {
                            run_state->tool_calls_success.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            run_state->failed_tool_calls.fetch_add(1, std::memory_order_relaxed);
                        }
                        emit_event(SubagentEvent{
                            .kind = SubagentEvent::Kind::ToolFinished,
                            .tool_call_id = tool_call.id,
                            .tool_name = tool_call.function.name,
                            .tool_arguments = tool_call.function.arguments,
                            .tool_result = tool_result.content,
                            .steps = run_state->steps.load(std::memory_order_relaxed),
                            .tool_calls = total,
                            .failed_tool_calls = run_state->failed_tool_calls.load(std::memory_order_relaxed),
                        });
                    },
                    .provider_override = std::move(provider),
                    .provider_name_override = std::move(provider_name),
                    .model_override = std::move(model),
                    .effort_override = std::move(effort),
                    .response_format_override = std::move(response_format),
                    .allowed_tools = std::move(allowed_tools),
                    .ledger_actor = std::move(ledger_actor),
                    .allow_efficiency_rotation = false,
                    .reserve_final_step = true,
                    .on_step_budget_event = [run_state](Agent::StepBudgetEvent event) {
                        run_state->budget_exhausted.store(true, std::memory_order_relaxed);
                        if (event == Agent::StepBudgetEvent::Exhausted) {
                            run_state->hard_step_limit.store(true, std::memory_order_relaxed);
                        }
                    },
                    .on_final_response = [run_state](const std::string& text) {
                        std::lock_guard lock(run_state->streamed_mutex);
                        run_state->has_final_response = true;
                        run_state->final_text = text;
                    },
                    .cancellation_requested = std::move(cancellation_requested),
                });
          } catch (const std::exception& error) {
            {
                std::lock_guard lock(run_state->streamed_mutex);
                run_state->threw = true;
                run_state->streamed_text += std::string("Worker execution failed: ") + error.what();
            }
            run_state->finish(false);
          } catch (...) {
            {
                std::lock_guard lock(run_state->streamed_mutex);
                run_state->threw = true;
                run_state->streamed_text += "Worker execution failed with an unknown exception.";
            }
            run_state->finish(false);
          }
          run_state->finish(true);
        });

    {
        std::unique_lock lock(run_state->done_mutex);
        const auto deadline = std::chrono::steady_clock::now() + request.timeout;
        while (!run_state->done) {
            if (request.cancellation_requested && request.cancellation_requested()) {
                result.stop_reason = Result::StopReason::Cancelled;
                run_state->stop_requested.store(true, std::memory_order_release);
                lock.unlock();
                agent->request_stop();
                break;
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                result.stop_reason = Result::StopReason::TimedOut;
                run_state->stop_requested.store(true, std::memory_order_release);
                lock.unlock();
                agent->request_stop();
                break;
            }

            run_state->done_cv.wait_for(lock, std::chrono::milliseconds(100));
        }
    }

    if (turn_thread.joinable()) {
        if (result.settled.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
            turn_thread.join();
        } else {
            turn_thread.detach();
        }
    }

    {
        std::lock_guard lock(run_state->streamed_mutex);
        result.streamed_text = run_state->streamed_text;
        result.final_text = run_state->final_text;
        if (result.stop_reason == Result::StopReason::Interrupted) {
            const bool hard_limit = run_state->hard_step_limit.load(std::memory_order_acquire);
            if (agent->is_stop_requested()) {
                result.stop_reason = Result::StopReason::Cancelled;
            } else if (run_state->threw || (agent->last_turn_failed() && !hard_limit)) {
                result.stop_reason = Result::StopReason::Failed;
            } else if (run_state->budget_exhausted.load(std::memory_order_acquire)) {
                result.stop_reason = Result::StopReason::StepLimit;
            } else if (run_state->has_final_response) {
                result.stop_reason = Result::StopReason::Completed;
            }
        }
    }
    result.context_summary = agent->get_context_summary();
    result.history = agent->get_history();
    result.steps = run_state->steps.load(std::memory_order_acquire);
    result.tool_calls_total = run_state->tool_calls_total.load(std::memory_order_acquire);
    result.tool_calls_success = run_state->tool_calls_success.load(std::memory_order_acquire);
    if (result.final_text.empty()) {
        result.final_text = result.streamed_text.empty()
            ? "The worker stopped before producing a final answer; resume from the saved state."
            : result.streamed_text;
    }
    if (result.timed_out()) result.final_text = "Worker timed out. Progress was saved; resume when the worker has stopped.";

    emit_event(SubagentEvent{
        .kind = result.cancelled() ? SubagentEvent::Kind::Cancelled
            : result.status() == "failed" ? SubagentEvent::Kind::Failed
            : result.status() == "partial" ? SubagentEvent::Kind::Partial : SubagentEvent::Kind::Finished,
        .summary = result.final_text,
        .steps = result.steps,
        .tool_calls = result.tool_calls_total,
        .failed_tool_calls = result.tool_calls_total - result.tool_calls_success,
    }, true);

    return result;
}

} // namespace core::agent
