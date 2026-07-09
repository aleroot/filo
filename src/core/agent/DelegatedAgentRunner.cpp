#include "DelegatedAgentRunner.hpp"

#include <atomic>
#include <condition_variable>
#include <thread>
#include <mutex>

namespace core::agent {

namespace {

[[nodiscard]] bool is_tool_error_payload(std::string_view payload) noexcept {
    return payload.find("\"error\"") != std::string_view::npos;
}

struct RunState {
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    std::mutex streamed_mutex;
    std::string streamed_text;

    std::atomic<int> steps{0};
    std::atomic<int> tool_calls_total{0};
    std::atomic<int> tool_calls_success{0};
    std::atomic<int> failed_tool_calls{0};
};

[[nodiscard]] std::string build_delegated_prompt(const DelegatedAgentRunner::Request& request) {
    if (request.worker_name.empty()
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
        request.session_context);
    agent->set_active_provider_name(request.provider_name);
    agent->set_mode(request.mode);
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
    const std::string delegated_prompt = build_delegated_prompt(request);
    std::string ledger_actor = request.worker_name.empty() ? std::string("subagent") : std::string("subagent:") + request.worker_name;

    auto emit_event = [callback = request.on_event,
                       parent_tool_call_id = request.parent_tool_call_id,
                       task_id = request.task_id,
                       worker_name = request.worker_name,
                       description = request.task_description,
                       provider_name = request.provider_name,
                       model_name = request.model](SubagentEvent event) {
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
         allowed_tools = std::move(request.allowed_tools),
         ledger_actor = std::move(ledger_actor),
         emit_event]() mutable {
            agent->send_message(
                delegated_prompt,
                [run_state, emit_event](const std::string& chunk) {
                    std::lock_guard lock(run_state->streamed_mutex);
                    run_state->streamed_text += chunk;
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
                    std::lock_guard lock(run_state->done_mutex);
                    run_state->done = true;
                    run_state->done_cv.notify_all();
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
                        if (!is_tool_error_payload(tool_result.content)) {
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
                    .allowed_tools = std::move(allowed_tools),
                    .ledger_actor = std::move(ledger_actor),
                    .allow_efficiency_rotation = false,
                });
        });

    {
        std::unique_lock lock(run_state->done_mutex);
        const auto deadline = std::chrono::steady_clock::now() + request.timeout;
        while (!run_state->done) {
            if (request.cancellation_requested && request.cancellation_requested()) {
                agent->request_stop();
                result.cancelled = true;
                break;
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                agent->request_stop();
                result.timed_out = true;
                result.cancelled = true;
                break;
            }

            run_state->done_cv.wait_for(lock, std::chrono::milliseconds(100));
        }
    }

    if (turn_thread.joinable()) {
        std::lock_guard lock(run_state->done_mutex);
        if (run_state->done) {
            turn_thread.join();
        } else {
            turn_thread.detach();
        }
    }

    {
        std::lock_guard lock(run_state->streamed_mutex);
        result.streamed_text = run_state->streamed_text;
    }
    result.context_summary = agent->get_context_summary();
    result.history = agent->get_history();
    result.steps = run_state->steps.load(std::memory_order_acquire);
    result.tool_calls_total = run_state->tool_calls_total.load(std::memory_order_acquire);
    result.tool_calls_success = run_state->tool_calls_success.load(std::memory_order_acquire);
    result.cancelled = result.cancelled
        || result.timed_out
        || (request.cancellation_requested && request.cancellation_requested())
        || agent->is_stop_requested();

    emit_event(SubagentEvent{
        .kind = result.cancelled
            ? (result.timed_out ? SubagentEvent::Kind::Failed : SubagentEvent::Kind::Cancelled)
            : SubagentEvent::Kind::Finished,
        .summary = result.streamed_text,
        .steps = result.steps,
        .tool_calls = result.tool_calls_total,
        .failed_tool_calls = result.tool_calls_total - result.tool_calls_success,
    });

    return result;
}

} // namespace core::agent
