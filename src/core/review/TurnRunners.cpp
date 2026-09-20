#include "TurnRunner.hpp"

#include "../llm/Models.hpp"
#include "../utils/StringUtils.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <ranges>
#include <thread>
#include <utility>

namespace core::review {
namespace {

using core::utils::str::trim_ascii_copy;

[[nodiscard]] bool contains_stop_marker(std::string_view text) noexcept {
    return text.find(kStopMarker) != std::string_view::npos;
}

[[nodiscard]] bool cancelled(const std::function<bool()>& cancellation_requested) {
    return cancellation_requested && cancellation_requested();
}

// Collapses an agent notice into a single reportable line. Status text can run
// to several lines; the review card and the transcript both want one.
[[nodiscard]] std::string first_line(std::string_view text,
                                     std::string_view fallback) {
    for (const auto line : std::views::split(text, '\n')) {
        const auto trimmed = trim_ascii_copy(std::string_view(line.begin(), line.end()));
        if (!trimmed.empty()) {
            return trimmed;
        }
    }
    return std::string(fallback);
}

// Shared sink for one agent turn.
//
// Agent::send_message is only synchronous while the model answers with plain
// text: as soon as a step produces tool calls the agent hands the rest of the
// turn to a detached thread and returns. A review turn that is allowed to use
// read-only tools therefore used to come back instantly with no text ("the
// model returned an empty review response") while the real turn kept writing
// into the caller's stack. The state lives on the heap and the caller blocks
// until the agent's done callback fires, so a late callback can never touch a
// destroyed frame.
struct TurnSink {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    std::string text;
    std::string status;
    bool interrupted = false;

    void append_text(const std::string& chunk) {
        std::lock_guard lock(mutex);
        text += chunk;
        if (chunk.find(kStopMarker) != std::string_view::npos) {
            interrupted = true;
        }
    }

    void append_status(const std::string& chunk) {
        std::lock_guard lock(mutex);
        status += chunk;
    }

    void finish() {
        {
            std::lock_guard lock(mutex);
            done = true;
        }
        cv.notify_all();
    }

    void wait() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return done; });
    }

    bool wait_for(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex);
        return cv.wait_for(lock, timeout, [this] { return done; });
    }
};

} // namespace

AgentTurnRunner::AgentTurnRunner(
    std::shared_ptr<core::agent::Agent> agent,
    std::function<bool()> cancellation_requested)
    : agent_(std::move(agent)),
      cancellation_requested_(std::move(cancellation_requested)) {}

std::unique_ptr<TurnRunner> AgentTurnRunner::fork_for_parallel_request() const {
    if (!agent_) {
        return {};
    }

    const auto provider = agent_->get_provider();
    if (!provider || !provider->capabilities().supports_parallel_requests) {
        return {};
    }

    auto isolated_provider = provider->fork_for_parallel_request();
    if (!isolated_provider) {
        return {};
    }

    auto isolated_agent = agent_->make_isolated_agent(std::move(isolated_provider));
    if (!isolated_agent) {
        return {};
    }
    return std::make_unique<AgentTurnRunner>(
        std::move(isolated_agent),
        cancellation_requested_);
}

TurnRunner::Response AgentTurnRunner::run(const Request& request) {
    Response response;
    if (!agent_) {
        response.error = "Review requires an active agent session.";
        return response;
    }

    const auto history = agent_->get_history();
    const auto mode = agent_->get_mode();
    const auto summary = agent_->get_context_summary();

    auto sink = std::make_shared<TurnSink>();
    core::agent::Agent::TurnCallbacks callbacks{
        .allow_efficiency_rotation = false,
        .allow_background_memory_review = false,
        .min_context_utilization_for_rotation = kReviewRotationMinContextUtilization,
    };
    // Agent notices ("hit an output limit; continuing automatically") are not
    // part of the model's answer. Routing them to their own sink keeps the JSON
    // the parser sees clean.
    callbacks.on_status_log = [sink](const std::string& chunk) {
        sink->append_status(chunk);
    };
    callbacks.effort_override = request.effort;
    callbacks.max_tokens_override = request.max_tokens;
    callbacks.max_steps_override = request.max_steps;
    // json_object + tool calls is rejected by several providers. Large groups
    // that need tools emit JSON as the final assistant message instead.
    if (request.json_object && request.allowed_tools.empty()) {
        callbacks.response_format_override = core::llm::ResponseFormat{
            .type = core::llm::ResponseFormat::Type::JsonObject,
        };
    }
    if (request.allowed_tools.empty()) {
        callbacks.allowed_tools = {"__filo_no_tools__"};
    } else {
        callbacks.allowed_tools = request.allowed_tools;
    }

    // A review unit is an independent request. Sending the live conversation
    // history with every group turns a 20-group review into repeated large
    // prompts and makes latency grow with unrelated chat state. Keep the
    // session's system/tool configuration, but give the unit a clean history;
    // the original transcript is restored after the turn below.
    agent_->load_history({}, "", mode);
    // Restore the host transcript from the completion callback itself. The
    // callback runs after Agent releases turn_in_progress, so a timeout can
    // safely return without racing a late provider/tool completion.
    auto restore_history = [agent = agent_, history, summary, mode]() mutable {
        agent->load_history(std::move(history), summary, mode);
    };
    agent_->send_message(
        request.prompt,
        [sink](const std::string& chunk) { sink->append_text(chunk); },
        [](const std::string&, const std::string&) {},
        [sink, restore_history]() mutable {
            restore_history();
            sink->finish();
        },
        std::move(callbacks));

    // Blocks until the agent reports the turn finished, including turns that
    // continue on the agent's own tool thread. A provider that never emits a
    // final callback must still be given a cancellation path; otherwise this
    // review worker can spin forever with a live footer pill.
    constexpr auto kReviewTurnTimeout = std::chrono::seconds(90);
    const auto deadline = std::chrono::steady_clock::now() + kReviewTurnTimeout;
    bool timed_out = false;
    const auto wait_for_cancellation_grace = [&] {
        static constexpr auto kCancellationGrace = std::chrono::seconds(2);
        static_cast<void>(sink->wait_for(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                kCancellationGrace)));
    };
    for (;;) {
        if (sink->wait_for(std::chrono::milliseconds(100))) {
            break;
        }
        if ((cancellation_requested_ && cancellation_requested_())
            || agent_->is_stop_requested()) {
            agent_->request_stop();
            // The completion callback restores history after Agent releases
            // its turn. Give cooperative cancellation a short grace period,
            // but do not let a broken transport block the review forever.
            wait_for_cancellation_grace();
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            agent_->request_stop();
            timed_out = true;
            // A broken transport must not hold the review worker forever. The
            // completion callback owns the restoration and keeps its state on
            // the heap if the provider eventually returns.
            wait_for_cancellation_grace();
            break;
        }
    }

    std::string collected;
    std::string status;
    bool interrupted = false;
    {
        std::lock_guard lock(sink->mutex);
        collected = sink->text;
        status = sink->status;
        interrupted = sink->interrupted;
    }

    response.text = trim_ascii_copy(collected);
    response.interrupted = interrupted
        || timed_out
        || agent_->is_stop_requested()
        || contains_stop_marker(response.text);
    if (response.interrupted) {
        return response;
    }

    // A provider/transport failure arrives as ordinary streamed text. Without
    // this the error body would be parsed as a review and reported as a
    if (agent_->last_turn_failed()) {
        response.error = response.text.empty()
            ? first_line(trim_ascii_copy(status), "the model turn failed")
            : response.text;
        response.text.clear();
        return response;
    }
    if (response.text.empty()) {
        response.error = first_line(
            trim_ascii_copy(status),
            "the model returned an empty review response");
    }
    return response;
}

ProviderTurnRunner::ProviderTurnRunner(std::shared_ptr<core::llm::LLMProvider> provider,
                                       std::string model,
                                       std::function<bool()> cancellation_requested)
    : provider_(std::move(provider)),
      model_(std::move(model)),
      cancellation_requested_(std::move(cancellation_requested)) {}

std::unique_ptr<TurnRunner> ProviderTurnRunner::fork_for_parallel_request() const {
    if (!provider_ || !provider_->capabilities().supports_parallel_requests) {
        return {};
    }
    auto isolated = provider_->fork_for_parallel_request();
    if (!isolated) {
        return {};
    }
    return std::make_unique<ProviderTurnRunner>(
        std::move(isolated),
        model_,
        cancellation_requested_);
}

TurnRunner::Response ProviderTurnRunner::run(const Request& request) {
    Response response;
    if (!provider_) {
        response.error = "no active provider";
        return response;
    }
    if (cancelled(cancellation_requested_)) {
        response.interrupted = true;
        return response;
    }

    auto execution_provider = provider_;
    if (auto isolated = provider_->fork_for_parallel_request()) {
        execution_provider = std::move(isolated);
    }

    core::llm::ChatRequest chat;
    chat.model = model_;
    chat.effort = request.effort;
    chat.max_tokens = request.max_tokens;
    if (request.json_object) {
        chat.response_format.type = core::llm::ResponseFormat::Type::JsonObject;
    }
    chat.messages.push_back(core::llm::Message{
        .role = "user",
        .content = request.prompt,
    });

    std::mutex output_mutex;
    std::string output;
    std::atomic_bool completed{false};
    std::atomic_bool failed{false};
    std::atomic_bool stop_watch{false};
    std::thread watcher;
    if (cancellation_requested_) {
        watcher = std::thread([&] {
            while (!stop_watch.load(std::memory_order_acquire)) {
                if (cancelled(cancellation_requested_)) {
                    execution_provider->cancel();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }

    const auto stop_watcher = [&] {
        stop_watch.store(true, std::memory_order_release);
        if (watcher.joinable()) {
            watcher.join();
        }
    };

    try {
        execution_provider->stream_response(chat, [&](const core::llm::StreamChunk& chunk) {
            if (!chunk.content.empty()) {
                std::lock_guard lock(output_mutex);
                output += chunk.content;
            }
            if (chunk.is_final) {
                failed.store(chunk.is_error, std::memory_order_release);
                completed.store(true, std::memory_order_release);
            }
        });
    } catch (const std::exception& error) {
        stop_watcher();
        if (cancelled(cancellation_requested_)) {
            response.interrupted = true;
            return response;
        }
        response.error = error.what();
        return response;
    } catch (...) {
        stop_watcher();
        if (cancelled(cancellation_requested_)) {
            response.interrupted = true;
            return response;
        }
        response.error = "unknown provider error";
        return response;
    }
    stop_watcher();

    if (cancelled(cancellation_requested_)) {
        response.interrupted = true;
        return response;
    }
    if (failed.load(std::memory_order_acquire)) {
        response.error = "provider returned an error";
        return response;
    }
    if (!completed.load(std::memory_order_acquire)) {
        response.error = "stream ended without a final response";
        return response;
    }

    {
        std::lock_guard lock(output_mutex);
        response.text = trim_ascii_copy(output);
    }
    response.interrupted = contains_stop_marker(response.text);
    return response;
}

} // namespace core::review
