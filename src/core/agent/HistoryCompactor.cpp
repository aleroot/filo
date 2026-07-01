#include "HistoryCompactor.hpp"

#include <atomic>
#include <format>
#include <thread>

namespace core::agent {

namespace {

[[nodiscard]] std::string trim_copy(std::string value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

void emit(const HistoryCompactionCallbacks& callbacks, const std::string& text) {
    if (callbacks.on_status) {
        callbacks.on_status(text);
    }
}

[[nodiscard]] std::string start_status(HistoryCompactionReason reason) {
    if (reason == HistoryCompactionReason::Manual) {
        return "\n\xc2\xbb  Compacting conversation...\n";
    }
    return "\n\n\xe2\x9a\x99  Auto-compacting history...\n";
}

[[nodiscard]] std::string success_status(HistoryCompactionReason reason) {
    if (reason == HistoryCompactionReason::Manual) {
        return "\n\xe2\x9c\x93  History compacted and summary preserved.\n";
    }
    return "\xe2\x9c\x93  History compacted.\n\n";
}

} // namespace

void HistoryCompactor::compact_async(HistoryCompactionRequest request,
                                     HistoryCompactionCallbacks callbacks) const {
    std::thread([request = std::move(request), callbacks = std::move(callbacks)]() mutable {
        emit(callbacks, start_status(request.reason));

        if (!request.provider) {
            emit(callbacks, "\xe2\x9c\x97  History compaction skipped: no active provider.\n\n");
            return;
        }

        auto summary = std::make_shared<std::string>();
        auto completed = std::make_shared<std::atomic<bool>>(false);
        try {
            const auto reason = request.reason;
            core::llm::ChatRequest llm_request = build_request(request);
            request.provider->stream_response(
                llm_request,
                [summary, callbacks, completed, reason](const core::llm::StreamChunk& chunk) {
                    if (!chunk.content.empty()) *summary += chunk.content;
                    if (!chunk.is_final) return;

                    completed->store(true, std::memory_order_release);
                    if (chunk.is_error) {
                        emit(callbacks, "\xe2\x9c\x97  History compaction failed.\n\n");
                        return;
                    }

                    const std::string compacted = trim_copy(*summary);
                    if (compacted.empty()) {
                        emit(callbacks, "\xe2\x9c\x97  History compaction failed: empty summary.\n\n");
                        return;
                    }

                    if (callbacks.on_summary) {
                        callbacks.on_summary(compacted);
                    }
                    emit(callbacks, success_status(reason));
                });

            if (!completed->load(std::memory_order_acquire)) {
                emit(callbacks, "\xe2\x9c\x97  History compaction failed: stream ended without a final response.\n\n");
            }
        } catch (const std::exception& e) {
            emit(callbacks, std::format(
                "\xe2\x9c\x97  History compaction failed: {}\n\n",
                e.what()));
        } catch (...) {
            emit(callbacks, "\xe2\x9c\x97  History compaction failed: unknown provider error.\n\n");
        }
    }).detach();
}

core::llm::ChatRequest HistoryCompactor::build_request(
    const HistoryCompactionRequest& request) {
    core::llm::ChatRequest llm_request;
    llm_request.model = request.model;
    llm_request.messages = request.history;
    llm_request.messages.push_back({
        "user",
        "Summarise the conversation so far in a single, dense paragraph. "
        "Focus on the technical context and the state of the task we are working on. "
        "Maintain all critical facts, paths, and requirements."
    });
    return llm_request;
}

} // namespace core::agent
