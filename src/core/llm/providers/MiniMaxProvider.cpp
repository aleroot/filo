#include "MiniMaxProvider.hpp"
#include "MiniMaxRequestBudget.hpp"
#include "core/llm/protocols/MiniMaxProtocol.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>

namespace core::llm::providers {
namespace {
/// M3 prefixes every structural tool-call tag with this marker. vLLM's parser
/// names it `NAMESPACE`; it is not a string the model emits as prose.
constexpr std::string_view kMiniMaxToolNamespace = "]<]minimax[>[";

constexpr std::string_view kLeakedToolCallError =
    "\n[MiniMax returned tool-call markup as plain text instead of a "
    "structured tool call. No tool was executed.]\n";

/// What one streamed attempt contained. The transport callback only records
/// it; the provider decides what to emit once the terminal event arrives.
struct AttemptObservation {
    std::string text;
    bool saw_structured_tools = false;
    bool failed = false;

    void clear() noexcept {
        text.clear();
        saw_structured_tools = false;
        failed = false;
    }

    void observe(const StreamChunk& chunk) {
        text += chunk.content;
        saw_structured_tools = saw_structured_tools || !chunk.tools.empty();
    }

    /// A leak is a clean text final. A structured tool_use is the gateway
    /// doing its job, and an incomplete or failed attempt already has a
    /// failure path. Neither is replaced by this one.
    [[nodiscard]] bool is_leaked_tool_call(const StreamChunk& terminal) const noexcept {
        return !failed
            && !terminal.incomplete_tool_call
            && !saw_structured_tools
            && minimax_leaked_tool_call_markup(text);
    }
};
} // namespace

bool minimax_leaked_tool_call_markup(std::string_view text) noexcept {
    return text.find(kMiniMaxToolNamespace) != std::string_view::npos;
}

MiniMaxProvider::MiniMaxProvider(Options options)
    : HttpLLMProvider(options.base_url, options.credentials, options.model,
        std::make_unique<protocols::MiniMaxProtocol>(
            options.subscription, options.default_effort, nullptr, options.context_window),
        core::config::ApiType::Anthropic, options.provider_name), options_(std::move(options)) {}

void MiniMaxProvider::stream_response(const ChatRequest& request,
                                      std::function<void(const StreamChunk&)> callback) {
    cancelled_.store(false, std::memory_order_release);
    // The gateway sometimes returns M3's native tool-call markup as a text
    // block and closes the turn normally. That text is not a tool call: it
    // can be truncated or malformed, and executing it would invent a call the
    // server did not structure. Retract it and fail the turn. Do not send the
    // request again. The response was a success, and a second identical call
    // only repeats the cost.
    AttemptObservation attempt;
    std::optional<StreamChunk> terminal;
    HttpLLMProvider::stream_response(request, [&](const StreamChunk& chunk) {
        if (chunk.reset_attempt) {
            attempt.clear();
            callback(chunk);
            return;
        }
        // Errors (transport cuts included) are never retracted, and the base
        // may follow one with a plain final marker: forward them as they
        // arrive instead of letting the final overwrite them.
        if (chunk.is_error) {
            attempt.failed = true;
            callback(chunk);
            return;
        }
        // Hold a clean terminal back until the whole attempt is observed. A
        // leak is only decidable once all text has arrived, and rejecting it
        // must replace this terminal with a reset and an error.
        attempt.observe(chunk);
        if (chunk.is_final) {
            terminal = chunk;
            return;
        }
        callback(chunk);
    });
    if (!terminal || cancelled_.load(std::memory_order_acquire)) {
        if (terminal) callback(*terminal);
        return;
    }
    if (!attempt.is_leaked_tool_call(*terminal)) {
        callback(*terminal);
        return;
    }
    callback(StreamChunk::make_attempt_reset());
    callback(StreamChunk::make_error(std::string(kLeakedToolCallError)));
}

void MiniMaxProvider::cancel() {
    cancelled_.store(true, std::memory_order_release);
    HttpLLMProvider::cancel();
}

std::size_t MiniMaxProvider::estimate_context_tokens(
    const std::vector<Message>& messages) const noexcept {
    return MiniMaxRequestBudget::estimate_history(messages);
}

int MiniMaxProvider::max_context_size_for(std::string_view model) const noexcept {
    const auto id = canonical_minimax_model(model.empty() ? std::string_view(options_.model) : model);
    const int selected = minimax_working_context_window(id, options_.context_window);
    if (const auto live = provider_model_info(id); live && live->context_window > 0) {
        return std::min(selected, live->context_window);
    }
    return selected;
}

int MiniMaxProvider::max_context_size() const noexcept {
    return max_context_size_for(options_.model);
}

std::shared_ptr<LLMProvider> MiniMaxProvider::fork_for_parallel_request() const {
    return std::make_shared<MiniMaxProvider>(options_);
}

} // namespace core::llm::providers
