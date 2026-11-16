#include "MiniMaxProvider.hpp"
#include "MiniMaxRequestBudget.hpp"
#include "core/llm/protocols/MiniMaxProtocol.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>

namespace core::llm::providers {
namespace {
/// The namespace marker M3 prefixes to every structural tool-call tag.
constexpr std::string_view kLeakedNamespace = "]<]minimax[>[";
} // namespace

bool minimax_leaked_tool_call_markup(std::string_view text) noexcept {
    return text.find(kLeakedNamespace) != std::string_view::npos
        && text.find("<invoke") != std::string_view::npos;
}

MiniMaxProvider::MiniMaxProvider(Options options)
    : HttpLLMProvider(options.base_url, options.credentials, options.model,
        std::make_unique<protocols::MiniMaxProtocol>(
            options.subscription, options.default_effort, nullptr, options.context_window),
        core::config::ApiType::Anthropic, options.provider_name), options_(std::move(options)) {}

void MiniMaxProvider::stream_response(const ChatRequest& request,
                                      std::function<void(const StreamChunk&)> callback) {
    cancelled_.store(false, std::memory_order_release);
    // The gateway usually structures the model's native tool-call markup into
    // tool_use blocks, but intermittently leaks it as plain text and closes the
    // turn normally. The leaked markup is not a usable call (it can even carry
    // malformed tags), so it is never executed: retract it and ask once more.
    // A second leak is reported honestly instead of shown as garbage.
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (cancelled_.load(std::memory_order_acquire)) {
            callback(StreamChunk::make_final());
            return;
        }
        std::string text;
        bool saw_structured_tools = false;
        bool failed = false;
        std::optional<StreamChunk> terminal;
        HttpLLMProvider::stream_response(request, [&](const StreamChunk& chunk) {
            if (chunk.reset_attempt) {
                text.clear();
                saw_structured_tools = false;
                failed = false;
                callback(chunk);
                return;
            }
            // Errors (transport cuts included) are never retracted, and the
            // base may follow one with a plain final marker: forward them
            // as they arrive instead of letting the final overwrite them.
            if (chunk.is_error) {
                failed = true;
                callback(chunk);
                return;
            }
            // Hold a clean terminal back until the whole attempt is observed:
            // a leaked call is only decidable once all text has arrived, and a
            // retry must replace this terminal with an attempt reset.
            if (chunk.is_final) {
                text += chunk.content;
                saw_structured_tools = saw_structured_tools || !chunk.tools.empty();
                terminal = chunk;
                return;
            }
            text += chunk.content;
            saw_structured_tools = saw_structured_tools || !chunk.tools.empty();
            callback(chunk);
        });
        if (!terminal) return;
        if (cancelled_.load(std::memory_order_acquire)) {
            callback(*terminal);
            return;
        }
        const bool leaked = !failed && !terminal->incomplete_tool_call
            && !saw_structured_tools && minimax_leaked_tool_call_markup(text);
        if (!leaked) {
            callback(*terminal);
            return;
        }
        if (attempt == 0) {
            callback(StreamChunk::make_attempt_reset(
                "[MiniMax returned a tool call as text; retrying (1/1)]"));
            continue;
        }
        callback(StreamChunk::make_attempt_reset());
        callback(StreamChunk::make_error(
            "\n[MiniMax returned its tool-call markup as plain text instead of a "
            "structured tool call, twice. No tool was executed. Send the message "
            "again; if it keeps happening, switch models.]"));
        return;
    }
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
