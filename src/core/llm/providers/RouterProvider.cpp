#include "RouterProvider.hpp"
#include "../../budget/BudgetTracker.hpp"
#include "../../logging/Logger.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

namespace core::llm::providers {

namespace {

[[nodiscard]] std::string extract_latest_user_prompt(const ChatRequest& request) {
    for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it) {
        if (it->role == "user" && !it->content.empty()) {
            return it->content;
        }
    }
    return {};
}

[[nodiscard]] bool has_tool_history(const ChatRequest& request) {
    for (const auto& message : request.messages) {
        if (message.role == "tool") {
            return true;
        }
    }
    return false;
}

// Rough token count for the full message history excluding the last user turn.
// GPT-style estimate: 1 token ≈ 4 characters.
[[nodiscard]] std::size_t estimate_history_tokens(const ChatRequest& request) {
    std::size_t chars = 0;
    bool first_user_from_end = true;
    for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it) {
        if (it->role == "user" && first_user_from_end) {
            first_user_from_end = false;
            continue; // skip latest user turn — that's the prompt itself
        }
        chars += it->content.size();
    }
    return chars / 4 + 1;
}

// Count the number of turns (user messages) in the conversation.
[[nodiscard]] int count_turns(const ChatRequest& request) {
    int turns = 0;
    for (const auto& msg : request.messages) {
        if (msg.role == "user") ++turns;
    }
    return turns;
}

// Classify errors as retryable (rate-limit, network, timeout) vs non-retryable
// (auth failure, invalid request).  We inspect the exception message because
// all providers surface errors as std::runtime_error / std::exception with a
// descriptive string.
[[nodiscard]] bool is_retryable_error(const std::exception& e) noexcept {
    const std::string_view msg{e.what()};
    // Non-retryable: authentication or client-side invalid request.
    for (const std::string_view marker : {
             "401", "403", "invalid_api_key", "authentication",
             "400", "invalid_request", "bad request",
         }) {
        if (msg.find(marker) != std::string_view::npos) {
            return false;
        }
    }
    return true; // timeout, 429, 5xx, network error → retryable
}

// Exponential backoff with ±25 % jitter.
// attempt=1 → ~200 ms, attempt=2 → ~400 ms, …, capped at 5 s.
void backoff_sleep(int attempt) {
    constexpr int base_ms = 200;
    constexpr int cap_ms  = 5000;
    const int delay_ms = std::min(base_ms * (1 << (attempt - 1)), cap_ms);
    // Simple deterministic jitter: ±25 % based on attempt number.
    const int jitter_ms = (delay_ms / 4) * ((attempt % 2 == 0) ? 1 : -1);
    const int sleep_ms  = std::max(1, delay_ms + jitter_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
}

[[nodiscard]] std::optional<std::string> evaluate_guardrails(
    const core::llm::routing::RouterGuardrails& guardrails,
    const core::llm::LLMProvider& provider) {
    if (!guardrails.enabled()) return std::nullopt;

    const auto caps = provider.capabilities();
    if (caps.is_local && !guardrails.enforce_on_local) {
        return std::nullopt;
    }

    if (guardrails.max_session_cost_usd > 0.0 && provider.should_estimate_cost()) {
        const double spent = core::budget::BudgetTracker::get_instance().session_cost_usd();
        if (spent >= guardrails.max_session_cost_usd) {
            return std::format("session spend ${:.4f} hit cap ${:.4f}",
                               spent,
                               guardrails.max_session_cost_usd);
        }
    }

    const auto rl = provider.get_last_rate_limit_info();

    if ((guardrails.min_requests_remaining_ratio > 0.0f
         || guardrails.min_tokens_remaining_ratio > 0.0f
         || guardrails.min_window_remaining_ratio > 0.0f)
        && rl.is_rate_limited) {
        if (rl.retry_after > 0) {
            return std::format("provider is rate-limited (retry_after={}s)", rl.retry_after);
        }
        return "provider is rate-limited";
    }

    if (guardrails.min_requests_remaining_ratio > 0.0f && rl.requests_limit > 0) {
        const float remaining_ratio = static_cast<float>(rl.requests_remaining)
                                      / static_cast<float>(rl.requests_limit);
        if (remaining_ratio < guardrails.min_requests_remaining_ratio) {
            return std::format("request reserve {:.0f}% breached ({:.0f}% remaining)",
                               guardrails.min_requests_remaining_ratio * 100.0f,
                               remaining_ratio * 100.0f);
        }
    }

    if (guardrails.min_tokens_remaining_ratio > 0.0f && rl.tokens_limit > 0) {
        const float remaining_ratio = static_cast<float>(rl.tokens_remaining)
                                      / static_cast<float>(rl.tokens_limit);
        if (remaining_ratio < guardrails.min_tokens_remaining_ratio) {
            return std::format("token reserve {:.0f}% breached ({:.0f}% remaining)",
                               guardrails.min_tokens_remaining_ratio * 100.0f,
                               remaining_ratio * 100.0f);
        }
    }

    if (guardrails.min_window_remaining_ratio > 0.0f && !rl.usage_windows.empty()) {
        const float remaining_ratio = std::max(0.0f, 1.0f - rl.max_window_utilization());
        if (remaining_ratio < guardrails.min_window_remaining_ratio) {
            return std::format("quota window reserve {:.0f}% breached ({:.0f}% remaining)",
                               guardrails.min_window_remaining_ratio * 100.0f,
                               remaining_ratio * 100.0f);
        }
    }

    return std::nullopt;
}

[[nodiscard]] std::string join_limited(const std::vector<std::string>& items,
                                       std::size_t max_items = 4) {
    if (items.empty()) return {};
    std::string out;
    const std::size_t limit = std::min(max_items, items.size());
    for (std::size_t i = 0; i < limit; ++i) {
        if (i > 0) out += "; ";
        out += items[i];
    }
    if (items.size() > max_items) {
        out += std::format("; +{} more", items.size() - max_items);
    }
    return out;
}

} // namespace

RouterProvider::RouterProvider(core::llm::ProviderManager& provider_manager,
                               std::shared_ptr<core::llm::routing::RouterEngine> router_engine,
                               std::unordered_map<std::string, std::string> provider_default_models)
    : provider_manager_(provider_manager)
    , router_engine_(std::move(router_engine))
    , provider_default_models_(std::move(provider_default_models)) {}

void RouterProvider::stream_response(
    const ChatRequest& request,
    std::function<void(const StreamChunk&)> callback) {

    {
        std::lock_guard lock(state_mutex_);
        last_should_estimate_cost_ = true;
        last_guardrail_summary_.clear();
    }

    if (!router_engine_) {
        callback(StreamChunk::make_error("\n[Router error: routing engine is not available]"));
        return;
    }

    const core::llm::routing::RouteContext route_context{
        .prompt            = extract_latest_user_prompt(request),
        .has_tool_messages = has_tool_history(request),
        .turn_count        = count_turns(request),
        .history_tokens    = estimate_history_tokens(request),
    };

    // Obtain the full fallback chain — ordered list of candidates to try.
    std::vector<core::llm::routing::RouteDecision> chain = router_engine_->route_chain(route_context);

    if (chain.empty()) {
        // route_chain() returns nothing only when the router is completely unconfigured
        // or all candidates are unavailable.  Fall back to the single-decision path to
        // preserve the original error message.
        auto single = router_engine_->route(route_context);
        callback(StreamChunk::make_error(std::format("\n[Router error: {}]", single.reason)));
        return;
    }

    // Iterate through the chain.  For each candidate, attempt up to (retries+1) calls.
    // We only fall back to the next candidate if no streaming content has been sent yet
    // (once bytes are in flight we cannot cleanly restart the response).
    const auto guardrails = router_engine_->guardrails();
    std::string last_error;
    std::vector<std::string> guardrail_blocks;

    for (auto& decision : chain) {
        // Resolve provider.
        std::shared_ptr<core::llm::LLMProvider> target_provider;
        try {
            target_provider = provider_manager_.get_provider(decision.provider);
        } catch (const std::exception& e) {
            last_error = std::format("failed to resolve provider '{}': {}", decision.provider, e.what());
            continue; // try next candidate
        }

        // Resolve model: use decision.model if set, else provider default.
        auto delegated_request = request;
        if (!decision.model.empty()) {
            delegated_request.model = decision.model;
        } else {
            const auto it = provider_default_models_.find(decision.provider);
            if (it != provider_default_models_.end()) {
                delegated_request.model = it->second;
            }
        }
        decision.model = delegated_request.model;

        if (guardrails.has_value()) {
            if (auto block_reason = evaluate_guardrails(*guardrails, *target_provider);
                block_reason.has_value()) {
                const std::string blocked_summary = std::format("{}: {}",
                                                                decision.provider,
                                                                *block_reason);
                guardrail_blocks.push_back(blocked_summary);
                last_error = std::format("candidate '{}' blocked by guardrails: {}",
                                         decision.provider,
                                         *block_reason);
                {
                    std::lock_guard lock(state_mutex_);
                    last_route_summary_ = std::format("{} via policy '{}' (skipped: {})",
                                                      decision.provider,
                                                      decision.policy,
                                                      *block_reason);
                }
                continue;
            }
        }

        {
            std::lock_guard lock(state_mutex_);
            last_route_summary_ = std::format("{} via policy '{}' ({})",
                                              decision.provider,
                                              decision.policy,
                                              decision.reason);
        }

        // Per-candidate retry loop.
        const int max_attempts = decision.retries + 1;
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            if (attempt > 0) {
                backoff_sleep(attempt);
            }

            bool any_content_sent = false;
            bool threw = false;
            bool non_retryable = false;
            const auto started_at = std::chrono::steady_clock::now();

            try {
                target_provider->stream_response(
                    delegated_request,
                    [&, started_at, tp = target_provider, dec = decision](
                        const StreamChunk& inner_chunk) mutable {

                        any_content_sent = true;

                        if (inner_chunk.is_final) {
                            const auto usage = tp->get_last_usage();
                            set_last_usage(usage.prompt_tokens, usage.completion_tokens);
                            set_last_rate_limit_info(tp->get_last_rate_limit_info());

                            if (router_engine_) {
                                const auto elapsed =
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - started_at);
                                router_engine_->record_latency(dec, static_cast<int>(elapsed.count()));
                            }

                            std::lock_guard lock(state_mutex_);
                            last_model_ = dec.model;
                            last_should_estimate_cost_ = tp->should_estimate_cost();
                            if (!guardrail_blocks.empty()) {
                                last_guardrail_summary_ = join_limited(guardrail_blocks);
                                last_route_summary_ = std::format(
                                    "{} via policy '{}' ({}) [guardrail fallback: {}]",
                                    dec.provider,
                                    dec.policy,
                                    dec.reason,
                                    join_limited(guardrail_blocks, 2));
                            } else {
                                last_guardrail_summary_.clear();
                            }
                        }

                        callback(inner_chunk);
                    });

                // stream_response returned without throwing — success.
                return;

            } catch (const std::exception& e) {
                threw = true;
                last_error = std::format("provider '{}' attempt {}/{}: {}",
                                         decision.provider, attempt + 1, max_attempts, e.what());
                non_retryable = !is_retryable_error(e);
            }

            if (any_content_sent) {
                // Content was already streamed to the user; we cannot cleanly retry
                // or fall back.  Surface the error inline and stop.
                callback(StreamChunk::make_error(std::format("\n[Router: provider '{}' failed mid-stream — fallback not possible: {}]",
                                     decision.provider, last_error)));
                return;
            }

            if (!threw) break;           // shouldn't happen — left for clarity
            if (non_retryable) break;    // skip remaining retries, try next candidate
            // else: retryable, loop continues with backoff
        }
    }

    // All candidates in the chain were exhausted.
    std::string guardrail_note;
    if (!guardrail_blocks.empty()) {
        {
            std::lock_guard lock(state_mutex_);
            last_guardrail_summary_ = join_limited(guardrail_blocks);
        }
        guardrail_note = std::format(" Guardrails blocked: {}.", join_limited(guardrail_blocks));
    }
    callback(StreamChunk::make_error(
        std::format("\n[Router error: all candidates failed.{} Last error: {}]",
                    guardrail_note,
                    last_error)));
}

std::string RouterProvider::get_last_model() const {
    std::lock_guard lock(state_mutex_);
    return last_model_;
}

bool RouterProvider::should_estimate_cost() const {
    std::lock_guard lock(state_mutex_);
    return last_should_estimate_cost_;
}

void RouterProvider::reset_conversation_state() {
    for (const auto& [provider_name, _] : provider_default_models_) {
        try {
            if (auto provider = provider_manager_.get_provider(provider_name)) {
                provider->reset_conversation_state();
            }
        } catch (const std::exception& e) {
            core::logging::warn("Failed to reset conversation state for provider {}: {}",
                                provider_name, e.what());
        }
    }
}

std::string RouterProvider::active_policy() const {
    if (!router_engine_) return {};
    return router_engine_->active_policy();
}

std::string RouterProvider::last_route_summary() const {
    std::lock_guard lock(state_mutex_);
    return last_route_summary_;
}

std::string RouterProvider::last_guardrail_summary() const {
    std::lock_guard lock(state_mutex_);
    return last_guardrail_summary_;
}

} // namespace core::llm::providers
