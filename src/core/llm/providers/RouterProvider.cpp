#include "RouterProvider.hpp"
#include "../../budget/BudgetTracker.hpp"
#include "../../logging/Logger.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
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

// Classify failures as retryable (rate-limit, network, timeout) vs
// non-retryable (auth failure, invalid request).  Providers surface errors
// both as thrown exceptions and as terminal error chunks whose text is the
// provider-formatted message, so the classifier works on message text and is
// shared by both paths.
[[nodiscard]] bool is_retryable_message(std::string_view msg) noexcept {
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

[[nodiscard]] bool is_retryable_error(const std::exception& e) noexcept {
    return is_retryable_message(e.what());
}

// "1h 05m" / "12m 30s" / "45s" — compact human durations for cooldown and
// countdown notices.
[[nodiscard]] std::string format_wait_duration(std::chrono::seconds duration) {
    const auto total = duration.count();
    if (total <= 0) return "0s";
    const auto hours   = total / 3600;
    const auto minutes = (total % 3600) / 60;
    const auto seconds = total % 60;
    if (hours > 0) return std::format("{}h {:02}m", hours, minutes);
    if (minutes > 0) return std::format("{}m {:02}s", minutes, seconds);
    return std::format("{}s", seconds);
}

[[nodiscard]] std::string format_wall_clock(
    std::chrono::system_clock::time_point tp) {
    const auto unix_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        tp.time_since_epoch()).count();
    const std::time_t time{static_cast<std::time_t>(unix_seconds)};
    std::tm broken_down{};
#if defined(_WIN32)
    if (localtime_s(&broken_down, &time) != 0) return "<unknown time>";
#else
    if (localtime_r(&time, &broken_down) == nullptr) return "<unknown time>";
#endif
    char buffer[16];
    if (std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &broken_down) == 0) {
        return "<unknown time>";
    }
    return std::string{buffer};
}

[[nodiscard]] std::string trim_leading_newline(std::string_view text) {
    while (!text.empty() && (text.front() == '\n' || text.front() == '\r')) {
        text.remove_prefix(1);
    }
    return std::string{text};
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
    const core::llm::LLMProvider& provider,
    std::string_view session_id) {
    if (!guardrails.enabled()) return std::nullopt;

    const auto caps = provider.capabilities();
    if (caps.is_local && !guardrails.enforce_on_local) {
        return std::nullopt;
    }

    if (guardrails.max_session_cost_usd > 0.0 && provider.should_estimate_cost()) {
        const double spent = core::budget::BudgetTracker::get_instance()
                                 .snapshot(core::budget::TokenLedgerFilter{
                                     .session_id = std::string(session_id),
                                     .kind = core::budget::TokenLedgerEventKind::Actual,
                                 })
                                 .cost_usd();
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
                               std::unordered_map<std::string, std::string> provider_default_models,
                               bool isolate_target_requests,
                               std::shared_ptr<core::llm::routing::ProviderHealthRegistry> health)
    : provider_manager_(provider_manager)
    , router_engine_(std::move(router_engine))
    , provider_default_models_(std::move(provider_default_models))
    , health_(health ? std::move(health)
                     : std::make_shared<core::llm::routing::ProviderHealthRegistry>())
    , isolate_target_requests_(isolate_target_requests) {}

void RouterProvider::stream_response(
    const ChatRequest& request,
    std::function<void(const StreamChunk&)> callback) {

    {
        std::lock_guard lock(state_mutex_);
        last_should_estimate_cost_ = true;
        last_guardrail_summary_.clear();
    }
    wait_aborted_.store(false, std::memory_order_release);

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

    const auto failover = router_engine_->failover_config();
    const auto wait_started_at = std::chrono::steady_clock::now();

    // Outer loop: one iteration per routing attempt.  With the wait-for-reset
    // failover policy enabled, an exhausted chain parks the turn until the
    // soonest provider rate-limit reset and then re-routes from scratch
    // (fresh chain, fresh health) — the in-process equivalent of "wait for
    // the printed reset time, then continue", minus the terminal scraping.
    while (true) {
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
    // We only fall back to the next candidate if no model output has been streamed yet
    // (once bytes are in flight we cannot cleanly restart the response).
    const auto guardrails = router_engine_->guardrails();
    std::string last_error;
    std::vector<std::string> guardrail_blocks;
    std::vector<std::string> cooldown_blocks;

    for (auto& decision : chain) {
        // Resolve provider.
        std::shared_ptr<core::llm::LLMProvider> target_provider;
        try {
            target_provider = provider_manager_.get_provider(decision.provider);
        } catch (const std::exception& e) {
            last_error = std::format("failed to resolve provider '{}': {}", decision.provider, e.what());
            continue; // try next candidate
        }
        if (isolate_target_requests_) {
            if (auto isolated = target_provider->fork_for_parallel_request()) {
                target_provider = std::move(isolated);
            }
            std::lock_guard lock(state_mutex_);
            active_isolated_provider_ = target_provider;
        }

        // Health check: skip candidates still in a rate-limit cooldown or
        // breaker window instead of burning an attempt that is destined to
        // fail.  Cooldowns come from structured provider data observed on
        // previous requests (retry_after, subscription window resets).
        {
            const auto now = core::llm::routing::ProviderHealthRegistry::Clock::now();
            const auto snap = health_->snapshot(decision.provider);
            if (!snap.available(now)) {
                const auto remaining = snap.cooldown_remaining(now);
                last_error = std::format(
                    "provider '{}' in cooldown ({}), available in {}",
                    decision.provider, snap.reason, format_wait_duration(remaining));
                cooldown_blocks.push_back(std::format(
                    "{}: {} (resumes in {})",
                    decision.provider, snap.reason, format_wait_duration(remaining)));
                {
                    std::lock_guard lock(state_mutex_);
                    last_route_summary_ = std::format(
                        "{} via policy '{}' (skipped: cooldown — {})",
                        decision.provider, decision.policy, snap.reason);
                }
                continue;
            }
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
            if (auto block_reason = evaluate_guardrails(
                    *guardrails,
                    *target_provider,
                    request.session_id);
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

            bool saw_output = false; // real model output reached the callback
            bool threw = false;
            bool non_retryable = false;
            bool rate_limited = false;
            bool auth_recovery = false;
            std::string terminal_error_text;
            const auto started_at = std::chrono::steady_clock::now();

            try {
                target_provider->stream_response(
                    delegated_request,
                    [&, started_at, tp = target_provider, dec = decision](
                        const StreamChunk& inner_chunk) mutable {

                        // Error chunks carry their message in `content`;
                        // only non-error chunks count as streamed model output
                        // (otherwise a terminal error would block fallback).
                        const bool is_output = !inner_chunk.is_error
                            && (!inner_chunk.content.empty()
                                || !inner_chunk.reasoning_content.empty()
                                || !inner_chunk.tools.empty());
                        if (is_output) saw_output = true;

                        // Authentication recovery drives the re-auth flow in
                        // the agent loop; pass it through untouched.
                        if (inner_chunk.is_error
                            && inner_chunk.authentication_recovery.has_value()) {
                            auth_recovery = true;
                            callback(inner_chunk);
                            return;
                        }

                        if (inner_chunk.is_final && inner_chunk.is_error) {
                            // Terminal failure.  HttpLLMProvider-based
                            // providers signal exhausted retries and hard
                            // errors this way rather than throwing.  Keep the
                            // provider's rate-limit snapshot visible to the
                            // status bar, then suppress the chunk: when no
                            // output was streamed we can still fall back to
                            // the next candidate cleanly.
                            set_last_rate_limit_info(tp->get_last_rate_limit_info());
                            terminal_error_text = inner_chunk.content;
                            if (saw_output) callback(inner_chunk);
                            return;
                        }

                        // Non-final error chunks are the provider's own
                        // transient retry notices; forward for visibility.
                        if (inner_chunk.is_error) {
                            callback(inner_chunk);
                            return;
                        }

                        if (inner_chunk.is_final) {
                            health_->observe_success(dec.provider);

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

                if (auth_recovery) return; // surfaced; agent loop owns re-auth
                if (terminal_error_text.empty()) {
                    // stream_response completed without a terminal failure — success.
                    return;
                }

                // Terminal error chunk: classify exactly like a thrown failure.
                threw = true;
                last_error = std::format("provider '{}' attempt {}/{}: {}",
                                         decision.provider, attempt + 1, max_attempts,
                                         trim_leading_newline(terminal_error_text));
                non_retryable = !is_retryable_message(terminal_error_text);

            } catch (const std::exception& e) {
                threw = true;
                last_error = std::format("provider '{}' attempt {}/{}: {}",
                                         decision.provider, attempt + 1, max_attempts, e.what());
                non_retryable = !is_retryable_error(e);
            }

            // Feed the structured health data the provider recorded before
            // failing: 429 headers, retry_after, subscription window resets.
            {
                const auto info = target_provider->get_last_rate_limit_info();
                const auto now = core::llm::routing::ProviderHealthRegistry::Clock::now();
                health_->observe_rate_limit(decision.provider, info, now);
                if (info.is_rate_limited || info.unified_status == "rate_limited") {
                    rate_limited = true;
                    cooldown_blocks.push_back(std::format(
                        "{}: rate limited (resumes in {})",
                        decision.provider,
                        format_wait_duration(
                            health_->snapshot(decision.provider)
                                .cooldown_remaining(now))));
                } else if (!non_retryable) {
                    health_->observe_failure(decision.provider, now);
                }
            }

            if (saw_output) {
                // Model output was already streamed to the user; we cannot cleanly
                // retry or fall back.  The provider's own error chunk (if any) was
                // forwarded inline above; otherwise surface the failure here.
                if (terminal_error_text.empty()) {
                    callback(StreamChunk::make_error(std::format("\n[Router: provider '{}' failed mid-stream — fallback not possible: {}]",
                                                         decision.provider, last_error)));
                }
                return;
            }

            if (rate_limited) break;     // cooldown owns the timing — next candidate
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
    std::string cooldown_note;
    if (!cooldown_blocks.empty()) {
        cooldown_note = std::format(" Provider cooldowns: {}.", join_limited(cooldown_blocks));
    }

    // Wait-for-reset policy: park the turn until the soonest provider
    // rate-limit reset, then re-route.  Only scheduled resets are waited
    // for — a chain that simply failed has nothing to wait for.
    if (failover.wait_on_exhaustion) {
        const auto now_sys = core::llm::routing::ProviderHealthRegistry::Clock::now();
        std::string reset_provider;
        const auto soonest = health_->soonest_rate_limit_reset(now_sys, &reset_provider);
        if (soonest.has_value()) {
            const auto target =
                *soonest + std::chrono::seconds(failover.reset_margin_seconds);

            if (failover.max_wait_seconds > 0) {
                const auto waited = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - wait_started_at);
                const auto budget_left =
                    std::chrono::seconds(failover.max_wait_seconds) - waited;
                const auto needed = std::chrono::duration_cast<std::chrono::seconds>(
                    target - now_sys);
                if (needed > budget_left) {
                    callback(StreamChunk::make_error(std::format(
                        "\n[Router error: all candidates failed.{}{} Next reset on '{}' "
                        "in {} exceeds the {}s wait budget. Last error: {}]",
                        guardrail_note, cooldown_note, reset_provider,
                        format_wait_duration(needed),
                        failover.max_wait_seconds, last_error)));
                    return;
                }
            }

            if (wait_for_reset(target, reset_provider, callback)) {
                continue; // reset reached — re-route with fresh health
            }
            callback(StreamChunk::make_final()); // cancelled while parked
            return;
        }
        // No scheduled reset to wait for — fall through to the terminal error.
    }

    callback(StreamChunk::make_error(
        std::format("\n[Router error: all candidates failed.{}{} Last error: {}]",
                    guardrail_note,
                    cooldown_note,
                    last_error)));
    return;
    } // wait/resume loop
}

bool RouterProvider::wait_for_reset(
    std::chrono::system_clock::time_point target,
    const std::string& provider,
    const std::function<void(const StreamChunk&)>& callback) {
    using clock = std::chrono::system_clock;

    const auto emit_notice = [&callback](std::string_view text) {
        callback(StreamChunk::make_content(std::string{text}));
    };

    {
        const auto remaining = std::chrono::duration_cast<std::chrono::seconds>(
            target - clock::now());
        emit_notice(std::format(
            "\n[Failover] All routed providers are rate-limited. Waiting for '{}' "
            "to reset at {} (in {}) before retrying — cancel to stop.",
            provider, format_wall_clock(target),
            format_wait_duration(remaining)));
    }

    auto next_update = clock::now() + std::chrono::seconds(60);
    while (clock::now() < target) {
        if (wait_aborted_.load(std::memory_order_acquire)) return false;

        const auto now = clock::now();
        if (now >= next_update) {
            const auto remaining = std::chrono::duration_cast<std::chrono::seconds>(
                target - now);
            emit_notice(std::format(
                "\n[Failover] Still waiting for '{}' to reset (in {}).",
                provider, format_wait_duration(remaining)));
            next_update = now + std::chrono::seconds(60);
        }
        // Short ticks on the wall clock: a system suspend consumes the wait
        // naturally, and cancel() takes effect within one tick.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (wait_aborted_.load(std::memory_order_acquire)) return false;
    emit_notice(std::format(
        "\n[Failover] Reset window reached for '{}'; re-routing.", provider));
    return true;
}

std::string RouterProvider::get_last_model() const {
    std::lock_guard lock(state_mutex_);
    return last_model_;
}

ProviderCapabilities RouterProvider::capabilities() const {
    return ProviderCapabilities{
        .supports_tool_calls = true,
        .is_local = false,
        .supports_parallel_requests = true,
    };
}

std::shared_ptr<LLMProvider> RouterProvider::fork_for_parallel_request() const {
    // Forks share the health registry so parallel requests pool their
    // provider knowledge (a 429 seen by one fork cools the provider for all).
    return std::make_shared<RouterProvider>(
        provider_manager_,
        router_engine_ ? router_engine_->fork() : nullptr,
        provider_default_models_,
        true,
        health_);
}

bool RouterProvider::should_estimate_cost() const {
    std::lock_guard lock(state_mutex_);
    return last_should_estimate_cost_;
}

void RouterProvider::cancel() {
    // Abort an active wait-for-reset park before forwarding to providers.
    wait_aborted_.store(true, std::memory_order_release);

    if (isolate_target_requests_) {
        std::shared_ptr<LLMProvider> active_provider;
        {
            std::lock_guard lock(state_mutex_);
            active_provider = active_isolated_provider_;
        }
        if (active_provider) active_provider->cancel();
        return;
    }
    for (const auto& [provider_name, _] : provider_default_models_) {
        try {
            if (auto provider = provider_manager_.get_provider(provider_name)) {
                provider->cancel();
            }
        } catch (...) {
            // Cancellation is best-effort across router candidates.
        }
    }
}

void RouterProvider::reset_conversation_state() {
    if (isolate_target_requests_) {
        std::shared_ptr<LLMProvider> active_provider;
        {
            std::lock_guard lock(state_mutex_);
            active_provider = active_isolated_provider_;
        }
        if (active_provider) active_provider->reset_conversation_state();
        return;
    }
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

bool RouterProvider::set_active_policy(std::string policy_name) {
    return router_engine_ && router_engine_->set_active_policy(std::move(policy_name));
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
