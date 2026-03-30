#pragma once

#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <atomic>
#include "Models.hpp"
#include "protocols/ApiProtocol.hpp"

namespace core::llm {

struct ProviderCapabilities {
    bool supports_tool_calls = true;
    bool is_local = false;
};

class LLMProvider {
public:
    virtual ~LLMProvider() = default;

    /**
     * @brief Stream a response from the LLM based on the conversation history.
     * @param request The chat request options.
     * @param callback A function called with each new chunk of response.
     *        The StreamChunk struct contains content, tool_calls, is_final flag,
     *        and is_error flag for HTTP/API errors.
     */
    virtual void stream_response(const ChatRequest& request,
                                 std::function<void(const StreamChunk&)> callback) = 0;

    /**
     * @brief Return the effective model used by the most recent completed request.
     *        Providers that do not track this can return an empty string.
     */
    [[nodiscard]] virtual std::string get_last_model() const { return {}; }

    /**
     * @brief Report provider feature support so the agent loop can adapt request
     *        construction for local or reduced-capability backends.
     */
    [[nodiscard]] virtual ProviderCapabilities capabilities() const { return {}; }

    /**
     * @brief Return whether heuristic USD cost estimation should be applied to
     *        the latest usage data reported by this provider.
     *        Local backends can override this to suppress synthetic pricing.
     */
    [[nodiscard]] virtual bool should_estimate_cost() const { return true; }

    /**
     * @brief Return the maximum context size (in tokens) for the current model.
     *        Returns 0 if unknown. Used for auto-compaction decisions.
     *        HTTP-based providers should override this based on the model's specs.
     */
    [[nodiscard]] virtual int max_context_size() const noexcept { return 0; }

    /**
     * @brief Return the token usage from the most recent completed request.
     *        Thread-safe: uses acquire/release atomics.
     *        Returns zero-initialized struct if the provider does not track usage.
     */
    [[nodiscard]] TokenUsage get_last_usage() const noexcept {
        const int32_t p = last_prompt_.load(std::memory_order_acquire);
        const int32_t c = last_completion_.load(std::memory_order_acquire);
        return TokenUsage{ .prompt_tokens = p, .completion_tokens = c, .total_tokens = p + c };
    }
    
    /**
     * @brief Return rate limit information from the most recent completed request.
     *
     * Thread-safe (lock-free): the full RateLimitInfo snapshot — including
     * the variable-length unified_windows vector — is published as an immutable
     * shared_ptr via atomic_load/atomic_store free functions. The render thread
     * can call this at 60 fps with no contention.
     *
     * Returns a zero-initialised RateLimitInfo when no response has been received.
     */
    [[nodiscard]] virtual protocols::RateLimitInfo get_last_rate_limit_info() const noexcept {
        auto snap = std::atomic_load_explicit(
            &last_rate_limit_snapshot_, std::memory_order_acquire);
        return snap ? *snap : protocols::RateLimitInfo{};
    }

protected:
    /**
     * @brief Called by subclasses (from the streaming thread) when usage data
     *        is available.  Must be called BEFORE the final is_final=true callback
     *        to guarantee visibility at the call-site.
     */
    void set_last_usage(int32_t prompt, int32_t completion) noexcept {
        last_prompt_.store(prompt,     std::memory_order_release);
        last_completion_.store(completion, std::memory_order_release);
    }

    /**
     * @brief Called by subclasses (from the streaming thread) when rate limit info
     *        is available.  Publishes a new immutable snapshot atomically so the
     *        render thread always sees a consistent, complete RateLimitInfo.
     *
     *        noexcept: if make_shared throws (OOM), the old snapshot is retained.
     */
    void set_last_rate_limit_info(const protocols::RateLimitInfo& info) noexcept {
        try {
            auto snapshot = std::make_shared<const protocols::RateLimitInfo>(info);
            std::atomic_store_explicit(
                &last_rate_limit_snapshot_, std::move(snapshot), std::memory_order_release);
        } catch (...) {
            // OOM: silently retain the previous snapshot.
        }
    }

private:
    std::atomic<int32_t> last_prompt_{0};
    std::atomic<int32_t> last_completion_{0};

    // Immutable snapshot of the last rate-limit response.  Written by the
    // streaming thread via set_last_rate_limit_info(); read lock-free by the
    // render thread via get_last_rate_limit_info().
    std::shared_ptr<const protocols::RateLimitInfo> last_rate_limit_snapshot_{
        std::make_shared<const protocols::RateLimitInfo>()};
};

} // namespace core::llm
