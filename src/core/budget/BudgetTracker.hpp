#pragma once

#include "../llm/Models.hpp"
#include <atomic>
#include <cctype>
#include <string>
#include <string_view>
#include <format>
#include <cstdint>

namespace core::budget {

[[nodiscard]] inline bool has_1m_context_suffix(std::string_view model) noexcept {
    std::size_t end = model.size();
    while (end > 0
           && std::isspace(static_cast<unsigned char>(model[end - 1]))) {
        --end;
    }
    if (end < 4) return false;
    const std::size_t base = end - 4;
    return model[base] == '['
        && model[base + 1] == '1'
        && (model[base + 2] == 'm' || model[base + 2] == 'M')
        && model[base + 3] == ']';
}

// ---------------------------------------------------------------------------
// Context window sizes in tokens, keyed by model name substring.
// ---------------------------------------------------------------------------
[[nodiscard]] inline int64_t context_window_for_model(std::string_view model) noexcept {
    // Claude-style explicit [1m] opt-in.
    if (has_1m_context_suffix(model)) return 1'000'000;

    // xAI Grok — check specific variants before generic prefixes
    if (model.find("grok-code-fast-1") != std::string_view::npos) return   131'072;
    if (model.find("grok-4.1")         != std::string_view::npos) return 2'097'152;
    if (model.find("grok-4")           != std::string_view::npos) return   256'000;
    if (model.find("grok-3-mini")      != std::string_view::npos) return   131'072;
    if (model.find("grok-3")           != std::string_view::npos) return   131'072;
    if (model.find("grok-2")           != std::string_view::npos) return   131'072;
    // Kimi / Moonshot
    if (model.find("moonshot-v1-8k")   != std::string_view::npos) return     8'192;
    if (model.find("moonshot-v1-32k")  != std::string_view::npos) return    32'768;
    if (model.find("moonshot-v1-128k") != std::string_view::npos) return   128'000;
    if (model.find("kimi-k2.5")        != std::string_view::npos) return   256'000;
    if (model.find("kimi-k2-5")        != std::string_view::npos) return   256'000;
    if (model.find("kimi-for-coding")  != std::string_view::npos) return   256'000;
    if (model.find("kimi-k2")          != std::string_view::npos) return   256'000;
    if (model.find("kimi-for-university") != std::string_view::npos) return 256'000;
    if (model.find("kimi-k1.5")        != std::string_view::npos) return   256'000;
    if (model.find("kimi-")            != std::string_view::npos) return   128'000;
    // Anthropic Claude — check specific versions before generic "claude"
    // Claude 4.6 models support 1M token context (beta)
    if (model.find("claude-opus-4-6")  != std::string_view::npos) return 1'000'000;
    if (model.find("claude-sonnet-4-6") != std::string_view::npos) return 1'000'000;
    // Claude 4.5 and earlier models have 200K context
    if (model.find("claude-opus-4-5")  != std::string_view::npos) return   200'000;
    if (model.find("claude-sonnet-4-5") != std::string_view::npos) return   200'000;
    if (model.find("claude-haiku-4-5") != std::string_view::npos) return   200'000;
    if (model.find("claude-opus-4")    != std::string_view::npos) return   200'000;
    if (model.find("claude-sonnet-4")  != std::string_view::npos) return   200'000;
    if (model.find("claude-haiku-4")   != std::string_view::npos) return   200'000;
    // Legacy Claude 3.x models
    if (model.find("claude-3")         != std::string_view::npos) return   200'000;
    // Generic claude fallback
    if (model.find("claude")           != std::string_view::npos) return   200'000;
    // OpenAI — gpt-5 before gpt-4 variants
    if (model.find("gpt-5.4")          != std::string_view::npos) return   200'000;
    if (model.find("gpt-5")            != std::string_view::npos) return   200'000;
    if (model.find("gpt-4o")           != std::string_view::npos) return   128'000;
    if (model.find("gpt-4-turbo")      != std::string_view::npos) return   128'000;
    if (model.find("gpt-4")            != std::string_view::npos) return     8'192;
    if (model.find("gpt-3.5")          != std::string_view::npos) return    16'385;
    // Google Gemini
    if (model.find("gemini-2.5")       != std::string_view::npos) return 1'048'576;
    if (model.find("gemini-2.0")       != std::string_view::npos) return 1'048'576;
    if (model.find("gemini-1.5")       != std::string_view::npos) return 2'097'152;
    // Default: conservative 128K
    return 128'000;
}

// ---------------------------------------------------------------------------
// Cost table — input/output rates in USD per 1 million tokens.
// ---------------------------------------------------------------------------
struct ModelRates {
    double input_per_m  = 2.00;  // USD / 1M prompt tokens
    double output_per_m = 8.00;  // USD / 1M completion tokens
};

[[nodiscard]] inline ModelRates rates_for_model(std::string_view model) noexcept {
    // xAI Grok
    if (model.find("grok-code-fast-1") != std::string_view::npos) return { 0.20,  1.50 };
    if (model.find("grok-4-fast")      != std::string_view::npos) return { 0.20,  0.50 };
    if (model.find("grok-4.1-fast")    != std::string_view::npos) return { 0.20,  0.50 };
    if (model.find("grok-3-mini") != std::string_view::npos) return { 0.30,  0.50 };
    if (model.find("grok-3")      != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("grok-2")      != std::string_view::npos) return { 2.00, 10.00 };
    if (model.find("grok-4")      != std::string_view::npos) return { 3.00, 15.00 };
    // Anthropic Claude — check specific versions before generic types
    // Claude 4.6 models (latest as of Feb 2026)
    if (model.find("claude-opus-4-6")  != std::string_view::npos) return { 5.00, 25.00 };
    if (model.find("claude-sonnet-4-6") != std::string_view::npos) return { 3.00, 15.00 };
    // Claude 4.5 models
    if (model.find("claude-opus-4-5")  != std::string_view::npos) return { 5.00, 25.00 };
    if (model.find("claude-sonnet-4-5") != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("claude-haiku-4-5") != std::string_view::npos) return { 1.00,  5.00 };
    // Claude 4.0/4.1 models (older)
    if (model.find("claude-opus-4")    != std::string_view::npos) return {15.00, 75.00 };
    if (model.find("claude-sonnet-4")  != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("claude-haiku-4")   != std::string_view::npos) return { 1.00,  5.00 };
    // Legacy Claude 3.x models
    if (model.find("claude-3-opus")    != std::string_view::npos) return {15.00, 75.00 };
    if (model.find("claude-3-5-sonnet") != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("claude-3-5-haiku") != std::string_view::npos) return { 0.80,  4.00 };
    if (model.find("claude-3-sonnet") != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("claude-3-haiku")  != std::string_view::npos) return { 0.25,  1.25 };
    // Generic type matching (fallback for unknown specific versions)
    if (model.find("opus")        != std::string_view::npos) return {15.00, 75.00 };
    if (model.find("sonnet")      != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("haiku")       != std::string_view::npos) return { 0.80,  4.00 };
    // OpenAI GPT
    if (model.find("gpt-5.4")      != std::string_view::npos) return { 2.50, 10.00 };
    if (model.find("gpt-5")        != std::string_view::npos) return { 2.50, 10.00 };
    if (model.find("gpt-4o")      != std::string_view::npos) return { 2.50, 10.00 };
    if (model.find("gpt-4")       != std::string_view::npos) return {30.00, 60.00 };
    if (model.find("gpt-3.5")     != std::string_view::npos) return { 0.50,  1.50 };
    // Gemini
    if (model.find("gemini-2.5")  != std::string_view::npos) return { 1.25,  5.00 };
    if (model.find("gemini-2.0")  != std::string_view::npos) return { 0.10,  0.40 };
    if (model.find("gemini-1.5")  != std::string_view::npos) return { 0.075, 0.30 };
    // Default: unknown model
    return { 2.00, 8.00 };
}

// ---------------------------------------------------------------------------
// BudgetTracker — session-wide token + cost accumulator.
// All public methods are thread-safe via atomic operations.
// ---------------------------------------------------------------------------
class BudgetTracker {
public:
    static BudgetTracker& get_instance() noexcept {
        static BudgetTracker instance;
        return instance;
    }

    // Called by Agent after each completed LLM step.
    void record(const core::llm::TokenUsage& usage,
                std::string_view model = "",
                bool should_estimate_cost = true) noexcept {
        if (!usage.has_data()) return;
        session_prompt_.fetch_add(usage.prompt_tokens,     std::memory_order_relaxed);
        session_completion_.fetch_add(usage.completion_tokens, std::memory_order_relaxed);
        last_prompt_.store(usage.prompt_tokens,     std::memory_order_release);
        last_completion_.store(usage.completion_tokens, std::memory_order_release);
        // cost accumulation using a lock-free double accumulation trick
        // (add integer micro-USD to avoid floating-point race)
        if (should_estimate_cost && !model.empty()) {
            auto r = rates_for_model(model);
            int64_t cost_micro = static_cast<int64_t>(
                (static_cast<double>(usage.prompt_tokens)     / 1'000'000.0 * r.input_per_m +
                 static_cast<double>(usage.completion_tokens) / 1'000'000.0 * r.output_per_m)
                * 1'000'000.0);  // store in micro-USD
            session_cost_micro_.fetch_add(cost_micro, std::memory_order_relaxed);
        }
    }

    void reset_session() noexcept {
        session_prompt_.store(0, std::memory_order_release);
        session_completion_.store(0, std::memory_order_release);
        session_cost_micro_.store(0, std::memory_order_release);
        last_prompt_.store(0, std::memory_order_release);
        last_completion_.store(0, std::memory_order_release);
    }

    [[nodiscard]] core::llm::TokenUsage session_total() const noexcept {
        const int32_t p = static_cast<int32_t>(session_prompt_.load(std::memory_order_acquire));
        const int32_t c = static_cast<int32_t>(session_completion_.load(std::memory_order_acquire));
        return { .prompt_tokens = p, .completion_tokens = c, .total_tokens = p + c };
    }

    [[nodiscard]] core::llm::TokenUsage last_turn() const noexcept {
        const int32_t p = last_prompt_.load(std::memory_order_acquire);
        const int32_t c = last_completion_.load(std::memory_order_acquire);
        return { .prompt_tokens = p, .completion_tokens = c, .total_tokens = p + c };
    }

    [[nodiscard]] double session_cost_usd() const noexcept {
        return static_cast<double>(session_cost_micro_.load(std::memory_order_acquire))
               / 1'000'000.0;
    }

    // Percentage of context window remaining based on the last turn's prompt tokens.
    // Returns -1 if no usage data is available yet.
    [[nodiscard]] int32_t context_remaining_pct(std::string_view model) const noexcept {
        const int32_t used = last_prompt_.load(std::memory_order_acquire);
        if (used <= 0) return -1;
        const int64_t window = context_window_for_model(model);
        const int64_t remaining = window - static_cast<int64_t>(used);
        if (remaining <= 0) return 0;
        return static_cast<int32_t>(
            (static_cast<double>(remaining) / static_cast<double>(window)) * 100.0);
    }

    // Human-readable summary for the status bar.
    // Format: "↑12.3k ↓4.5k  $0.018"
    // ↑ = prompt tokens going UP to the LLM (input/sent)
    // ↓ = completion tokens coming DOWN from the LLM (output/received)
    [[nodiscard]] std::string status_string() const {
        auto total = session_total();
        if (!total.has_data()) return "";
        auto fmt_k = [](int32_t n) -> std::string {
            if (n >= 1000) return std::format("{:.1f}k", n / 1000.0);
            return std::to_string(n);
        };
        double cost = session_cost_usd();
        if (cost > 0.0) {
            return std::format("↑{} ↓{}  ${:.4f}",
                fmt_k(total.prompt_tokens), fmt_k(total.completion_tokens), cost);
        }
        return std::format("↑{} ↓{}", fmt_k(total.prompt_tokens), fmt_k(total.completion_tokens));
    }

private:
    BudgetTracker() noexcept = default;

    std::atomic<int64_t> session_prompt_{0};
    std::atomic<int64_t> session_completion_{0};
    std::atomic<int64_t> session_cost_micro_{0};
    std::atomic<int32_t> last_prompt_{0};
    std::atomic<int32_t> last_completion_{0};
};

} // namespace core::budget
