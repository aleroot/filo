#pragma once

#include "core/llm/Models.hpp"
#include "core/budget/BudgetTracker.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <ranges>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::session {

// ---------------------------------------------------------------------------
// ModelCallStats — per-model usage accumulator (lock-free once the slot
// exists; slot creation is protected by SessionStats::models_mutex_).
// ---------------------------------------------------------------------------
struct ModelCallStats {
    std::atomic<int32_t> call_count{0};
    std::atomic<int32_t> prompt_tokens{0};
    std::atomic<int32_t> completion_tokens{0};
    std::atomic<int64_t> cost_micro_usd{0};

    ModelCallStats() = default;
    ModelCallStats(const ModelCallStats&) = delete;
    ModelCallStats& operator=(const ModelCallStats&) = delete;
};

// ---------------------------------------------------------------------------
// SessionStats — rich, session-wide metrics singleton.
// Augments BudgetTracker with per-model breakdowns and turn/tool counters.
// All public methods are thread-safe.
// ---------------------------------------------------------------------------
class SessionStats {
public:
    static SessionStats& get_instance() noexcept {
        static SessionStats instance;
        return instance;
    }

    // -----------------------------------------------------------------------
    // Mutation helpers — called from Agent on each completed step.
    // -----------------------------------------------------------------------

    void record_turn(std::string_view model,
                     const core::llm::TokenUsage& usage,
                     bool should_estimate_cost = true) noexcept {
        turn_count_.fetch_add(1, std::memory_order_relaxed);
        if (!model.empty()) {
            ModelCallStats* slot = get_or_create_slot(model);
            slot->call_count.fetch_add(1, std::memory_order_relaxed);
            if (usage.has_data()) {
                slot->prompt_tokens.fetch_add(usage.prompt_tokens,     std::memory_order_relaxed);
                slot->completion_tokens.fetch_add(usage.completion_tokens, std::memory_order_relaxed);
                if (should_estimate_cost) {
                    const auto r = core::budget::rates_for_model(model);
                    const int64_t cost_micro = static_cast<int64_t>(
                        (static_cast<double>(usage.prompt_tokens)     / 1'000'000.0 * r.input_per_m +
                         static_cast<double>(usage.completion_tokens) / 1'000'000.0 * r.output_per_m)
                        * 1'000'000.0);
                    slot->cost_micro_usd.fetch_add(cost_micro, std::memory_order_relaxed);
                }
            }
        }
    }

    void record_tool_call(bool success) noexcept {
        tool_calls_total_.fetch_add(1, std::memory_order_relaxed);
        if (success) tool_calls_success_.fetch_add(1, std::memory_order_relaxed);
    }

    void record_api_call(bool success) noexcept {
        api_calls_total_.fetch_add(1, std::memory_order_relaxed);
        if (success) api_calls_success_.fetch_add(1, std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Reset — call on /clear so stats reflect the new session segment.
    // -----------------------------------------------------------------------
    void reset() noexcept {
        started_at_ = std::chrono::system_clock::now();
        turn_count_.store(0, std::memory_order_release);
        tool_calls_total_.store(0, std::memory_order_release);
        tool_calls_success_.store(0, std::memory_order_release);
        api_calls_total_.store(0, std::memory_order_release);
        api_calls_success_.store(0, std::memory_order_release);
        std::unique_lock lock(models_mutex_);
        per_model_.clear();
    }

    // -----------------------------------------------------------------------
    // Immutable snapshot for reporting and serialization.
    // -----------------------------------------------------------------------
    struct PerModelSnapshot {
        std::string model;
        int32_t call_count;
        int32_t prompt_tokens;
        int32_t completion_tokens;
        double  cost_usd;
    };

    struct Snapshot {
        std::chrono::system_clock::time_point started_at;
        int32_t turn_count;
        int32_t tool_calls_total;
        int32_t tool_calls_success;
        int32_t api_calls_total;
        int32_t api_calls_success;
        std::vector<PerModelSnapshot> per_model;
    };

    [[nodiscard]] Snapshot snapshot() const {
        Snapshot s;
        s.started_at         = started_at_;
        s.turn_count         = turn_count_.load(std::memory_order_acquire);
        s.tool_calls_total   = tool_calls_total_.load(std::memory_order_acquire);
        s.tool_calls_success = tool_calls_success_.load(std::memory_order_acquire);
        s.api_calls_total    = api_calls_total_.load(std::memory_order_acquire);
        s.api_calls_success  = api_calls_success_.load(std::memory_order_acquire);
        {
            std::shared_lock lock(models_mutex_);
            s.per_model.reserve(per_model_.size());
            for (const auto& [name, stats] : per_model_) {
                s.per_model.push_back({
                    .model             = name,
                    .call_count        = stats->call_count.load(std::memory_order_acquire),
                    .prompt_tokens     = stats->prompt_tokens.load(std::memory_order_acquire),
                    .completion_tokens = stats->completion_tokens.load(std::memory_order_acquire),
                    .cost_usd = static_cast<double>(
                        stats->cost_micro_usd.load(std::memory_order_acquire)) / 1'000'000.0,
                });
            }
        }
        // Most-used model first for consistent display ordering.
        std::ranges::sort(s.per_model, [](const auto& a, const auto& b) {
            return a.call_count > b.call_count;
        });
        return s;
    }

    [[nodiscard]] std::chrono::system_clock::time_point started_at() const noexcept {
        return started_at_;
    }

    [[nodiscard]] int32_t turn_count() const noexcept {
        return turn_count_.load(std::memory_order_acquire);
    }

private:
    SessionStats() noexcept : started_at_(std::chrono::system_clock::now()) {}

    // Double-checked locking: shared read first, unique write only on miss.
    ModelCallStats* get_or_create_slot(std::string_view model) noexcept {
        const std::string key{model};
        {
            std::shared_lock lock(models_mutex_);
            if (const auto it = per_model_.find(key); it != per_model_.end()) {
                return it->second.get();
            }
        }
        std::unique_lock lock(models_mutex_);
        if (const auto it = per_model_.find(key); it != per_model_.end()) {
            return it->second.get();
        }
        auto [it, _] = per_model_.emplace(key, std::make_unique<ModelCallStats>());
        return it->second.get();
    }

    std::chrono::system_clock::time_point started_at_;
    std::atomic<int32_t>    turn_count_{0};
    std::atomic<int32_t>    tool_calls_total_{0};
    std::atomic<int32_t>    tool_calls_success_{0};
    std::atomic<int32_t>    api_calls_total_{0};
    std::atomic<int32_t>    api_calls_success_{0};
    mutable std::shared_mutex                                        models_mutex_;
    std::unordered_map<std::string, std::unique_ptr<ModelCallStats>> per_model_;
};

} // namespace core::session
