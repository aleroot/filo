#pragma once

#include "core/llm/Models.hpp"
#include "core/budget/TokenAccounting.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <ranges>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <limits>
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

struct ToolCallStats {
    std::atomic<int32_t> call_count{0};
    std::atomic<int32_t> success_count{0};
    std::atomic<int32_t> argument_tokens{0};
    std::atomic<int32_t> result_tokens{0};
    std::atomic<int32_t> attributed_completion_tokens{0};
    std::atomic<int64_t> attributed_cost_micro_usd{0};

    ToolCallStats() = default;
    ToolCallStats(const ToolCallStats&) = delete;
    ToolCallStats& operator=(const ToolCallStats&) = delete;
};

// ---------------------------------------------------------------------------
// SessionStats — rich, session-wide metrics singleton.
// Tracks per-model breakdowns and turn/tool counters.
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
        if (model.empty()) return;

        const std::string key{model};
        auto update_slot = [&](ModelCallStats& slot) noexcept {
            slot.call_count.fetch_add(1, std::memory_order_relaxed);
            if (!usage.has_data()) return;

            slot.prompt_tokens.fetch_add(usage.prompt_tokens, std::memory_order_relaxed);
            slot.completion_tokens.fetch_add(usage.completion_tokens, std::memory_order_relaxed);
            if (should_estimate_cost) {
                const auto r = core::budget::rates_for_model(model);
                const int64_t cost_micro = static_cast<int64_t>(
                    (static_cast<double>(usage.prompt_tokens)     / 1'000'000.0 * r.input_per_m +
                     static_cast<double>(usage.completion_tokens) / 1'000'000.0 * r.output_per_m)
                    * 1'000'000.0);
                slot.cost_micro_usd.fetch_add(cost_micro, std::memory_order_relaxed);
            }
        };

        {
            std::shared_lock lock(models_mutex_);
            if (const auto it = per_model_.find(key); it != per_model_.end()) {
                update_slot(*it->second);
                return;
            }
        }

        std::unique_lock lock(models_mutex_);
        auto [it, _] = per_model_.try_emplace(key, std::make_unique<ModelCallStats>());
        update_slot(*it->second);
    }

    static int32_t estimate_payload_tokens(std::string_view payload) noexcept {
        if (payload.empty()) return 0;
        constexpr std::size_t kCharsPerToken = 4;
        constexpr std::size_t kMaxInt32 =
            static_cast<std::size_t>(std::numeric_limits<int32_t>::max());
        const std::size_t estimate = (payload.size() + kCharsPerToken - 1) / kCharsPerToken;
        return static_cast<int32_t>(std::min(estimate, kMaxInt32));
    }

    static int64_t estimate_completion_cost_micro_usd(std::string_view model,
                                                      int32_t completion_tokens,
                                                      bool should_estimate_cost) noexcept {
        if (!should_estimate_cost || model.empty() || completion_tokens <= 0) return 0;
        const auto rates = core::budget::rates_for_model(model);
        return static_cast<int64_t>(
            static_cast<double>(completion_tokens) * rates.output_per_m);
    }

    template <typename Integer>
    static Integer split_integer_share(Integer total,
                                       std::size_t part_count,
                                       std::size_t part_index) noexcept {
        if (total <= 0 || part_count == 0 || part_index >= part_count) return 0;
        const auto count = static_cast<Integer>(part_count);
        const Integer base = total / count;
        const Integer remainder = total % count;
        return base + (static_cast<Integer>(part_index) < remainder ? 1 : 0);
    }

    void record_tool_call(bool success) noexcept {
        record_tool_call(std::string_view{}, success, 0, 0);
    }

    void record_tool_call(std::string_view tool_name,
                          bool success,
                          int32_t argument_tokens = 0,
                          int32_t result_tokens = 0,
                          int32_t attributed_completion_tokens = 0,
                          int64_t attributed_cost_micro_usd = 0) noexcept {
        tool_calls_total_.fetch_add(1, std::memory_order_relaxed);
        if (success) tool_calls_success_.fetch_add(1, std::memory_order_relaxed);
        if (!tool_name.empty()) {
            const std::string key{tool_name};
            auto update_slot = [&](ToolCallStats& slot) noexcept {
                slot.call_count.fetch_add(1, std::memory_order_relaxed);
                if (success) slot.success_count.fetch_add(1, std::memory_order_relaxed);
                if (argument_tokens > 0) {
                    slot.argument_tokens.fetch_add(argument_tokens, std::memory_order_relaxed);
                }
                if (result_tokens > 0) {
                    slot.result_tokens.fetch_add(result_tokens, std::memory_order_relaxed);
                }
                if (attributed_completion_tokens > 0) {
                    slot.attributed_completion_tokens.fetch_add(
                        attributed_completion_tokens, std::memory_order_relaxed);
                }
                if (attributed_cost_micro_usd > 0) {
                    slot.attributed_cost_micro_usd.fetch_add(
                        attributed_cost_micro_usd, std::memory_order_relaxed);
                }
            };

            {
                std::shared_lock lock(tools_mutex_);
                if (const auto it = per_tool_.find(key); it != per_tool_.end()) {
                    update_slot(*it->second);
                    return;
                }
            }

            std::unique_lock lock(tools_mutex_);
            auto [it, _] = per_tool_.try_emplace(key, std::make_unique<ToolCallStats>());
            update_slot(*it->second);
        }
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
        std::unique_lock tool_lock(tools_mutex_);
        per_tool_.clear();
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

    struct PerToolSnapshot {
        std::string tool;
        int32_t call_count;
        int32_t success_count;
        int32_t argument_tokens;
        int32_t result_tokens;
        int32_t attributed_completion_tokens;
        double  attributed_cost_usd;
    };

    struct Snapshot {
        std::chrono::system_clock::time_point started_at;
        int32_t turn_count;
        int32_t tool_calls_total;
        int32_t tool_calls_success;
        int32_t api_calls_total;
        int32_t api_calls_success;
        std::vector<PerModelSnapshot> per_model;
        std::vector<PerToolSnapshot> per_tool;
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
        {
            std::shared_lock lock(tools_mutex_);
            s.per_tool.reserve(per_tool_.size());
            for (const auto& [name, stats] : per_tool_) {
                s.per_tool.push_back({
                    .tool            = name,
                    .call_count      = stats->call_count.load(std::memory_order_acquire),
                    .success_count   = stats->success_count.load(std::memory_order_acquire),
                    .argument_tokens = stats->argument_tokens.load(std::memory_order_acquire),
                    .result_tokens   = stats->result_tokens.load(std::memory_order_acquire),
                    .attributed_completion_tokens =
                        stats->attributed_completion_tokens.load(std::memory_order_acquire),
                    .attributed_cost_usd =
                        static_cast<double>(
                            stats->attributed_cost_micro_usd.load(std::memory_order_acquire))
                        / 1'000'000.0,
                });
            }
        }
        // Most-used model first for consistent display ordering.
        std::ranges::sort(s.per_model, [](const auto& a, const auto& b) {
            return a.call_count > b.call_count;
        });
        // Tool output is the part most likely to inflate later prompt tokens.
        std::ranges::sort(s.per_tool, [](const auto& a, const auto& b) {
            if (a.attributed_cost_usd != b.attributed_cost_usd) {
                return a.attributed_cost_usd > b.attributed_cost_usd;
            }
            if (a.attributed_completion_tokens != b.attributed_completion_tokens) {
                return a.attributed_completion_tokens > b.attributed_completion_tokens;
            }
            const int32_t a_payload = a.argument_tokens + a.result_tokens;
            const int32_t b_payload = b.argument_tokens + b.result_tokens;
            if (a_payload != b_payload) return a_payload > b_payload;
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

    std::chrono::system_clock::time_point started_at_;
    std::atomic<int32_t>    turn_count_{0};
    std::atomic<int32_t>    tool_calls_total_{0};
    std::atomic<int32_t>    tool_calls_success_{0};
    std::atomic<int32_t>    api_calls_total_{0};
    std::atomic<int32_t>    api_calls_success_{0};
    mutable std::shared_mutex                                        models_mutex_;
    std::unordered_map<std::string, std::unique_ptr<ModelCallStats>> per_model_;
    mutable std::shared_mutex                                       tools_mutex_;
    std::unordered_map<std::string, std::unique_ptr<ToolCallStats>> per_tool_;
};

} // namespace core::session
