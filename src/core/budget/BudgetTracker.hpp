#pragma once

#include "TokenAccounting.hpp"
#include "TokenLedger.hpp"

#include "../llm/Models.hpp"

#include <cstdint>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace core::budget {

struct BudgetRecordContext {
    std::string session_id;
    std::string turn_id;
    std::string request_id;
    std::string parent_id;
    std::string actor = "agent";
    std::string provider;
    std::string model;
    TokenLedgerSource source = TokenLedgerSource::ModelCall;
    std::string note;
};

// BudgetTracker is the legacy presentation facade. TokenLedger is the source
// of truth for token movement, attribution, and cost. New accounting writes
// should record directly to TokenLedger.
class BudgetTracker {
public:
    static BudgetTracker& get_instance() noexcept {
        static BudgetTracker instance;
        return instance;
    }

    void set_session_id(std::string session_id) {
        std::lock_guard lock(mutex_);
        session_id_ = std::move(session_id);
    }

    [[nodiscard]] std::string session_id() const {
        std::lock_guard lock(mutex_);
        return session_id_;
    }

    // Compatibility shim for older tests/callers. Production accounting writes
    // directly to TokenLedger.
    void record(const core::llm::TokenUsage& usage,
                std::string_view model = "",
                bool should_estimate_cost = true) noexcept {
        BudgetRecordContext context;
        context.session_id = session_id();
        context.model = std::string(model);
        record(usage, std::move(context), should_estimate_cost);
    }

    void record(const core::llm::TokenUsage& usage,
                BudgetRecordContext context,
                bool should_estimate_cost = true) noexcept {
        if (!usage.has_data()) return;
        try {
            if (!context.session_id.empty()) {
                set_session_id(context.session_id);
            }
            TokenLedger::get_instance().record({
                .kind = TokenLedgerEventKind::Actual,
                .source = context.source,
                .session_id = std::move(context.session_id),
                .turn_id = std::move(context.turn_id),
                .request_id = std::move(context.request_id),
                .parent_id = std::move(context.parent_id),
                .actor = std::move(context.actor),
                .provider = std::move(context.provider),
                .model = std::move(context.model),
                .note = std::move(context.note),
                .usage = usage,
                .should_estimate_cost = should_estimate_cost,
                .billable = should_estimate_cost,
            });
        } catch (...) {
        }
    }

    void reset_session() noexcept {
        try {
            const std::string scoped_session = session_id();
            if (scoped_session.empty()) {
                TokenLedger::get_instance().reset();
            } else {
                TokenLedger::get_instance().reset_session(scoped_session);
            }
        } catch (...) {
        }
    }

    [[nodiscard]] core::llm::TokenUsage session_total() const noexcept {
        try {
            const auto snapshot = actual_snapshot();
            const int32_t prompt = static_cast<int32_t>(snapshot.prompt_tokens);
            const int32_t completion = static_cast<int32_t>(snapshot.completion_tokens);
            return {
                .prompt_tokens = prompt,
                .completion_tokens = completion,
                .total_tokens = prompt + completion,
            };
        } catch (...) {
            return {};
        }
    }

    [[nodiscard]] core::llm::TokenUsage last_turn() const noexcept {
        try {
            const auto events = TokenLedger::get_instance().recent_events(
                1,
                actual_filter());
            if (events.empty()) {
                return {};
            }
            return events.front().usage;
        } catch (...) {
            return {};
        }
    }

    [[nodiscard]] double session_cost_usd() const noexcept {
        try {
            return actual_snapshot().cost_usd();
        } catch (...) {
            return 0.0;
        }
    }

    [[nodiscard]] int32_t context_remaining_pct(std::string_view model) const noexcept {
        const int32_t used = last_turn().prompt_tokens;
        if (used <= 0) return -1;
        const int64_t window = context_window_for_model(model);
        const int64_t remaining = window - static_cast<int64_t>(used);
        if (remaining <= 0) return 0;
        return static_cast<int32_t>(
            (static_cast<double>(remaining) / static_cast<double>(window)) * 100.0);
    }

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

    [[nodiscard]] TokenLedgerFilter actual_filter() const {
        return TokenLedgerFilter{
            .session_id = session_id(),
            .kind = TokenLedgerEventKind::Actual,
        };
    }

    [[nodiscard]] TokenLedgerSnapshot actual_snapshot() const {
        return TokenLedger::get_instance().snapshot(
            actual_filter());
    }

    mutable std::mutex mutex_;
    std::string session_id_;
};

} // namespace core::budget
