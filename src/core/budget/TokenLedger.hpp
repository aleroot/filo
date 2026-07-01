#pragma once

#include "../llm/Models.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::budget {

enum class TokenLedgerEventKind {
    Actual,
    Estimate,
    Reservation,
    Adjustment,
};

enum class TokenLedgerSource {
    ModelCall,
    ToolPayload,
    Subagent,
    Compaction,
    Gateway,
    Unknown,
};

[[nodiscard]] std::string_view to_string(TokenLedgerEventKind kind) noexcept;
[[nodiscard]] std::string_view to_string(TokenLedgerSource source) noexcept;

struct TokenLedgerEvent {
    uint64_t sequence = 0;
    std::chrono::system_clock::time_point recorded_at{};

    TokenLedgerEventKind kind = TokenLedgerEventKind::Actual;
    TokenLedgerSource source = TokenLedgerSource::Unknown;

    std::string session_id;
    std::string turn_id;
    std::string request_id;
    std::string parent_id;
    std::string actor;
    std::string provider;
    std::string model;
    std::string tool_name;
    std::string note;

    core::llm::TokenUsage usage;
    int64_t cost_micro_usd = 0;
    bool billable = true;
};

struct TokenLedgerFilter {
    std::string session_id{};
    std::string actor{};
    std::string model{};
    std::optional<TokenLedgerSource> source{};
    std::optional<TokenLedgerEventKind> kind{};
};

struct TokenLedgerRecordOptions {
    TokenLedgerEventKind kind = TokenLedgerEventKind::Actual;
    TokenLedgerSource source = TokenLedgerSource::ModelCall;

    std::string session_id{};
    std::string turn_id{};
    std::string request_id{};
    std::string parent_id{};
    std::string actor = "agent";
    std::string provider{};
    std::string model{};
    std::string tool_name{};
    std::string note{};

    core::llm::TokenUsage usage{};
    bool should_estimate_cost = true;
    bool billable = true;
    std::optional<int64_t> cost_micro_usd{};
};

struct TokenLedgerBucketSnapshot {
    std::string key;
    int32_t event_count = 0;
    int64_t prompt_tokens = 0;
    int64_t completion_tokens = 0;
    int64_t total_tokens = 0;
    int64_t cost_micro_usd = 0;

    [[nodiscard]] double cost_usd() const noexcept {
        return static_cast<double>(cost_micro_usd) / 1'000'000.0;
    }
};

struct TokenLedgerSnapshot {
    int32_t event_count = 0;
    int64_t prompt_tokens = 0;
    int64_t completion_tokens = 0;
    int64_t total_tokens = 0;
    int64_t cost_micro_usd = 0;
    std::vector<TokenLedgerBucketSnapshot> by_source;
    std::vector<TokenLedgerBucketSnapshot> by_model;
    std::vector<TokenLedgerBucketSnapshot> by_actor;

    [[nodiscard]] double cost_usd() const noexcept {
        return static_cast<double>(cost_micro_usd) / 1'000'000.0;
    }
};

class TokenLedger {
public:
    TokenLedger() = default;

    uint64_t record(TokenLedgerRecordOptions options);
    uint64_t record_event(TokenLedgerEvent event);

    void reset() noexcept;
    void reset_session(std::string_view session_id);

    [[nodiscard]] TokenLedgerSnapshot snapshot(
        const TokenLedgerFilter& filter = {}) const;

    [[nodiscard]] std::vector<TokenLedgerEvent> recent_events(
        std::size_t limit,
        const TokenLedgerFilter& filter = {}) const;

    [[nodiscard]] std::size_t event_count() const noexcept;

private:
    mutable std::mutex mutex_;
    uint64_t next_sequence_ = 1;
    std::vector<TokenLedgerEvent> events_;
};

[[nodiscard]] int64_t estimate_cost_micro_usd(
    const core::llm::TokenUsage& usage,
    std::string_view model,
    bool should_estimate_cost) noexcept;

} // namespace core::budget
