#include "TokenLedger.hpp"

#include "TokenAccounting.hpp"

#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <utility>

namespace core::budget {

namespace {

[[nodiscard]] bool matches_filter(const TokenLedgerEvent& event,
                                  const TokenLedgerFilter& filter) {
    if (!filter.session_id.empty() && event.session_id != filter.session_id) return false;
    if (!filter.actor.empty() && event.actor != filter.actor) return false;
    if (!filter.model.empty() && event.model != filter.model) return false;
    if (filter.source.has_value() && event.source != *filter.source) return false;
    if (filter.kind.has_value() && event.kind != *filter.kind) return false;
    return true;
}

void normalize_event(TokenLedgerEvent& event) {
    if (event.recorded_at == std::chrono::system_clock::time_point{}) {
        event.recorded_at = std::chrono::system_clock::now();
    }
    if (event.actor.empty()) {
        event.actor = "agent";
    }
    if (event.usage.total_tokens <= 0) {
        event.usage.total_tokens =
            event.usage.prompt_tokens + event.usage.completion_tokens;
    }
}

void add_to_bucket(std::unordered_map<std::string, TokenLedgerBucketSnapshot>& buckets,
                   std::string key,
                   const TokenLedgerEvent& event) {
    if (key.empty()) {
        key = "<unknown>";
    }
    auto& bucket = buckets[key];
    bucket.key = std::move(key);
    ++bucket.event_count;
    bucket.prompt_tokens += event.usage.prompt_tokens;
    bucket.completion_tokens += event.usage.completion_tokens;
    bucket.total_tokens += event.usage.total_tokens;
    bucket.cached_prompt_tokens += event.usage.cached_prompt_tokens;
    bucket.cache_creation_prompt_tokens += event.usage.cache_creation_prompt_tokens;
    bucket.reasoning_tokens += event.usage.reasoning_tokens;
    bucket.cost_micro_usd += event.cost_micro_usd;
}

[[nodiscard]] std::vector<TokenLedgerBucketSnapshot> sorted_buckets(
    std::unordered_map<std::string, TokenLedgerBucketSnapshot> buckets) {
    std::vector<TokenLedgerBucketSnapshot> out;
    out.reserve(buckets.size());
    for (auto& [_, bucket] : buckets) {
        out.push_back(std::move(bucket));
    }
    std::ranges::sort(out, [](const auto& a, const auto& b) {
        if (a.cost_micro_usd != b.cost_micro_usd) {
            return a.cost_micro_usd > b.cost_micro_usd;
        }
        if (a.total_tokens != b.total_tokens) {
            return a.total_tokens > b.total_tokens;
        }
        if (a.event_count != b.event_count) {
            return a.event_count > b.event_count;
        }
        return a.key < b.key;
    });
    return out;
}

} // namespace

std::string_view to_string(TokenLedgerEventKind kind) noexcept {
    switch (kind) {
        case TokenLedgerEventKind::Actual: return "actual";
        case TokenLedgerEventKind::Estimate: return "estimate";
        case TokenLedgerEventKind::Reservation: return "reservation";
        case TokenLedgerEventKind::Adjustment: return "adjustment";
    }
    return "unknown";
}

std::string_view to_string(TokenLedgerSource source) noexcept {
    switch (source) {
        case TokenLedgerSource::ModelCall: return "model_call";
        case TokenLedgerSource::ToolPayload: return "tool_payload";
        case TokenLedgerSource::Subagent: return "subagent";
        case TokenLedgerSource::Compaction: return "compaction";
        case TokenLedgerSource::Gateway: return "gateway";
        case TokenLedgerSource::Unknown: return "unknown";
    }
    return "unknown";
}

int64_t estimate_cost_micro_usd(const core::llm::TokenUsage& usage,
                                std::string_view model,
                                bool should_estimate_cost) noexcept {
    if (!should_estimate_cost || model.empty() || !usage.has_data()) {
        return 0;
    }

    const auto rates = rates_for_model(model);
    return static_cast<int64_t>(
        (static_cast<double>(usage.prompt_tokens) / 1'000'000.0 * rates.input_per_m
         + static_cast<double>(usage.completion_tokens) / 1'000'000.0 * rates.output_per_m)
        * 1'000'000.0);
}

uint64_t TokenLedger::record(TokenLedgerRecordOptions options) {
    TokenLedgerEvent event;
    event.kind = options.kind;
    event.source = options.source;
    event.session_id = std::move(options.session_id);
    event.turn_id = std::move(options.turn_id);
    event.request_id = std::move(options.request_id);
    event.parent_id = std::move(options.parent_id);
    event.actor = std::move(options.actor);
    event.provider = std::move(options.provider);
    event.model = std::move(options.model);
    event.tool_name = std::move(options.tool_name);
    event.note = std::move(options.note);
    event.usage = options.usage;
    event.billable = options.billable;
    event.cost_micro_usd = options.cost_micro_usd.value_or(
        estimate_cost_micro_usd(event.usage, event.model, options.should_estimate_cost));
    return record_event(std::move(event));
}

uint64_t TokenLedger::record_event(TokenLedgerEvent event) {
    normalize_event(event);
    std::lock_guard lock(mutex_);
    event.sequence = next_sequence_++;
    const uint64_t sequence = event.sequence;
    events_.push_back(std::move(event));
    return sequence;
}

void TokenLedger::reset() noexcept {
    std::lock_guard lock(mutex_);
    events_.clear();
    next_sequence_ = 1;
}

void TokenLedger::reset_session(std::string_view session_id) {
    if (session_id.empty()) {
        return;
    }
    std::lock_guard lock(mutex_);
    std::erase_if(events_, [&](const TokenLedgerEvent& event) {
        return event.session_id == session_id;
    });
}

TokenLedgerSnapshot TokenLedger::snapshot(const TokenLedgerFilter& filter) const {
    TokenLedgerSnapshot snapshot;
    std::unordered_map<std::string, TokenLedgerBucketSnapshot> by_source;
    std::unordered_map<std::string, TokenLedgerBucketSnapshot> by_model;
    std::unordered_map<std::string, TokenLedgerBucketSnapshot> by_actor;

    std::lock_guard lock(mutex_);
    for (const auto& event : events_) {
        if (!matches_filter(event, filter)) continue;

        ++snapshot.event_count;
        snapshot.prompt_tokens += event.usage.prompt_tokens;
        snapshot.completion_tokens += event.usage.completion_tokens;
        snapshot.total_tokens += event.usage.total_tokens;
        snapshot.cached_prompt_tokens += event.usage.cached_prompt_tokens;
        snapshot.cache_creation_prompt_tokens += event.usage.cache_creation_prompt_tokens;
        snapshot.reasoning_tokens += event.usage.reasoning_tokens;
        snapshot.cost_micro_usd += event.cost_micro_usd;

        add_to_bucket(by_source, std::string(to_string(event.source)), event);
        add_to_bucket(by_model, event.model, event);
        add_to_bucket(by_actor, event.actor, event);
    }

    snapshot.by_source = sorted_buckets(std::move(by_source));
    snapshot.by_model = sorted_buckets(std::move(by_model));
    snapshot.by_actor = sorted_buckets(std::move(by_actor));
    return snapshot;
}

std::vector<TokenLedgerEvent> TokenLedger::recent_events(
    std::size_t limit,
    const TokenLedgerFilter& filter) const {
    std::vector<TokenLedgerEvent> out;
    if (limit == 0) {
        return out;
    }

    std::lock_guard lock(mutex_);
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (!matches_filter(*it, filter)) continue;
        out.push_back(*it);
        if (out.size() >= limit) break;
    }
    return out;
}

std::size_t TokenLedger::event_count() const noexcept {
    std::lock_guard lock(mutex_);
    return events_.size();
}

} // namespace core::budget
