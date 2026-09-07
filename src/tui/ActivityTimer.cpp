#include "ActivityTimer.hpp"

#include <array>
#include <cstdint>
#include <format>
#include <string_view>

namespace tui {

namespace {

/// One field of a decomposed duration.
struct DurationField {
    int64_t          value;
    std::string_view suffix;
};

using DurationFields = std::array<DurationField, 4>;

/// Days carried in 64 bits. std::chrono::days is only required to hold 25
/// bits and is a 32-bit int in practice, so casting a 64-bit seconds count
/// straight into it truncates silently at the extreme; the finer fields are
/// each bounded by their parent's carry and need no such widening.
using Days64 = std::chrono::duration<int64_t, std::chrono::days::period>;

/// Splits a duration into days/hours/minutes/seconds. Each field takes only
/// the remainder left by the coarser one, so no field can hold a value that
/// belongs in its parent: a "200m" span is not representable here. The
/// arithmetic is delegated to std::chrono rather than written out with 3600
/// and 86400 literals, which is where unit bugs come from in the first place.
[[nodiscard]] DurationFields decompose(std::chrono::seconds elapsed) {
    using namespace std::chrono;
    const auto d = duration_cast<Days64>(elapsed);
    const auto h = duration_cast<hours>(elapsed - d);
    const auto m = duration_cast<minutes>(elapsed - d - h);
    const auto s = elapsed - d - h - m;
    return DurationFields{{
        {d.count(), "d"},
        {h.count(), "h"},
        {m.count(), "m"},
        {s.count(), "s"},
    }};
}

/// Index of the coarsest non-zero field, or seconds for a sub-minute span.
[[nodiscard]] std::size_t leading_field(const DurationFields& fields) {
    for (std::size_t i = 0; i + 1 < fields.size(); ++i) {
        if (fields[i].value > 0) return i;
    }
    return fields.size() - 1;
}

/// Timer style: the leading unit, then every finer unit zero-padded.
[[nodiscard]] std::string render_precise(const DurationFields& fields,
                                         std::size_t lead) {
    std::string out = std::format("{}{}", fields[lead].value, fields[lead].suffix);
    for (std::size_t i = lead + 1; i < fields.size(); ++i) {
        out += std::format(" {:02}{}", fields[i].value, fields[i].suffix);
    }
    return out;
}

/// Human style: the leading unit plus, at most, the one immediately below it.
/// Because the pair is always adjacent, hours are followed by minutes and
/// never by seconds; the "drop the seconds past an hour" rule is a
/// consequence of the structure instead of a case that can be forgotten.
[[nodiscard]] std::string render_humanized(const DurationFields& fields,
                                           std::size_t lead) {
    const DurationField& head = fields[lead];
    if (lead + 1 == fields.size()) {
        return std::format("{}{}", head.value, head.suffix);
    }
    const DurationField& tail = fields[lead + 1];
    if (tail.value == 0) {
        return std::format("{}{}", head.value, head.suffix);
    }
    // Seconds are the one tail worth zero-padding: they change every frame,
    // so a fixed width keeps the surrounding layout from shifting, and "6m 6s"
    // reads like a truncation where "6m 06s" reads like a clock.
    return tail.suffix == "s"
        ? std::format("{}{} {:02}{}", head.value, head.suffix,
                      tail.value, tail.suffix)
        : std::format("{}{} {}{}", head.value, head.suffix,
                      tail.value, tail.suffix);
}

} // namespace

void ActivityTimerRegistry::start(std::string_view operation_id) {
    start_at(operation_id, Clock::now());
}

void ActivityTimerRegistry::start_at(std::string_view operation_id, Clock::time_point start_time) {
    if (operation_id.empty()) {
        return;
    }
    std::lock_guard lock(mutex_);
    starts_[std::string(operation_id)] = start_time;
}

void ActivityTimerRegistry::stop(std::string_view operation_id) {
    if (operation_id.empty()) {
        return;
    }
    std::lock_guard lock(mutex_);
    starts_.erase(std::string(operation_id));
}

void ActivityTimerRegistry::clear() {
    std::lock_guard lock(mutex_);
    starts_.clear();
}

std::optional<std::chrono::seconds> ActivityTimerRegistry::elapsed(
    std::string_view operation_id,
    Clock::time_point now) const {
    if (operation_id.empty()) {
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);
    const auto it = starts_.find(std::string(operation_id));
    if (it == starts_.end()) {
        return std::nullopt;
    }

    auto delta = std::chrono::duration_cast<std::chrono::seconds>(now - it->second);
    if (delta < std::chrono::seconds::zero()) {
        delta = std::chrono::seconds::zero();
    }
    return delta;
}

std::string format_elapsed_compact(std::chrono::seconds elapsed,
                                   ElapsedFormat format) {
    if (elapsed < std::chrono::seconds::zero()) {
        elapsed = std::chrono::seconds::zero();
    }

    const DurationFields fields = decompose(elapsed);
    const std::size_t    lead   = leading_field(fields);
    return format == ElapsedFormat::humanized
        ? render_humanized(fields, lead)
        : render_precise(fields, lead);
}

} // namespace tui
