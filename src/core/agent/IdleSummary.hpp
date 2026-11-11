#pragma once

#include "HistoryCompactionPlanner.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace core::agent {

/// A compaction summary written ahead while a conversation idles, kept aside
/// for the history revision it describes. Any later revision makes it stale:
/// it is then neither reported nor handed out. Holds at most one summary and
/// at most one write in flight. Not synchronised; its owner guards it.
class IdleSummary {
public:
    struct Summary {
        std::string text;
        HistoryCompactionPlan plan;
        std::string active_skill_context;
    };

    /// Claims the write for `revision`. False while another write runs or a
    /// summary for that revision is already kept.
    [[nodiscard]] bool begin(std::uint64_t revision) noexcept {
        if (writing_ || holds(revision)) return false;
        writing_ = true;
        kept_.reset();
        return true;
    }

    void keep(std::uint64_t revision, Summary summary) {
        kept_.emplace(revision, std::move(summary));
    }

    /// Ends the write claimed by begin(), whatever its outcome.
    void finish() noexcept { writing_ = false; }

    [[nodiscard]] bool holds(std::uint64_t revision) const noexcept {
        return kept_.has_value() && kept_->first == revision;
    }

    /// The summary when it still describes `revision`. Empty afterwards
    /// either way.
    [[nodiscard]] std::optional<Summary> take(std::uint64_t revision) {
        auto kept = std::exchange(kept_, std::nullopt);
        if (!kept.has_value() || kept->first != revision) return std::nullopt;
        return std::move(kept->second);
    }

    void clear() noexcept { kept_.reset(); }

private:
    std::optional<std::pair<std::uint64_t, Summary>> kept_;
    bool writing_ = false;
};

} // namespace core::agent
