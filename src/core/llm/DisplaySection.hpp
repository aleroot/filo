#pragma once

#include <string>

namespace core::llm {

/// Display-only part of an assistant message that the transcript renders on
/// its own: markdown, optionally followed by collapsed details (a review
/// comment and its "Why this was flagged"). Persisted for resume and never
/// sent upstream; the message's `content` stays the complete text the model
/// reads, so a host that ignores sections loses nothing.
struct DisplaySection {
    std::string markdown;
    std::string details_summary = {};
    std::string details_markdown = {};

    [[nodiscard]] bool has_details() const noexcept {
        return !details_summary.empty() && !details_markdown.empty();
    }

    bool operator==(const DisplaySection&) const = default;
};

} // namespace core::llm
