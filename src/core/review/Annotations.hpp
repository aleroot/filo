#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simdjson::dom { class object; }

namespace core::review {

inline constexpr std::size_t kMaxReviewDocumentBytes = 16 * 1024 * 1024;

/// A user comment and its anchor in one revision of a comparison.
struct Annotation {
    std::string id;
    std::string path;
    std::string side;
    int start_line = 0;
    int end_line = 0;
    std::string selected_text;
    std::string comment;
    std::string tone = "neutral";
    std::string revision;
    std::string anchor_state = "attached";
    std::vector<std::string> context_before = {};
    std::vector<std::string> context_after = {};

    bool operator==(const Annotation&) const = default;
};

[[nodiscard]] std::string encode_review_request(
    std::string_view patch, std::span<const Annotation> annotations);

/// A response is a complete snapshot. Invalid responses must not replace it.
[[nodiscard]] std::expected<std::vector<Annotation>, std::string>
parse_review_response(std::string_view json);

/// Render comments for the prompt draft.
[[nodiscard]] std::string render_annotation_prompt(std::span<const Annotation> annotations);

/// What a returned review means for the turn that holds the comments.
struct ReviewMerge {
    /// The turn's comments after the review.
    std::vector<Annotation> kept;
    /// Comments the turn did not hold before: the only ones the prompt gets,
    /// so each comment reaches the model once, as in Lampo's own chat.
    std::vector<Annotation> added;
};

/// Replaces the comments the review was shown with the ones it returned.
/// Comments on files outside the review stay as they were.
[[nodiscard]] ReviewMerge merge_review(
    std::span<const Annotation> held,
    std::span<const Annotation> shown,
    std::vector<Annotation> returned);

/// Appends `annotations` to `out` as a JSON array.
void append_annotations_json(std::string& out, std::span<const Annotation> annotations);

/// The comments saved in `object` under `annotations`. A malformed list is
/// dropped whole rather than half trusted.
[[nodiscard]] std::vector<Annotation> parse_saved_annotations(simdjson::dom::object object);

} // namespace core::review
