#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace core::review {

/// The four ways `/review` with no arguments can start. One table feeds the
/// TUI picker, the prompter console menu, and the keyboard shortcuts, so a
/// new target cannot appear in one surface and vanish from another.
enum class ReviewMenuTarget {
    Uncommitted,
    Staged,
    BaseBranch,
    Custom,
};

enum class ReviewMenuFollowUp {
    None,
    BaseBranch,
    CustomPrompt,
};

struct ReviewMenuOption {
    ReviewMenuTarget target = ReviewMenuTarget::Uncommitted;
    std::string_view label;
    std::string_view description;
    /// `/review` arguments used when @p follow_up is None. Empty means the
    /// historical uncommitted default (`/review` with no tokens).
    std::string_view request;
    ReviewMenuFollowUp follow_up = ReviewMenuFollowUp::None;
};

[[nodiscard]] constexpr std::array<ReviewMenuOption, 4> review_menu_options() noexcept {
    return {{
        {
            .target = ReviewMenuTarget::Uncommitted,
            .label = "Uncommitted changes",
            .description = "Review the current staged, unstaged, and untracked changes.",
            .request = {},
            .follow_up = ReviewMenuFollowUp::None,
        },
        {
            .target = ReviewMenuTarget::Staged,
            .label = "Staged changes",
            .description = "Review only the files that are staged for commit.",
            .request = "staged",
            .follow_up = ReviewMenuFollowUp::None,
        },
        {
            .target = ReviewMenuTarget::BaseBranch,
            .label = "Base branch",
            .description = "Review the current branch against a base branch you choose.",
            .request = {},
            .follow_up = ReviewMenuFollowUp::BaseBranch,
        },
        {
            .target = ReviewMenuTarget::Custom,
            .label = "Customised",
            .description = "Write your own review instructions for the model to follow.",
            .request = {},
            .follow_up = ReviewMenuFollowUp::CustomPrompt,
        },
    }};
}

[[nodiscard]] constexpr std::size_t review_menu_option_count() noexcept {
    return review_menu_options().size();
}

[[nodiscard]] constexpr int step_review_menu_index(int selected, int delta) noexcept {
    const int count = static_cast<int>(review_menu_option_count());
    if (count <= 0) return 0;
    const int wrapped_delta = delta % count;
    return (selected + wrapped_delta + count) % count;
}

} // namespace core::review
