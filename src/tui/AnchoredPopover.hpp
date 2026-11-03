#pragma once

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>
#include <string_view>

namespace tui {

/// The box of something not on screen: contains no cell.
inline constexpr ftxui::Box kEmptyBox{0, -1, 0, -1};

/// A clickable label that opens a popover: where it was drawn (filled in
/// during layout, like `reflect`) and whether its popover is open.
struct PopoverTrigger {
    ftxui::Box* hitbox = nullptr;
    bool open = false;
};

/// Draws a trigger label with a disclosure chevron (▾ closed, ▴ open), inverted
/// while open, and reports its box through `trigger.hitbox`.
[[nodiscard]] ftxui::Element render_popover_trigger(std::string_view label,
                                                    const PopoverTrigger& trigger);

/// Which edge of the popover lines up with the same edge of its anchor: `start`
/// for triggers on the left of the screen, `end` for triggers on the right,
/// so the popover always grows toward the open side.
enum class PopoverAlign { start, end };

struct PopoverPlacement {
    int preferred_width = 96;
    /// The popover narrows down to this width to stay under its anchor before
    /// it slides left.
    int min_width = 72;
    /// Cells kept free between the popover and the edges of the bounds.
    int margin = 1;
    PopoverAlign align = PopoverAlign::start;
};

/// Pure geometry: the box a popover of `content_height` rows occupies when it
/// drops down from `anchor` inside `bounds`. The edge chosen by `align` lines
/// up with the anchor's: the popover first narrows (not below `min_width`),
/// then slides only as far as needed to stay inside the bounds; its height is
/// cut to the space left below the anchor. An empty anchor (the trigger is not
/// on screen) centres the popover at the top of the bounds.
[[nodiscard]] ftxui::Box place_popover_below(const ftxui::Box& anchor,
                                             const ftxui::Box& bounds,
                                             int content_height,
                                             const PopoverPlacement& placement = {});

/// Overlays `popover` on `base`, dropping down from `*anchor`. The anchor is
/// read during layout, after `base` has been laid out, so a trigger reflected
/// inside `base` positions the popover in the same frame — no stale box after
/// a resize. The popover clears the cells beneath it and takes no layout space.
/// `popover_box`, when given, receives where the popover was drawn.
[[nodiscard]] ftxui::Element overlay_popover_below(ftxui::Element base,
                                                   ftxui::Element popover,
                                                   const ftxui::Box* anchor,
                                                   PopoverPlacement placement = {},
                                                   ftxui::Box* popover_box = nullptr);

} // namespace tui
