#include "AnchoredPopover.hpp"

#include "TuiTheme.hpp"

#include <ftxui/dom/node.hpp>
#include <ftxui/dom/requirement.hpp>
#include <ftxui/screen/screen.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>

namespace tui {
namespace {

using namespace ftxui;

[[nodiscard]] bool is_empty(const Box& box) {
    return box.x_max < box.x_min || box.y_max < box.y_min;
}

/// Lays its child out at `place_popover_below(*anchor, own box)` and draws it
/// on top of whatever shares the box. It asks for no space and exports no
/// focus, so wrapping it in a dbox never changes the base layout.
class PopoverBelow final : public Node {
public:
    PopoverBelow(Element popover, const Box* anchor, PopoverPlacement placement, Box* popover_box)
        : Node({std::move(popover)}),
          anchor_(anchor),
          placement_(placement),
          popover_box_(popover_box) {}

    void ComputeRequirement() override {
        children_[0]->ComputeRequirement();
        requirement_ = Requirement{};
    }

    void SetBox(Box box) override {
        Node::SetBox(box);
        placed_ = place_popover_below(anchor_ ? *anchor_ : kEmptyBox,
                                      box,
                                      children_[0]->requirement().min_y,
                                      placement_);
        if (popover_box_) *popover_box_ = placed_;
        children_[0]->SetBox(placed_);
    }

    void Render(Screen& screen) override {
        if (!is_empty(placed_)) children_[0]->Render(screen);
    }

private:
    const Box* anchor_;
    PopoverPlacement placement_;
    Box* popover_box_;
    Box placed_ = kEmptyBox;
};

} // namespace

Element render_popover_trigger(std::string_view label, const PopoverTrigger& trigger) {
    auto element = trigger.open
        ? hbox({text(std::string(label)), text(" ▴")}) | color(ColorYellowBright) | bold
        : hbox({text(std::string(label)), text(" ▾") | color(ColorTextMuted)});
    if (trigger.hitbox) element = std::move(element) | reflect(*trigger.hitbox);
    return element;
}

Box place_popover_below(const Box& anchor,
                        const Box& bounds,
                        int content_height,
                        const PopoverPlacement& placement) {
    const int margin = std::max(0, placement.margin);
    const int min_left = bounds.x_min + margin;
    const int right = bounds.x_max - margin;
    const bool anchored = !is_empty(anchor);

    const bool align_end = placement.align == PopoverAlign::end;

    const int available_width = std::max(1, right - min_left + 1);
    int width = std::clamp(placement.preferred_width, 1, available_width);
    if (anchored) {
        // Room between the aligned anchor edge and the far side of the bounds.
        const int width_from_anchor = align_end
            ? std::min(anchor.x_max, right) - min_left + 1
            : right - std::max(anchor.x_min, min_left) + 1;
        const int narrowest = std::min(width, std::max(1, placement.min_width));
        width = std::clamp(width_from_anchor, narrowest, width);
    }
    const int max_left = std::max(min_left, right - width + 1);

    const int aligned_left = align_end ? anchor.x_max - width + 1 : anchor.x_min;
    const int left = anchored ? std::clamp(aligned_left, min_left, max_left)
                              : min_left + (max_left - min_left) / 2;
    const int top = anchored ? anchor.y_max + 1 : bounds.y_min + margin;
    const int room = std::max(0, bounds.y_max - margin - top + 1);
    const int height = std::clamp(content_height, 0, room);
    return Box{left, left + width - 1, top, top + height - 1};
}

Element overlay_popover_below(Element base,
                              Element popover,
                              const Box* anchor,
                              PopoverPlacement placement,
                              Box* popover_box) {
    return dbox({
        std::move(base),
        std::make_shared<PopoverBelow>(std::move(popover) | clear_under,
                                       anchor, placement, popover_box),
    });
}

} // namespace tui
