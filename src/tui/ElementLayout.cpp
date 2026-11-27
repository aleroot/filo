#include "ElementLayout.hpp"

#include <algorithm>

#include <ftxui/dom/node.hpp>
#include <ftxui/screen/box.hpp>

namespace tui {
namespace {

constexpr int kMaxLayoutIterations = 20;

} // namespace

int layout_element(ftxui::Element& element, int width, int x, int y) {
    width = std::max(width, 1);
    ftxui::Node::Status status;
    element->Check(&status);

    int height = 1;
    while (status.need_iteration && status.iteration < kMaxLayoutIterations) {
        element->ComputeRequirement();
        height = std::max(element->requirement().min_y, 1);
        element->SetBox(ftxui::Box{x, x + width - 1, y, y + height - 1});
        status.need_iteration = false;
        ++status.iteration;
        element->Check(&status);
    }

    return height;
}

} // namespace tui
