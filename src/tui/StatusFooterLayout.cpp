#include "StatusFooterLayout.hpp"

#include <format>
#include <memory>
#include <utility>

#include <ftxui/dom/node.hpp>
#include <ftxui/screen/box.hpp>

namespace tui {

namespace {

using namespace ftxui;

/// The components after the root: "~/a/b" → {a, b}, "/a/b" → {a, b}.
[[nodiscard]] std::vector<std::string_view> path_components(std::string_view path) {
    if (path.starts_with('~')) {
        path.remove_prefix(1);
    }
    std::vector<std::string_view> components;
    while (!path.empty()) {
        const auto slash = path.find('/');
        const auto component = path.substr(0, slash);
        if (!component.empty()) {
            components.push_back(component);
        }
        if (slash == std::string_view::npos) {
            break;
        }
        path.remove_prefix(slash + 1);
    }
    return components;
}

class FirstThatFits final : public Node {
public:
    explicit FirstThatFits(Elements candidates) : Node(std::move(candidates)) {}

    void ComputeRequirement() override {
        for (auto& candidate : children_) {
            candidate->ComputeRequirement();
        }
        // Claim only what the most compact candidate needs: a parent that
        // sees a larger minimum would squeeze its other children for it.
        requirement_ = children_.back()->requirement();
    }

    void SetBox(Box box) override {
        Node::SetBox(box);
        const int width = box.x_max - box.x_min + 1;
        chosen_ = children_.size() - 1;
        for (std::size_t i = 0; i < children_.size(); ++i) {
            if (children_[i]->requirement().min_x <= width) {
                chosen_ = i;
                break;
            }
        }
        children_[chosen_]->SetBox(box);
    }

    void Select(Selection& selection) override {
        children_[chosen_]->Select(selection);
    }

    void Render(Screen& screen) override {
        children_[chosen_]->Render(screen);
    }

    std::string GetSelectedContent(Selection& selection) override {
        return children_[chosen_]->GetSelectedContent(selection);
    }

private:
    std::size_t chosen_ = 0;
};

} // namespace

std::size_t workspace_path_components(std::string_view path) {
    return path_components(path).size();
}

std::string elide_workspace_path(std::string_view path, std::size_t kept) {
    const auto components = path_components(path);
    if (kept == 0 || kept >= components.size()) {
        return std::string(path);
    }
    std::string elided = "…";
    for (std::size_t i = components.size() - kept; i < components.size(); ++i) {
        elided += '/';
        elided += components[i];
    }
    return elided;
}

std::string format_context_left_label(int percent, ContextLeftLabel label) {
    switch (label) {
        case ContextLeftLabel::full:
            return std::format(" {}% Context Left ", percent);
        case ContextLeftLabel::compact:
            return std::format(" {}% ctx ", percent);
        case ContextLeftLabel::percent:
            break;
    }
    return std::format(" {}% ", percent);
}

std::vector<StatusFooterDensity> status_footer_densities(
    std::size_t workspace_path_components) {
    std::vector<StatusFooterDensity> densities;
    StatusFooterDensity density;
    densities.push_back(density);
    const auto step = [&](auto&& give_up) {
        give_up(density);
        densities.push_back(density);
    };

    // Keeping all of them would only swap "~" for "…", so the first step
    // already drops a directory.
    for (std::size_t kept = workspace_path_components; kept-- > 1;) {
        step([kept](auto& d) { d.workspace_kept_components = kept; });
    }
    step([](auto& d) { d.context_label = ContextLeftLabel::compact; });
    step([](auto& d) { d.show_provider = false; });
    step([](auto& d) { d.show_usage_totals = false; });
    step([](auto& d) { d.remote_detail = RemoteFooterDetail::subject; });
    step([](auto& d) { d.show_effort = false; });
    step([](auto& d) { d.context_label = ContextLeftLabel::percent; });
    step([](auto& d) { d.remote_detail = RemoteFooterDetail::glyph; });
    step([](auto& d) { d.show_workspace = false; });
    return densities;
}

Element first_that_fits(Elements candidates) {
    if (candidates.empty()) {
        return text("");
    }
    return std::make_shared<FirstThatFits>(std::move(candidates));
}

} // namespace tui
