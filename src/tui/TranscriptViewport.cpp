#include "TranscriptViewport.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include <ftxui/dom/node.hpp>
#include <ftxui/dom/selection.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/util/autoreset.hpp>

namespace tui {
namespace {

constexpr int kScrollbarColumns = 1;
constexpr int kMaxLayoutIterations = 20;

std::size_t combine_hash(std::size_t seed, std::size_t value) {
    constexpr std::size_t kMix = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    seed ^= value + kMix + (seed << 6) + (seed >> 2);
    return seed;
}

std::size_t message_fingerprint(const UiMessage& msg) {
    std::size_t seed = 0;
    const auto add_value = [&seed](std::size_t value) {
        seed = combine_hash(seed, value);
    };
    const auto add_text = [&add_value](std::string_view value) {
        add_value(std::hash<std::string_view>{}(value));
    };
    const auto add_optional_int = [&add_value](const std::optional<int>& value) {
        add_value(value.has_value() ? 1U : 0U);
        if (value.has_value()) {
            add_value(static_cast<std::size_t>(*value));
        }
    };

    add_text(msg.id);
    add_value(static_cast<std::size_t>(msg.type));
    add_text(msg.text);
    add_text(msg.secondary_text);
    add_text(msg.disclosure_text);
    add_text(msg.icon);
    add_value(msg.repeat_count);
    add_value(msg.custom_color.has_value() ? 1U : 0U);
    if (msg.custom_color.has_value()) {
        add_text(msg.custom_color->Print(false));
    }
    add_value(static_cast<std::size_t>(msg.margin_top));
    add_value(static_cast<std::size_t>(msg.margin_bottom));
    add_value(static_cast<std::size_t>(msg.pending));
    add_value(static_cast<std::size_t>(msg.finalized));
    add_value(static_cast<std::size_t>(msg.thinking));
    add_value(static_cast<std::size_t>(msg.show_activity_status));
    add_value(static_cast<std::size_t>(msg.stopped));
    add_text(msg.activity_elapsed);
    // Activity disclosure state — streaming reasoning grows text, and the phase
    // kind/label/finished-trace all change the rendered box.
    add_value(static_cast<std::size_t>(msg.reasoning_kind));
    add_text(msg.reasoning_text);
    add_value(static_cast<std::size_t>(msg.reasoning_active));
    add_text(msg.reasoning_elapsed);
    add_value(static_cast<std::size_t>(msg.activity_recorded));
    add_text(msg.timestamp);
    add_value(static_cast<std::size_t>(msg.tool_group_border_top));
    add_value(static_cast<std::size_t>(msg.tool_group_border_bottom));
    add_value(msg.tools.size());

    for (const auto& tool : msg.tools) {
        add_text(tool.id);
        add_text(tool.name);
        add_text(tool.args);
        add_text(tool.description);
        add_text(tool.result.summary);
        add_optional_int(tool.result.exit_code);
        add_value(static_cast<std::size_t>(tool.result.truncated));
        add_value(static_cast<std::size_t>(tool.auto_approved));
        add_value(static_cast<std::size_t>(tool.status));
        add_optional_int(tool.progress);
        add_optional_int(tool.progress_total);
        add_text(tool.progress_message);

        add_text(tool.diff_preview.title);
        add_value(tool.diff_preview.lines.size());
        for (const auto& line : tool.diff_preview.lines) {
            add_value(static_cast<std::size_t>(line.kind));
            add_text(line.content);
        }
        add_value(tool.diff_preview.hidden_line_count);

        add_value(tool.subagents.size());
        for (const auto& subagent : tool.subagents) {
            add_text(subagent.id);
            add_text(subagent.worker_name);
            add_text(subagent.description);
            add_text(subagent.provider);
            add_text(subagent.model);
            add_text(subagent.latest_text);
            add_text(subagent.summary);
            add_value(static_cast<std::size_t>(subagent.steps));
            add_value(static_cast<std::size_t>(subagent.tool_calls));
            add_value(static_cast<std::size_t>(subagent.failed_tool_calls));
            add_value(static_cast<std::size_t>(subagent.status));

            add_value(subagent.recent_tools.size());
            for (const auto& child_tool : subagent.recent_tools) {
                add_text(child_tool.id);
                add_text(child_tool.name);
                add_text(child_tool.args);
                add_text(child_tool.description);
                add_value(static_cast<std::size_t>(child_tool.status));
            }
        }
    }
    return seed;
}

/// Lay out an element at an exact content width without allocating a full
/// transcript Screen. This mirrors ftxui::Render's convergence loop, but fixes
/// the x-axis and lets the element discover its wrapped natural height.
int layout_at(ftxui::Element& element, int width, int x, int y) {
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

} // namespace

struct TranscriptViewport::State {
    struct Entry {
        std::string id;
        std::size_t fingerprint = 0;
        int height = 1;
        int offset = 0;
        bool measured = false;
    };

    struct Frame {
        int width = 0;
        int height = 0;
        int scroll_top = 0;
        std::size_t generation = 0;
        std::size_t animation_tick = 0;
        bool animated = false;
        std::optional<ftxui::Screen> raster;
        std::unordered_map<std::string, ftxui::Box> hitboxes;
        std::vector<std::size_t> visible_entries;
    };

    MessageSnapshot snapshot = std::make_shared<const std::vector<UiMessage>>();
    std::vector<Entry> entries;
    std::size_t structural_key = 0;
    bool has_snapshot = false;
    int measured_width = 0;
    int total_height = 1;
    std::size_t generation = 0;
    std::size_t cache_build_count = 0;
    std::size_t viewport_build_count = 0;
    std::size_t message_measure_count = 0;
    bool measurements_dirty = true;
    Frame frame;

    void invalidate_frame() {
        frame.raster.reset();
    }

    void invalidate_measurements() {
        for (auto& entry : entries) {
            entry.measured = false;
        }
        measured_width = 0;
        measurements_dirty = true;
        invalidate_frame();
    }

    bool ensure_measurements(int width,
                             std::size_t tick,
                             ConversationRenderOptions options) {
        width = std::max(width, 1);
        const int previous_total = total_height;
        if (measured_width != width) {
            for (auto& entry : entries) {
                entry.measured = false;
            }
            measured_width = width;
            measurements_dirty = true;
        }
        if (!measurements_dirty) {
            return false;
        }

        options.system_disclosure_hitboxes = nullptr;
        int offset = 0;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            auto& entry = entries[i];
            if (!entry.measured) {
                auto element = render_history_message((*snapshot)[i], tick, options);
                entry.height = layout_at(element, width, 0, 0);
                entry.measured = true;
                ++message_measure_count;
            }
            entry.offset = offset;
            offset += entry.height;
        }
        total_height = std::max(offset, 1);
        measurements_dirty = false;
        return total_height != previous_total;
    }

    std::vector<std::size_t> visible_entries(int scroll_top, int viewport_height) const {
        std::vector<std::size_t> result;
        const int scroll_bottom = scroll_top + std::max(viewport_height, 1);
        const auto first = std::ranges::upper_bound(
            entries,
            scroll_top,
            {},
            [](const Entry& entry) { return entry.offset + entry.height; });
        result.reserve(static_cast<std::size_t>(std::max(viewport_height, 1)));
        for (auto it = first; it != entries.end() && it->offset < scroll_bottom; ++it) {
            const std::size_t i = static_cast<std::size_t>(it - entries.begin());
            result.push_back(i);
        }
        return result;
    }

    bool prepare(int width,
                 int viewport_height,
                 std::size_t tick,
                 ConversationRenderOptions options,
                 const std::shared_ptr<ConversationScrollAnchor>& anchor) {
        const bool height_changed = ensure_measurements(width, tick, options);
        anchor->content_height = total_height;
        if (anchor->follow_bottom) {
            anchor->focus_y = total_height;
        } else {
            anchor->focus_y = std::clamp(anchor->focus_y, 0, total_height);
        }

        viewport_height = std::max(viewport_height, 1);
        const int max_scroll = std::max(total_height - viewport_height, 0);
        const int scroll_top = std::clamp(
            anchor->focus_y - (viewport_height - 1) / 2,
            0,
            max_scroll);
        const bool geometry_reusable = frame.raster.has_value()
            && frame.width == width
            && frame.height == viewport_height
            && frame.scroll_top == scroll_top
            && frame.generation == generation;
        if (geometry_reusable
            && (!frame.animated || frame.animation_tick == tick)) {
            return height_changed;
        }

        auto visible = visible_entries(scroll_top, viewport_height);
        const bool animated = std::ranges::any_of(visible, [&](std::size_t index) {
            return message_uses_animation((*snapshot)[index], options.show_spinner);
        });
        const std::size_t animation_key = animated ? tick : 0;

        frame.visible_entries = std::move(visible);
        frame.scroll_top = scroll_top;
        frame.width = width;
        frame.height = viewport_height;
        frame.generation = generation;
        frame.animation_tick = animation_key;
        frame.animated = animated;
        frame.raster.emplace(width, viewport_height);
        frame.raster->Clear();
        frame.hitboxes.clear();

        options.system_disclosure_hitboxes = &frame.hitboxes;
        for (const std::size_t index : frame.visible_entries) {
            auto element = render_history_message((*snapshot)[index], tick, options);
            const auto& entry = entries[index];
            const int local_y = entry.offset - scroll_top;
            layout_at(element, width, 0, local_y);
            element->Render(*frame.raster);
        }
        ++viewport_build_count;
        return height_changed;
    }
};

class TranscriptViewportNode final : public ftxui::Node {
public:
    TranscriptViewportNode(
        std::shared_ptr<TranscriptViewport::State> state,
        std::size_t tick,
        ConversationRenderOptions options,
        std::shared_ptr<ConversationScrollAnchor> anchor,
        std::unordered_map<std::string, ftxui::Box>* disclosure_hitboxes)
        : state_(std::move(state)),
          tick_(tick),
          options_(std::move(options)),
          anchor_(std::move(anchor)),
          disclosure_hitboxes_(disclosure_hitboxes) {}

    void ComputeRequirement() override {
        requirement_ = {};
        requirement_.min_x = kScrollbarColumns + 1;
        requirement_.min_y = std::max(state_->total_height, 1);
        requirement_.flex_grow_x = 1;
        requirement_.flex_shrink_x = 1;
        requirement_.flex_grow_y = 1;
        requirement_.flex_shrink_y = 1;
    }

    void SetBox(ftxui::Box box) override {
        Node::SetBox(box);
        const int viewport_width = std::max(box.x_max - box.x_min + 1, 1);
        content_width_ = std::max(viewport_width - kScrollbarColumns, 1);
        viewport_height_ = std::max(box.y_max - box.y_min + 1, 1);
        needs_layout_iteration_ |= state_->prepare(
            content_width_, viewport_height_, tick_, options_, anchor_);
        publish_hitboxes();
    }

    void Check(Status* status) override {
        if (status->iteration == 0) {
            status->need_iteration = true;
        }
        if (needs_layout_iteration_) {
            status->need_iteration = true;
            needs_layout_iteration_ = false;
        }
    }

    void Select(ftxui::Selection& selection) override {
        if (ftxui::Box::Intersection(selection.GetBox(), box_).IsEmpty()) {
            return;
        }

        // Selection is rare and semantic fidelity matters more than caching.
        // Rebuild only the visible cards, let their native Select() methods add
        // exact text parts, then render those selected nodes in Render().
        selection_elements_.clear();
        auto selection_options = options_;
        selection_options.system_disclosure_hitboxes = disclosure_hitboxes_;
        for (const std::size_t index : state_->frame.visible_entries) {
            auto element = render_history_message(
                (*state_->snapshot)[index], tick_, selection_options);
            const auto& entry = state_->entries[index];
            const int y = box_.y_min + entry.offset - state_->frame.scroll_top;
            layout_at(element, content_width_, box_.x_min, y);
            element->Select(selection);
            selection_elements_.push_back(std::move(element));
        }
    }

    void Render(ftxui::Screen& screen) override {
        if (!selection_elements_.empty()) {
            // Selection needs the native message nodes so FTXUI can collect
            // their text and apply per-cell highlighting. Those nodes are laid
            // out in transcript coordinates, which means the first/last visible
            // card can extend beyond this viewport. Clip the direct render just
            // like FTXUI's frame decorator does; otherwise an active selection
            // can paint the offscreen portion over siblings such as the startup
            // banner, prompt, and status bar.
            const ftxui::AutoReset<ftxui::Box> stencil(
                &screen.stencil,
                ftxui::Box::Intersection(box_, screen.stencil));
            for (const auto& element : selection_elements_) {
                element->Render(screen);
            }
        } else if (state_->frame.raster.has_value()) {
            blit(screen, *state_->frame.raster);
        }
        render_scrollbar(screen);
    }

private:
    void publish_hitboxes() {
        if (disclosure_hitboxes_ == nullptr) {
            return;
        }
        disclosure_hitboxes_->clear();
        for (const auto& [id, local] : state_->frame.hitboxes) {
            auto translated = local;
            translated.Shift(box_.x_min, box_.y_min);
            (*disclosure_hitboxes_)[id] = translated;
        }
    }

    void blit(ftxui::Screen& target, const ftxui::Screen& source) const {
        const int rows = std::min(viewport_height_, source.dimy());
        const int columns = std::min(content_width_, source.dimx());
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < columns; ++x) {
                const auto& source_cell = source.CellAt(x, y);
                auto& target_cell = target.CellAt(box_.x_min + x, box_.y_min + y);

                // A cached viewport is a transparent child surface. Preserve
                // styles already applied by outer FTXUI decorators, exactly as
                // direct child rendering would, and layer the cached card style
                // over them instead of replacing the whole Cell.
                target_cell.character = source_cell.character;
                target_cell.bold |= source_cell.bold;
                target_cell.dim |= source_cell.dim;
                target_cell.italic |= source_cell.italic;
                target_cell.inverted ^= source_cell.inverted;
                target_cell.underlined |= source_cell.underlined;
                target_cell.underlined_double |= source_cell.underlined_double;
                target_cell.strikethrough |= source_cell.strikethrough;
                target_cell.automerge |= source_cell.automerge;

                if (source_cell.foreground_color != ftxui::Color::Default) {
                    target_cell.foreground_color = source_cell.foreground_color.IsOpaque()
                        ? source_cell.foreground_color
                        : ftxui::Color::Blend(
                            target_cell.foreground_color,
                            source_cell.foreground_color);
                }
                if (source_cell.background_color != ftxui::Color::Default) {
                    target_cell.background_color = source_cell.background_color.IsOpaque()
                        ? source_cell.background_color
                        : ftxui::Color::Blend(
                            target_cell.background_color,
                            source_cell.background_color);
                }
                if (source_cell.hyperlink != 0) {
                    target_cell.hyperlink = target.RegisterHyperlink(
                        source.Hyperlink(source_cell.hyperlink));
                }
            }
        }
    }

    void render_scrollbar(ftxui::Screen& screen) const {
        const int total = state_->total_height;
        if (total <= viewport_height_ || box_.x_max < box_.x_min) {
            return;
        }

        const int x = box_.x_max;
        const int doubled_size = std::max(
            2 * viewport_height_ * viewport_height_ / std::max(total, 1),
            1);
        const int doubled_start =
            2 * state_->frame.scroll_top * viewport_height_ / std::max(total, 1);
        for (int row = 0; row < viewport_height_; ++row) {
            const int upper = 2 * row;
            const int lower = upper + 1;
            const bool up = doubled_start <= upper
                && upper <= doubled_start + doubled_size;
            const bool down = doubled_start <= lower
                && lower <= doubled_start + doubled_size;
            screen.CellAt(x, box_.y_min + row).character =
                up ? (down ? "┃" : "╹") : (down ? "╻" : " ");
        }
    }

    std::shared_ptr<TranscriptViewport::State> state_;
    std::size_t tick_;
    ConversationRenderOptions options_;
    std::shared_ptr<ConversationScrollAnchor> anchor_;
    std::unordered_map<std::string, ftxui::Box>* disclosure_hitboxes_;
    int content_width_ = 1;
    int viewport_height_ = 1;
    bool needs_layout_iteration_ = false;
    std::vector<ftxui::Element> selection_elements_;
};

TranscriptViewport::TranscriptViewport()
    : state_(std::make_shared<State>()) {}

TranscriptViewport::~TranscriptViewport() = default;
TranscriptViewport::TranscriptViewport(TranscriptViewport&&) noexcept = default;
TranscriptViewport& TranscriptViewport::operator=(TranscriptViewport&&) noexcept = default;

TranscriptViewport::SyncResult TranscriptViewport::Sync(
    MessageSnapshot snapshot,
    std::size_t structural_key) {
    static const auto empty = std::make_shared<const std::vector<UiMessage>>();
    if (!snapshot) {
        snapshot = empty;
    }

    const std::size_t previous_count = state_->entries.size();
    if (snapshot.get() == state_->snapshot.get()
        && structural_key == state_->structural_key
        && state_->has_snapshot) {
        return {previous_count, previous_count, false};
    }

    std::vector<std::size_t> fingerprints;
    fingerprints.reserve(snapshot->size());
    for (const auto& message : *snapshot) {
        const auto fingerprint = message_fingerprint(message);
        fingerprints.push_back(fingerprint);
    }

    bool exact_content_match = state_->has_snapshot
        && state_->entries.size() == snapshot->size();
    for (std::size_t i = 0; exact_content_match && i < snapshot->size(); ++i) {
        exact_content_match = state_->entries[i].id == (*snapshot)[i].id
            && state_->entries[i].fingerprint == fingerprints[i];
    }
    const bool content_changed = state_->has_snapshot && !exact_content_match;
    const bool structure_changed = !state_->has_snapshot
        || structural_key != state_->structural_key;
    const bool visual_changed = !state_->has_snapshot || content_changed || structure_changed;

    if (visual_changed) {
        std::unordered_map<std::string, State::Entry> reusable;
        if (!structure_changed) {
            reusable.reserve(state_->entries.size());
            for (auto& entry : state_->entries) {
                reusable.emplace(entry.id, std::move(entry));
            }
        }

        std::vector<State::Entry> next;
        next.reserve(snapshot->size());
        for (std::size_t i = 0; i < snapshot->size(); ++i) {
            const auto& message = (*snapshot)[i];
            auto found = reusable.find(message.id);
            if (found != reusable.end()
                && found->second.fingerprint == fingerprints[i]) {
                next.push_back(std::move(found->second));
            } else {
                next.push_back(State::Entry{
                    .id = message.id,
                    .fingerprint = fingerprints[i],
                });
            }
        }
        state_->entries = std::move(next);
        if (structure_changed) {
            state_->invalidate_measurements();
        } else {
            state_->measurements_dirty = true;
            state_->invalidate_frame();
        }
        ++state_->generation;
        ++state_->cache_build_count;
    }

    state_->snapshot = std::move(snapshot);
    state_->structural_key = structural_key;
    state_->has_snapshot = true;
    return {previous_count, state_->entries.size(), content_changed};
}

ftxui::Element TranscriptViewport::Render(
    std::size_t tick,
    ConversationRenderOptions options,
    std::shared_ptr<ConversationScrollAnchor> scroll_anchor,
    std::unordered_map<std::string, ftxui::Box>* disclosure_hitboxes) {
    return std::make_shared<TranscriptViewportNode>(
        state_, tick, std::move(options), std::move(scroll_anchor), disclosure_hitboxes);
}

std::size_t TranscriptViewport::CacheBuildCount() const noexcept {
    return state_->cache_build_count;
}

std::size_t TranscriptViewport::ViewportBuildCount() const noexcept {
    return state_->viewport_build_count;
}

std::size_t TranscriptViewport::MessageMeasureCount() const noexcept {
    return state_->message_measure_count;
}

} // namespace tui
