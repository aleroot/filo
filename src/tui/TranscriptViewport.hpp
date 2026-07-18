#pragma once

#include "Conversation.hpp"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>

namespace tui {

class TranscriptViewportNode;

/// A bounded-memory, message-virtualized transcript renderer.
///
/// FTXUI deliberately recomputes an Element tree's requirements and boxes on
/// every frame. Reusing a large vbox therefore saves construction but not the
/// O(transcript) layout walk. TranscriptViewport instead caches only message
/// heights and one terminal-sized raster. Content/width changes remeasure the
/// affected messages; ordinary input frames copy the already-rendered viewport.
class TranscriptViewport {
public:
    using MessageSnapshot = std::shared_ptr<const std::vector<UiMessage>>;

    struct SyncResult {
        std::size_t previous_count = 0;
        std::size_t current_count = 0;
        bool content_changed = false;
    };

    TranscriptViewport();
    ~TranscriptViewport();

    TranscriptViewport(const TranscriptViewport&) = delete;
    TranscriptViewport& operator=(const TranscriptViewport&) = delete;
    TranscriptViewport(TranscriptViewport&&) noexcept;
    TranscriptViewport& operator=(TranscriptViewport&&) noexcept;

    /// Reconcile a new immutable UI snapshot. Unchanged message measurements
    /// are retained by stable message id; changed/appended cards alone become
    /// dirty. structural_key represents render options/disclosure state.
    SyncResult Sync(MessageSnapshot snapshot, std::size_t structural_key);

    ftxui::Element Render(
        std::size_t tick,
        ConversationRenderOptions options,
        std::shared_ptr<ConversationScrollAnchor> scroll_anchor,
        std::unordered_map<std::string, ftxui::Box>* disclosure_hitboxes);

    [[nodiscard]] std::size_t CacheBuildCount() const noexcept;
    [[nodiscard]] std::size_t ViewportBuildCount() const noexcept;
    [[nodiscard]] std::size_t MessageMeasureCount() const noexcept;

private:
    friend class TranscriptViewportNode;
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace tui
