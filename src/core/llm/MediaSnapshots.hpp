#pragma once

#include "../changes/ObjectStore.hpp"
#include "Models.hpp"

#include <filesystem>

namespace core::llm {

/// Immutable copies of the local media a conversation refers to, for media
/// too large to carry inline (see ContentPart::snapshot). One directory per
/// session; copies are named by their contents, so the same file attached
/// twice is stored once, and a provider cache keyed by the copy's path (Kimi
/// video uploads) can never hand back another file's upload.
class MediaSnapshots {
public:
    /// The largest file pinned: the most any provider accepts as a video.
    static constexpr std::size_t kMaxBytes = kMaxLocalVideoBytes;

    explicit MediaSnapshots(std::filesystem::path directory);

    /// Pins the file @p part names: afterwards its bytes are read from the
    /// copy, whatever the file becomes. Leaves @p part unchanged when there
    /// is nothing to pin (no local file, already pinned, inline bytes) or the
    /// file cannot be copied; it is then read from its path as before.
    void pin(ContentPart& part) const;

    /// Whether this store holds @p part's snapshot, or it has none.
    [[nodiscard]] bool owns(const ContentPart& part) const;

    /// Makes @p part's snapshot this store's own: a conversation branched
    /// or rotated from another session must not lose its media when that
    /// session is deleted. Stored once per content; a snapshot that can no
    /// longer be read is left as it is (and is sent as a note).
    void adopt(ContentPart& part) const;

    [[nodiscard]] const std::filesystem::path& directory() const noexcept {
        return objects_.directory();
    }

private:
    core::changes::ObjectStore objects_;
};

} // namespace core::llm
