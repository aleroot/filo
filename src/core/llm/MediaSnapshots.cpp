#include "MediaSnapshots.hpp"

#include "../logging/Logger.hpp"

#include <exception>

namespace core::llm {

MediaSnapshots::MediaSnapshots(std::filesystem::path directory)
    : objects_(std::move(directory)) {}

void MediaSnapshots::pin(ContentPart& part) const {
    if (!is_media_part(part.type) || part.path.empty() || !part.snapshot.empty()
        || !part.url.empty()) {
        return;
    }
    try {
        const auto stored = objects_.store(part.path, kMaxBytes);
        if (!stored) {
            core::logging::warn("[media] Cannot pin {}: {}", part.path, stored.error());
            return;
        }
        // Absent and empty files carry nothing worth pinning.
        if (stored->image.hash.empty() || stored->stamp.size == 0) return;
        part.snapshot = (objects_.directory() / stored->image.hash).string();
    } catch (const std::exception& error) {
        core::logging::warn("[media] Cannot pin {}: {}", part.path, error.what());
    }
}

bool MediaSnapshots::owns(const ContentPart& part) const {
    return part.snapshot.empty()
        || std::filesystem::path(part.snapshot).parent_path() == objects_.directory();
}

void MediaSnapshots::adopt(ContentPart& part) const {
    if (owns(part)) return;
    const std::filesystem::path source = part.snapshot;
    // Snapshots are named by the hash of their contents.
    const auto hash = source.filename().string();
    if (hash.empty() || !core::changes::is_content_hash(hash)) return;
    try {
        if (!objects_.contains(hash)) {
            const auto stored = objects_.store(source, kMaxBytes);
            if (!stored || stored->image.hash != hash) {
                core::logging::warn("[media] Cannot adopt {}: {}", part.snapshot,
                                    stored ? "its contents no longer match its name" : stored.error());
                return;
            }
        }
        part.snapshot = (objects_.directory() / hash).string();
    } catch (const std::exception& error) {
        core::logging::warn("[media] Cannot adopt {}: {}", part.snapshot, error.what());
    }
}

} // namespace core::llm
