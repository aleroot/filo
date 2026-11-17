#pragma once

#include "SessionStore.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace core::session {

/// Append-only observable execution history, independent of resume snapshots.
/// One JSON object per line; each append is serialized across processes. It
/// is not synced: the journal is an audit record, not crash-recovery state.
/// An interrupted final append is discarded on the next write, never earlier events.
///
/// The journal last appended to stays open between appends. Closing the last
/// writable descriptor of a file makes macOS write its dirty pages out at
/// once, so an open-append-close cycle rewrote a whole page per record
/// (measured: 5.5x the journal's own size); held open, the page cache merges
/// consecutive records into one write. Copies share the open journal.
class ExecutionJournal final {
public:
    static constexpr int kVersion = 1;
    explicit ExecutionJournal(std::filesystem::path root = default_root());
    [[nodiscard]] static std::filesystem::path default_root();
    /// Where the journals of the sessions saved in @p sessions_dir live.
    [[nodiscard]] static std::filesystem::path root_in(const std::filesystem::path& sessions_dir);
    [[nodiscard]] std::filesystem::path path_for(std::string_view session_id) const;

    bool remove(std::string_view session_id, std::string* error = nullptr) const;

    /// payload is a JSON object produced by Filo's serializers, not model text.
    bool append(std::string_view session_id, std::string_view actor,
                std::string_view parent_tool_call_id, std::string_view event,
                std::string_view payload, std::string* error = nullptr) const;
private:
    struct Open;
    std::filesystem::path root_;
    std::shared_ptr<Open> open_;
};

} // namespace core::session
