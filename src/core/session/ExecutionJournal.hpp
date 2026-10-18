#pragma once

#include "SessionStore.hpp"
#include <filesystem>
#include <string>
#include <string_view>

namespace core::session {

/// Append-only observable execution history, independent of resume snapshots.
/// One JSON object per line; each fsynced append is serialized across processes.
/// An interrupted final append is discarded on the next write, never earlier events.
class ExecutionJournal final {
public:
    static constexpr int kVersion = 1;
    explicit ExecutionJournal(std::filesystem::path root = default_root());
    [[nodiscard]] static std::filesystem::path default_root();
    [[nodiscard]] std::filesystem::path path_for(std::string_view session_id) const;

    bool remove(std::string_view session_id, std::string* error = nullptr) const;

    /// payload is a JSON object produced by Filo's serializers, not model text.
    bool append(std::string_view session_id, std::string_view actor,
                std::string_view parent_tool_call_id, std::string_view event,
                std::string_view payload, std::string* error = nullptr) const;
private:
    std::filesystem::path root_;
};

} // namespace core::session
