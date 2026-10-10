#pragma once

#include "SessionStore.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace core::session {

inline constexpr std::size_t kDefaultTranscriptPageBytes = 4096;
inline constexpr std::size_t kMaxTranscriptPageBytes = 8192;

struct TranscriptReadOptions {
    std::uint64_t offset = 0;
    std::size_t limit = kDefaultTranscriptPageBytes;
    std::string_view revision = {};
};

struct SessionTranscriptPage {
    std::string session_id;
    std::string revision;
    std::string content;
    std::uint64_t offset = 0;
    std::uint64_t next_offset = 0;
    std::uint64_t total_bytes = 0;
    bool complete = false;
};

/// Provider-neutral saved transcript paging. The caller adapts requests and
/// responses to its transport; this reader has no JSON or tool dependencies.
class SessionTranscriptReader final {
public:
    explicit SessionTranscriptReader(SessionStore store);

    [[nodiscard]] std::expected<SessionTranscriptPage, std::string> read(
        std::string_view session_id,
        std::string_view current_session_id,
        const TranscriptReadOptions& options = {}) const;

private:
    SessionStore store_;
};

} // namespace core::session
