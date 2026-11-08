#pragma once

#include <chrono>
#include <optional>
#include <string>

namespace core::jsonrpc {

/// Newline-delimited reader for a byte stream.
///
/// MCP stdio and the Agent Client Protocol both frame JSON-RPC as one message
/// per line. The reader keeps a bounded buffer, resumes the newline scan where
/// it stopped, and hands a message that fills the buffer over without copying
/// it. `fd` is used on POSIX; Windows reads `std::cin` and ignores it.
class StdioLineReader {
public:
    enum class Status {
        line,
        timeout,
        eof,
    };

    struct Result {
        Status status = Status::eof;
        std::string line{};
    };

    explicit StdioLineReader(int fd = 0);

    [[nodiscard]] Result read_until(
        const std::optional<std::chrono::steady_clock::time_point>& deadline);

private:
    static constexpr std::size_t kReadChunkBytes = 64 * 1024;
    /// One large request must not pin its buffer for the rest of the session.
    static constexpr std::size_t kMaxRetainedBufferBytes = 256 * 1024;

    [[nodiscard]] std::optional<std::string> take_buffered_line();
    static void trim_cr(std::string& line);
    void release_buffer();
    void compact_buffer_if_needed();

    int fd_ = 0;
    std::string buffer_;
    std::size_t buffer_offset_ = 0;
    /// Everything in [buffer_offset_, scan_offset_) is known to hold no newline.
    std::size_t scan_offset_ = 0;
};

} // namespace core::jsonrpc
