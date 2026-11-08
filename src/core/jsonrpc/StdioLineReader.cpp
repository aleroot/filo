#include "StdioLineReader.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cerrno>
#include <iostream>
#include <utility>

#if !defined(_WIN32)
#include <poll.h>
#include <unistd.h>
#endif

namespace core::jsonrpc {

StdioLineReader::StdioLineReader(int fd) : fd_(fd) {}

StdioLineReader::Result StdioLineReader::read_until(
    const std::optional<std::chrono::steady_clock::time_point>& deadline)
{
    if (auto line = take_buffered_line()) {
        return {.status = Status::line, .line = std::move(*line)};
    }

#if defined(_WIN32)
    (void)deadline;
    (void)fd_;
    std::string line;
    if (!std::getline(std::cin, line)) {
        return {.status = Status::eof};
    }
    trim_cr(line);
    return {.status = Status::line, .line = std::move(line)};
#else
    while (true) {
        int timeout_ms = -1;
        if (deadline.has_value()) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= *deadline) {
                return {.status = Status::timeout};
            }
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now);
            timeout_ms = static_cast<int>(std::max<std::int64_t>(1, remaining.count()));
        }

        pollfd input{
            .fd = fd_,
            .events = POLLIN,
            .revents = 0,
        };
        const int ready = ::poll(&input, 1, timeout_ms);
        if (ready == 0) {
            return {.status = Status::timeout};
        }
        if (ready < 0) {
            if (errno == EINTR) continue;
            return {.status = Status::eof};
        }
        if ((input.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0
            && (input.revents & POLLIN) == 0) {
            return {.status = Status::eof};
        }

        char chunk[kReadChunkBytes];
        const ssize_t bytes_read = ::read(fd_, chunk, sizeof(chunk));
        if (bytes_read == 0) {
            if (buffer_offset_ < buffer_.size()) {
                std::string line = buffer_.substr(buffer_offset_);
                release_buffer();
                trim_cr(line);
                return {.status = Status::line, .line = std::move(line)};
            }
            return {.status = Status::eof};
        }
        if (bytes_read < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            return {.status = Status::eof};
        }

        buffer_.append(chunk, static_cast<std::size_t>(bytes_read));
        if (auto line = take_buffered_line()) {
            return {.status = Status::line, .line = std::move(*line)};
        }
    }
#endif
}

std::optional<std::string> StdioLineReader::take_buffered_line() {
    // Resume the newline search where the previous one stopped; rescanning the
    // whole partial line after every chunk is quadratic in its size.
    const auto newline = buffer_.find('\n', std::max(buffer_offset_, scan_offset_));
    if (newline == std::string::npos) {
        scan_offset_ = buffer_.size();
        return std::nullopt;
    }
    std::string line;
    if (buffer_offset_ == 0 && newline + 1 == buffer_.size()
        && buffer_.size() > kMaxRetainedBufferBytes) {
        // A large message alone in the buffer: hand the storage over instead
        // of copying it, which also releases it from the reader.
        line = std::move(buffer_);
        line.pop_back();
        release_buffer();
    } else {
        // Room for SIMDJSON_PADDING lets the parser read the line in place.
        const std::size_t length = newline - buffer_offset_;
        line.reserve(length + simdjson::SIMDJSON_PADDING);
        line.assign(buffer_, buffer_offset_, length);
        buffer_offset_ = newline + 1;
        compact_buffer_if_needed();
    }
    trim_cr(line);
    return line;
}

void StdioLineReader::trim_cr(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

void StdioLineReader::release_buffer() {
    if (buffer_.capacity() > kMaxRetainedBufferBytes) {
        buffer_ = std::string{};
    } else {
        buffer_.clear();
    }
    buffer_offset_ = 0;
    scan_offset_ = 0;
}

void StdioLineReader::compact_buffer_if_needed() {
    if (buffer_offset_ == 0) return;
    if (buffer_offset_ == buffer_.size()) {
        release_buffer();
        return;
    }
    if (buffer_offset_ < 4096 || buffer_offset_ * 2 < buffer_.size()) return;
    buffer_.erase(0, buffer_offset_);
    scan_offset_ = 0;
    buffer_offset_ = 0;
}

} // namespace core::jsonrpc
