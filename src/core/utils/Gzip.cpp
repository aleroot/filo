#include "Gzip.hpp"

#include <zlib.h>

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

namespace core::utils::gzip {
namespace {

/// zlib's windowBits for a gzip wrapper: 15-bit window + 16.
constexpr int kGzipWindowBits = 15 + 16;
constexpr int kDefaultMemoryLevel = 8;
/// z_stream sizes each call's input and output in uInt.
constexpr std::size_t kMaxChunk = std::numeric_limits<uInt>::max();
constexpr std::size_t kInitialInflateBytes = 64 * 1024;

/// Ends a z_stream with the matching deflateEnd/inflateEnd.
class StreamGuard {
public:
    StreamGuard(z_stream& stream, int (*end)(z_streamp)) noexcept : stream_(stream), end_(end) {}
    ~StreamGuard() { end_(&stream_); }
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

private:
    z_stream& stream_;
    int (*end_)(z_streamp);
};

/// An input being consumed and an output being filled by zlib. Positions are
/// kept here in size_t rather than read from z_stream's total_in/total_out:
/// those are uLong (32-bit on LLP64 platforms) and inflateReset clears them
/// between gzip members.
class Buffers {
public:
    Buffers(std::string_view input, std::size_t initial_output)
        : input_(input), output_(initial_output, '\0') {}

    [[nodiscard]] std::string_view remaining_input() const noexcept { return input_.substr(consumed_); }
    [[nodiscard]] std::size_t produced() const noexcept { return produced_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return output_.size(); }
    [[nodiscard]] bool output_full() const noexcept { return produced_ == output_.size(); }
    void grow_to(std::size_t size) { output_.resize(size); }

    /// Runs one zlib call over at most kMaxChunk bytes of input and output.
    template <typename Call>
    int step(z_stream& stream, Call call) {
        const auto avail_in = static_cast<uInt>(std::min(input_.size() - consumed_, kMaxChunk));
        const auto avail_out = static_cast<uInt>(std::min(output_.size() - produced_, kMaxChunk));
        stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(input_.data() + consumed_));
        stream.avail_in = avail_in;
        stream.next_out = reinterpret_cast<Bytef*>(output_.data() + produced_);
        stream.avail_out = avail_out;
        const int status = call(stream);
        consumed_ += avail_in - stream.avail_in;
        produced_ += avail_out - stream.avail_out;
        return status;
    }

    [[nodiscard]] std::string take() && {
        output_.resize(produced_);
        return std::move(output_);
    }

private:
    std::string_view input_;
    std::string output_;
    std::size_t consumed_ = 0;
    std::size_t produced_ = 0;
};

[[nodiscard]] std::string zlib_error(int status, const z_stream& stream) {
    if (stream.msg != nullptr) return stream.msg;
    return std::format("zlib error {}", status);
}

} // namespace

bool is_compressed(std::string_view bytes) noexcept {
    return bytes.size() >= 2 && static_cast<unsigned char>(bytes[0]) == 0x1f
        && static_cast<unsigned char>(bytes[1]) == 0x8b;
}

std::string compress(std::string_view data, CompressOptions options) {
    if (options.level < kMinLevel || options.level > kMaxLevel) {
        throw std::invalid_argument(std::format("gzip: compression level {} is outside {}-{}",
                                                options.level, kMinLevel, kMaxLevel));
    }
    z_stream stream{};
    if (deflateInit2(&stream, options.level, Z_DEFLATED, kGzipWindowBits, kDefaultMemoryLevel,
                     Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("gzip: cannot initialise the compressor");
    }
    const StreamGuard guard(stream, deflateEnd);
    // zlib keeps a pointer to the header until the stream finishes.
    gz_header header{};
    header.os = options.operating_system;
    if (deflateSetHeader(&stream, &header) != Z_OK) {
        throw std::runtime_error("gzip: cannot set the header");
    }

    const auto bound = deflateBound(&stream, static_cast<uLong>(std::min<std::size_t>(
        data.size(), std::numeric_limits<uLong>::max())));
    Buffers buffers(data, std::max<std::size_t>(bound, 64));
    for (int status = Z_OK; status != Z_STREAM_END;) {
        if (buffers.output_full()) buffers.grow_to(buffers.capacity() * 2);
        const bool last_input = buffers.remaining_input().size() <= kMaxChunk;
        status = buffers.step(stream, [last_input](z_stream& s) {
            return deflate(&s, last_input ? Z_FINISH : Z_NO_FLUSH);
        });
        // Z_BUF_ERROR only means "no room left"; the next pass grows the output.
        if (status != Z_OK && status != Z_STREAM_END && status != Z_BUF_ERROR) {
            throw std::runtime_error("gzip: " + zlib_error(status, stream));
        }
    }
    return std::move(buffers).take();
}

std::expected<std::string, std::string> decompress(std::string_view bytes,
                                                   std::size_t max_output_bytes) {
    z_stream stream{};
    if (inflateInit2(&stream, kGzipWindowBits) != Z_OK) {
        return std::unexpected("cannot initialise the decompressor");
    }
    const StreamGuard guard(stream, inflateEnd);

    // One byte past the limit tells "exactly at the limit" from "over it".
    const std::size_t capacity = max_output_bytes == std::numeric_limits<std::size_t>::max()
        ? max_output_bytes : max_output_bytes + 1;
    const auto too_large = [&] {
        return std::unexpected(std::format("decompressed data exceeds {} bytes", max_output_bytes));
    };
    const std::size_t estimate = bytes.size() > capacity / 4 ? capacity : bytes.size() * 4;
    Buffers buffers(bytes, std::min(capacity, std::max(kInitialInflateBytes, estimate)));
    for (;;) {
        if (buffers.output_full()) {
            if (buffers.capacity() == capacity) return too_large();
            buffers.grow_to(buffers.capacity() > capacity / 2 ? capacity : buffers.capacity() * 2);
        }
        const int status = buffers.step(stream, [](z_stream& s) { return inflate(&s, Z_NO_FLUSH); });
        if (status == Z_STREAM_END) {
            const auto rest = buffers.remaining_input();
            if (rest.empty()) break;
            // RFC 1952: a gzip file is a series of members (`cat a.gz b.gz`).
            if (!is_compressed(rest)) return std::unexpected("unexpected data after the gzip stream");
            if (inflateReset(&stream) != Z_OK) {
                return std::unexpected("cannot reset the decompressor");
            }
            continue;
        }
        // There is always output room here, so no progress means no input left.
        if (status == Z_BUF_ERROR) return std::unexpected("truncated gzip stream");
        if (status != Z_OK) return std::unexpected("corrupt gzip stream: " + zlib_error(status, stream));
    }
    if (buffers.produced() > max_output_bytes) return too_large();
    return std::move(buffers).take();
}

} // namespace core::utils::gzip
