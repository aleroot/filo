#pragma once

// ---------------------------------------------------------------------------
// gzip (RFC 1952) over in-memory buffers, backed by zlib. compress() writes
// one member; decompress() reads every member of a stream, as gunzip does,
// and rejects anything after the last one. Sizes are not limited by zlib's
// 32-bit counters.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace core::utils::gzip {

/// RFC 1952 OS field value meaning "unknown".
inline constexpr std::uint8_t kUnknownOperatingSystem = 0xff;
/// Default decompression ceiling: far above any real payload Filo reads, low
/// enough that a tiny crafted stream cannot exhaust memory.
inline constexpr std::size_t kDefaultMaxOutputBytes = std::size_t{256} << 20;
/// Compression levels: 0 stores, 9 is smallest and slowest.
inline constexpr int kMinLevel = 0;
inline constexpr int kMaxLevel = 9;

struct CompressOptions {
    int level = 6;  ///< kMinLevel-kMaxLevel.
    /// Written in the header outside the CRC. The modification time is always
    /// 0, so identical input yields identical bytes on every platform.
    std::uint8_t operating_system = kUnknownOperatingSystem;
};

/// Whether @p bytes start with the gzip magic number (1F 8B).
[[nodiscard]] bool is_compressed(std::string_view bytes) noexcept;

/// Throws std::invalid_argument for a level outside kMinLevel-kMaxLevel and
/// std::runtime_error if zlib itself fails.
[[nodiscard]] std::string compress(std::string_view data, CompressOptions options = {});

/// Inflates a gzip stream and concatenates its members. Fails on a truncated
/// or corrupt stream, on non-gzip data after the last member, and when the
/// total output would exceed @p max_output_bytes. Never throws for bad input.
[[nodiscard]] std::expected<std::string, std::string> decompress(
    std::string_view bytes, std::size_t max_output_bytes = kDefaultMaxOutputBytes);

} // namespace core::utils::gzip
