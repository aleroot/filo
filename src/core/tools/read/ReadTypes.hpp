#pragma once

#include "../Tool.hpp"
#include <cstddef>
#include <expected>
#include <filesystem>
#include <istream>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::read {
inline constexpr std::size_t kMaxSourceBytes = 2 * 1024 * 1024;
inline constexpr std::size_t kMaxOutputChars = 8192;
inline constexpr std::size_t kMaxSliceChars = 512 * 1024;
// Remote bytes are bounded by what `fetch_url` may already return, so a line
// slice cannot become a wider door into context than the fetch tool itself.
// Local and stored sources keep the larger ceiling: an exact recovery of a
// file Filo already read must never be silently shortened.
inline constexpr std::size_t kMaxRemoteSliceChars = 64 * 1024;
inline constexpr std::size_t kMaxResources = 8;
// Subagent profile key that configures the evidence worker. The orchestrator
// reserves the same name, so both sides agree on what `reader` means.
inline constexpr std::string_view kReaderProfile = "reader";

struct Options {
    std::vector<std::string> paths;
    std::string view = "exact";
    std::string question;
    std::string cell;
    std::string expected_digest;
    int offset_line = 1;
    int limit_lines = 0;
    bool sliced = false;
};
struct Section {
    std::size_t first_line = 1;
    std::size_t last_line = 1;
    std::string locator;
};
struct Resource {
    std::string uri;
    std::string kind = "text";
    std::string text;
    std::string digest;
    bool truncated = false;
    std::vector<Section> sections;
};

[[nodiscard]] std::string read_text_file(const Options& options, const core::context::SessionContext& context);
[[nodiscard]] std::string read_prefix(std::istream& stream, std::size_t bytes);
[[nodiscard]] std::expected<Options, std::string> parse_options(std::string_view json);
[[nodiscard]] std::string digest(std::string_view text);
[[nodiscard]] std::string bounded_prefix(std::string_view text, std::size_t bytes);
[[nodiscard]] std::vector<std::string_view> lines(std::string_view text);
/// Same count `lines()` would produce, without materializing the split.
[[nodiscard]] std::size_t line_count(std::string_view text);
[[nodiscard]] std::string slice(const Resource& source, int first, int count);
[[nodiscard]] std::string outline(const Resource& source, std::size_t budget);
[[nodiscard]] bool is_instruction_resource(std::string_view uri);
[[nodiscard]] std::expected<Resource, std::string> decode_notebook(Resource resource, const Options& options);
} // namespace core::tools::read
