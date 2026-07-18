#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>

namespace core::agent {

struct StoredToolResult {
    std::string reference;
    std::string digest;
    std::size_t original_bytes = 0;
};

struct ToolResultChunk {
    std::string content;
    std::uint64_t offset = 0;
    std::uint64_t next_offset = 0;
    std::uint64_t total_bytes = 0;
    bool complete = false;
};

class ToolResultStore final {
public:
    static constexpr std::size_t kDefaultReadChars = 4096;
    static constexpr std::size_t kMaxReadChars = 4096;

    explicit ToolResultStore(std::filesystem::path root = default_root());

    [[nodiscard]] std::expected<StoredToolResult, std::string> store(
        std::string_view session_id,
        std::string_view tool_call_id,
        std::string_view content) const;

    [[nodiscard]] std::expected<ToolResultChunk, std::string> read(
        std::string_view session_id,
        std::string_view reference,
        std::uint64_t offset,
        std::size_t limit = kDefaultReadChars) const;

    [[nodiscard]] static std::filesystem::path default_root();

    [[nodiscard]] static std::string attach_reference(
        std::string compact_payload,
        const StoredToolResult& stored);

private:
    [[nodiscard]] static std::string sanitize_component(
        std::string_view value,
        std::string_view fallback);
    [[nodiscard]] static bool valid_reference(std::string_view reference) noexcept;

    std::filesystem::path root_;
};

} // namespace core::agent
