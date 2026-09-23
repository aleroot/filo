#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>

namespace core::tools::detail {

// Simple glob matcher — supports * (any sequence) and ? (any one char).
// Path separators are treated as regular characters (there is no special '**' handling).
// Brace expansion is intentionally unsupported.
inline bool glob_match(std::string_view pattern, std::string_view name) {
    if (pattern.empty()) return name.empty();
    if (pattern[0] == '*') {
        // Collapse consecutive stars.
        while (pattern.size() > 1 && pattern[1] == '*') pattern.remove_prefix(1);
        pattern.remove_prefix(1);
        // Try anchoring the remainder at every position in name.
        for (size_t i = 0; i <= name.size(); ++i)
            if (glob_match(pattern, name.substr(i))) return true;
        return false;
    }
    if (name.empty()) return false;
    if (pattern[0] == '?' || pattern[0] == name[0])
        return glob_match(pattern.substr(1), name.substr(1));
    return false;
}

// Directories that are never useful to search or descend into.
inline bool should_skip_dir(const std::filesystem::path& p) {
    const auto name = p.filename().string();
    for (const char* d : {
        ".git", "node_modules", "build", ".build", "dist", "out",
        ".cache", "DerivedData", "__pycache__", ".gradle", ".idea",
        "vendor", "target"
    }) {
        if (name == d) return true;
    }
    return false;
}

// Reads a whole file in one allocation sized from the file itself. Streaming
// into an ostringstream and copying out with str() instead holds about three
// times the file at once. Bytes appended after the size was taken are still
// read. nullopt only when the file cannot be opened.
[[nodiscard]] inline std::optional<std::string> read_whole_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;

    std::string content;
    std::error_code ec;
    if (const auto size = std::filesystem::file_size(path, ec); !ec) content.resize(size);
    in.read(content.data(), static_cast<std::streamsize>(content.size()));
    content.resize(static_cast<std::size_t>(in.gcount()));

    std::array<char, 16 * 1024> chunk;
    while (in.read(chunk.data(), chunk.size()) || in.gcount() > 0) {
        content.append(chunk.data(), static_cast<std::size_t>(in.gcount()));
    }
    return content;
}

} // namespace core::tools::detail
