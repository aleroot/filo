#pragma once

#include <filesystem>
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

} // namespace core::tools::detail
