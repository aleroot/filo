#include "PatchTargets.hpp"

#include <cctype>
#include <optional>

namespace core::tools {
namespace {

std::string trim_ascii(std::string_view input) {
    std::size_t start = 0;
    while (start < input.size()
           && std::isspace(static_cast<unsigned char>(input[start])) != 0) {
        ++start;
    }

    std::size_t end = input.size();
    while (end > start
           && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return std::string(input.substr(start, end - start));
}

std::optional<std::string> extract_patch_path(std::string_view line) {
    if (!(line.starts_with("--- ") || line.starts_with("+++ "))) {
        return std::nullopt;
    }

    std::string raw = trim_ascii(line.substr(4));
    if (raw.empty()) {
        return std::nullopt;
    }

    const auto tab_pos = raw.find('\t');
    if (tab_pos != std::string::npos) {
        raw.erase(tab_pos);
        raw = trim_ascii(raw);
    }

    if (raw == "/dev/null") {
        return std::nullopt;
    }

    if ((raw.starts_with("a/") || raw.starts_with("b/")) && raw.size() > 2) {
        raw.erase(0, 2);
    }

    if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') {
        raw = raw.substr(1, raw.size() - 2);
    }

    if (raw.empty()) {
        return std::nullopt;
    }
    return raw;
}

} // namespace

std::vector<std::string> patch_target_paths(std::string_view patch_text) {
    std::vector<std::string> paths;
    while (!patch_text.empty()) {
        const auto newline = patch_text.find('\n');
        std::string_view line = patch_text.substr(0, newline);
        patch_text.remove_prefix(newline == std::string_view::npos
            ? patch_text.size() : newline + 1);
        if (line.ends_with('\r')) {
            line.remove_suffix(1);
        }
        if (auto path = extract_patch_path(line)) {
            paths.push_back(std::move(*path));
        }
    }
    return paths;
}

} // namespace core::tools
