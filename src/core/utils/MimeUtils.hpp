#pragma once

#include "AsciiUtils.hpp"

#include <filesystem>
#include <httplib.h>
#include <map>
#include <ranges>
#include <string>

namespace core::utils::mime {

[[nodiscard]] inline std::string guess_type(const std::filesystem::path& path, bool binary) {
    if (binary) {
        return "application/octet-stream";
    }

    static const std::map<std::string, std::string> kProjectOverrides{
        // cpp-httplib already handles common web/media extensions. We override
        // source-code types used in this project that are not covered there.
        {"c", "text/x-c++src"},
        {"cc", "text/x-c++src"},
        {"cpp", "text/x-c++src"},
        {"cxx", "text/x-c++src"},
        {"h", "text/x-c++src"},
        {"hh", "text/x-c++src"},
        {"hpp", "text/x-c++src"},
        {"bash", "text/x-shellscript"},
        {"markdown", "text/markdown"},
        {"md", "text/markdown"},
        {"py", "text/x-python"},
        {"sh", "text/x-shellscript"},
        {"ts", "text/typescript"},
        {"zsh", "text/x-shellscript"},
    };

    std::filesystem::path normalized = path;
    std::string ext = normalized.extension().string();
    std::ranges::transform(
        ext,
        ext.begin(),
        [](unsigned char ch) { return ascii::to_lower(static_cast<char>(ch)); });
    if (!ext.empty()) {
        normalized.replace_extension(ext);
    }

    return httplib::detail::find_content_type(
        normalized.generic_string(),
        kProjectOverrides,
        "text/plain");
}

} // namespace core::utils::mime
