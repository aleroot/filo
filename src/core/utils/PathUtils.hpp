#pragma once

#include "StringUtils.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

// Small, dependency-free helpers for turning human-typed path text into
// `std::filesystem::path` values and back again. They live here (rather than
// inside a single feature) because both the slash-command layer and the TUI
// filesystem browser need exactly the same `~` semantics; divergence between
// the two would be a user-visible inconsistency.
namespace core::utils::path {

/// The user's home directory, or an empty path when `$HOME` is unset.
[[nodiscard]] inline std::filesystem::path home_directory() {
    const char* home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0') {
        return {};
    }
    return std::filesystem::path(home);
}

/// Expand a leading `~` / `~/` to the user's home directory.
///
/// Surrounding whitespace is trimmed so that pasted paths behave. Unknown
/// forms such as `~other-user` are intentionally left untouched: guessing
/// another account's home directory would be worse than an honest failure.
[[nodiscard]] inline std::filesystem::path expand_user_path(std::string_view input) {
    const std::string text = core::utils::str::trim_ascii_copy(input);
    if (text != "~" && !text.starts_with("~/")) {
        return std::filesystem::path(text);
    }
    const auto home = home_directory();
    if (home.empty()) {
        return std::filesystem::path(text);
    }
    return text == "~" ? home : home / text.substr(2);
}

/// Inverse of `expand_user_path`: render @p path with the home prefix folded
/// back into `~` so breadcrumbs stay short and privacy-friendly.
[[nodiscard]] inline std::string abbreviate_user_path(const std::filesystem::path& path) {
    const auto home = home_directory();
    if (home.empty()) {
        return path.generic_string();
    }

    const std::string text = path.generic_string();
    const std::string home_text = home.generic_string();
    if (text == home_text) {
        return "~";
    }
    if (text.size() > home_text.size()
        && text.starts_with(home_text)
        && text[home_text.size()] == '/') {
        return "~" + text.substr(home_text.size());
    }
    return text;
}

} // namespace core::utils::path
