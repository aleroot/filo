#pragma once

#include "StringUtils.hpp"

#include <cstdlib>
#include <filesystem>
#include <optional>
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

/// The part of @p path below @p base: "." when they are the same directory,
/// nullopt when @p path lies outside @p base. Purely lexical (no I/O), and
/// component-wise, so `/home/al` is not inside `/home/a`.
[[nodiscard]] inline std::optional<std::filesystem::path> relative_within(
    const std::filesystem::path& path, const std::filesystem::path& base) {
    if (base.empty()) {
        return std::nullopt;
    }
    auto relative = path.lexically_relative(base);
    if (relative.empty() || *relative.begin() == "..") {
        return std::nullopt;
    }
    return relative;
}

/// Expand a leading `~` / `~/` to @p home (left untouched when @p home is empty).
///
/// Surrounding whitespace is trimmed so that pasted paths behave. Unknown
/// forms such as `~other-user` are intentionally left untouched: guessing
/// another account's home directory would be worse than an honest failure.
[[nodiscard]] inline std::filesystem::path expand_user_path(std::string_view input,
                                                            const std::filesystem::path& home) {
    const std::string text = core::utils::str::trim_ascii_copy(input);
    if (home.empty() || (text != "~" && !text.starts_with("~/"))) {
        return std::filesystem::path(text);
    }
    return text == "~" ? home : home / text.substr(2);
}

/// Expand a leading `~` / `~/` to the user's home directory.
[[nodiscard]] inline std::filesystem::path expand_user_path(std::string_view input) {
    return expand_user_path(input, home_directory());
}

/// Inverse of `expand_user_path`: render @p path with the @p home prefix
/// folded back into `~` so breadcrumbs stay short and privacy-friendly.
[[nodiscard]] inline std::string abbreviate_user_path(const std::filesystem::path& path,
                                                      const std::filesystem::path& home) {
    const auto relative = relative_within(path, home);
    if (!relative) {
        return path.generic_string();
    }
    return *relative == "." ? "~" : "~/" + relative->generic_string();
}

/// `abbreviate_user_path` against the user's home directory.
[[nodiscard]] inline std::string abbreviate_user_path(const std::filesystem::path& path) {
    return abbreviate_user_path(path, home_directory());
}

} // namespace core::utils::path
