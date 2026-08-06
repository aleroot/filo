#pragma once

#include "core/utils/StringUtils.hpp"

#include <string>
#include <string_view>

namespace core::auth {

// Terminals with bracketed-paste mode enabled wrap pasted text in these
// control sequences. ConsoleAuthUI reads directly from stdin while the TUI is
// suspended, so the wrappers can otherwise become part of a persisted secret.
[[nodiscard]] inline std::string normalize_secret_input(std::string_view input) {
    constexpr std::string_view kPasteBegin = "\x1b[200~";
    constexpr std::string_view kPasteEnd = "\x1b[201~";

    input = core::utils::str::trim_ascii_view(input);
    while (input.starts_with(kPasteBegin)) {
        input.remove_prefix(kPasteBegin.size());
        input = core::utils::str::trim_ascii_view(input);
    }
    while (input.ends_with(kPasteEnd)) {
        input.remove_suffix(kPasteEnd.size());
        input = core::utils::str::trim_ascii_view(input);
    }
    return std::string(input);
}

} // namespace core::auth
