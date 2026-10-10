#pragma once

#include "../utils/AsciiUtils.hpp"

#include <cstddef>
#include <string_view>

// Token rules shared by every prompt sigil (`@file`, `#conversation`), so the
// two kinds of reference start and end in exactly the same places.
namespace core::context {

/// True when a sigil at @p pos starts a token: start of text, whitespace, or
/// an opening bracket/quote. Keeps `a@b.com`, `C#` and URL fragments out.
[[nodiscard]] constexpr bool is_mention_boundary(std::string_view input,
                                                 std::size_t pos) noexcept {
    if (pos == 0) return true;
    const char before = input[pos - 1];
    return core::utils::ascii::is_space(static_cast<unsigned char>(before))
        || std::string_view{"([{\"'"}.contains(before);
}

/// Sentence punctuation that may follow an unquoted token without being part
/// of it (`see @a.cpp, then`).
[[nodiscard]] constexpr bool is_trailing_mention_punctuation(char ch) noexcept {
    return std::string_view{",.;:!?)]}"}.contains(ch);
}

} // namespace core::context
