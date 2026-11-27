#include "DocumentLink.hpp"

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/PathUtils.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/utils/UriUtils.hpp"

#include <charconv>
#include <string>

namespace tui {
namespace {

[[nodiscard]] std::optional<int> parse_positive(std::string_view digits) {
    int value = 0;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), value);
    if (error != std::errc{} || end != digits.data() + digits.size() || value < 1) {
        return std::nullopt;
    }
    return value;
}

[[nodiscard]] bool all_digits(std::string_view text) {
    return !text.empty() && text.find_first_not_of("0123456789") == std::string_view::npos;
}

/// `L12`, `L12-L20`, `L12C3`, `L12-20`: the first line, or nothing.
[[nodiscard]] std::optional<int> fragment_line(std::string_view fragment) {
    if (!fragment.starts_with('L') && !fragment.starts_with('l')) return std::nullopt;
    fragment.remove_prefix(1);
    const auto end = fragment.find_first_not_of("0123456789");
    return parse_positive(fragment.substr(0, end));
}

/// Splits a trailing `:12` or `:12:3` off @p path.
[[nodiscard]] std::optional<int> take_line_suffix(std::string& path) {
    const auto last = path.rfind(':');
    if (last == std::string::npos || !all_digits(std::string_view(path).substr(last + 1))) {
        return std::nullopt;
    }
    std::size_t line_colon = last;
    if (const auto previous = path.rfind(':', last == 0 ? 0 : last - 1);
        previous != std::string::npos && previous < last
        && all_digits(std::string_view(path).substr(previous + 1, last - previous - 1))) {
        line_colon = previous;  // path:line:column
    }
    if (line_colon == 0) return std::nullopt;
    const auto digits = std::string_view(path).substr(
        line_colon + 1, (line_colon == last ? path.size() : last) - line_colon - 1);
    const auto line = parse_positive(digits);
    if (line) path.erase(line_colon);
    return line;
}

/// A URI scheme (`https:`, `mailto:`, `vscode:`), told apart from a Windows
/// drive (`C:\`) and from a file name followed by a line (`README.md:12`).
[[nodiscard]] bool has_scheme(std::string_view href) {
    const auto colon = href.find(':');
    if (colon == std::string_view::npos || colon < 2) return false;
    if (!core::utils::ascii::is_alpha(static_cast<unsigned char>(href[0]))) return false;
    for (const char ch : href.substr(0, colon)) {
        const auto c = static_cast<unsigned char>(ch);
        if (!core::utils::ascii::is_alnum(c) && ch != '+' && ch != '-' && ch != '.') return false;
    }
    const auto rest = href.substr(colon + 1);
    const auto rest_end = rest.find(':');
    return !all_digits(rest.substr(0, rest_end));
}

} // namespace

std::optional<DocumentTarget> parse_document_link(std::string_view href) {
    href = core::utils::str::trim_ascii_view(href);
    if (href.starts_with('<') && href.ends_with('>') && href.size() >= 2) {
        href = href.substr(1, href.size() - 2);
    } else if (const auto title = href.find_first_of(" \t");
               title != std::string_view::npos
               && title + 1 < href.size()
               && (href[title + 1] == '"' || href[title + 1] == '\'')) {
        // [label](path "title")
        href = core::utils::str::trim_ascii_view(href.substr(0, title));
    }
    if (href.empty() || href.starts_with('#')) return std::nullopt;

    if (core::utils::ascii::istarts_with(href, "file://")) {
        href.remove_prefix(7);
        // file://localhost/abs and file:///abs both name /abs.
        if (const auto slash = href.find('/'); slash != std::string_view::npos) {
            href.remove_prefix(slash);
        } else {
            return std::nullopt;
        }
    } else if (has_scheme(href)) {
        return std::nullopt;
    }

    std::optional<int> line;
    if (const auto hash = href.find('#'); hash != std::string_view::npos) {
        line = fragment_line(href.substr(hash + 1));
        href = href.substr(0, hash);
    }
    if (const auto query = href.find('?'); query != std::string_view::npos) {
        href = href.substr(0, query);
    }

    std::string path;
    if (!core::utils::uri::percent_decode(href, path)) path.assign(href);
    if (!line) line = take_line_suffix(path);
    if (path.empty()) return std::nullopt;
    return DocumentTarget{.path = std::filesystem::path(path), .line = line};
}

DocumentTarget resolve_document_target(DocumentTarget target, const std::filesystem::path& base) {
    auto path = core::utils::path::expand_user_path(target.path.string());
    if (path.is_relative()) path = base / path;
    target.path = path.lexically_normal();
    return target;
}

} // namespace tui
