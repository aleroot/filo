#pragma once

// ---------------------------------------------------------------------------
// Configuration bundles: the portable file `filo --export` writes and
// `filo --import` reads, plus the selector language both use to pick items.
//
// A bundle is a flat list of items. Every item has a kind and a name, and the
// pair is what a selector matches:
//
//   providers            every provider            (kind only)
//   provider:claude      one provider              (kind:pattern)
//   memories:~/src/*     memories of matching projects
//   claude               anything named claude     (bare pattern, any kind)
//
// Patterns are shell-style globs (`*`, `?`) over the whole name; `*` also
// crosses `/` so memory project paths can be matched by suffix.
// ---------------------------------------------------------------------------

#include "core/utils/JsonValue.hpp"

#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::portability {

enum class ItemKind {
    Setting,     ///< One top-level key of the user configuration files.
    Provider,    ///< One entry of `providers` across the configuration files.
    Mcp,         ///< One user-level MCP server definition.
    Credential,  ///< One OAuth token store (oauth_<name>.json + companions).
    Memory,      ///< The active memories of one project (or `global`).
};

struct KindInfo {
    ItemKind kind;
    std::string_view name;         ///< Canonical selector spelling (plural).
    std::string_view singular;     ///< Accepted alias.
    std::string_view description;
    bool secret;                   ///< Exported only with --include-secrets.
};

/// The one table of item kinds; selectors, help text and listings derive from it.
[[nodiscard]] std::span<const KindInfo> item_kinds() noexcept;
[[nodiscard]] const KindInfo& kind_info(ItemKind kind) noexcept;
/// Accepts the canonical or singular spelling, case-insensitively.
[[nodiscard]] std::optional<ItemKind> parse_kind(std::string_view token) noexcept;

struct BundleItem {
    ItemKind kind = ItemKind::Setting;
    std::string name;
    /// Kind-specific payload. Configuration kinds map a file name to the
    /// fragment that file holds; credentials map a file name to its content;
    /// memories hold {"entries": [...]}.
    core::utils::JsonValue data = core::utils::JsonValue::object();
    /// Secret fields left out of `data`, as RFC 6901 JSON Pointers into it
    /// ("/<file>/<key>/<key>"). Import keeps whatever value the destination
    /// already has at these paths.
    std::vector<std::string> redacted = {};
    /// The item itself is a secret (credential), not merely redacted.
    bool secret = false;
};

struct Bundle {
    static constexpr std::string_view kFormat = "filo-config-bundle";
    static constexpr int kVersion = 1;

    int version = kVersion;
    std::string created_at;
    std::string filo_version;
    /// Home directory of the exporting machine. Paths under it are stored as
    /// `~/...`; this lets --map-path accept the original absolute spelling.
    std::string source_home;
    std::vector<BundleItem> items;

    [[nodiscard]] bool contains_secrets() const noexcept;
};

[[nodiscard]] std::string serialize_bundle(const Bundle& bundle);
/// Validates format and version; rejects bundles written by a newer Filo.
[[nodiscard]] std::expected<Bundle, std::string> parse_bundle(std::string_view text);

struct Selector {
    std::optional<ItemKind> kind;   ///< nullopt: any kind.
    std::string pattern = "*";

    /// `pattern` is a POSIX fnmatch(3) glob over the whole name; `*` also
    /// crosses '/'.
    [[nodiscard]] bool matches(ItemKind item_kind, std::string_view name) const;
};

[[nodiscard]] std::expected<Selector, std::string> parse_selector(std::string_view token);

/// --only / --except. An item passes when it matches some `only` selector
/// (or `only` is empty) and no `except` selector.
struct ItemFilter {
    std::vector<Selector> only;
    std::vector<Selector> except;

    [[nodiscard]] bool matches(ItemKind kind, std::string_view name) const;

    [[nodiscard]] static std::expected<ItemFilter, std::string> parse(
        std::span<const std::string> only_tokens,
        std::span<const std::string> except_tokens);
};

} // namespace core::portability
