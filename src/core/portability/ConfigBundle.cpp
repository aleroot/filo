#include "ConfigBundle.hpp"

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/Gzip.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <fnmatch.h>
#include <ranges>

namespace core::portability {
namespace {

using core::utils::JsonValue;

constexpr std::array kKinds{
    KindInfo{ItemKind::Setting, "settings", "setting",
             "top-level options of config.json and the other user config files", false},
    KindInfo{ItemKind::Provider, "providers", "provider",
             "provider definitions (model, endpoint, auth type; API keys are secrets)", false},
    KindInfo{ItemKind::Mcp, "mcp", "mcp-server",
             "user-level MCP servers (headers and env are secrets)", false},
    KindInfo{ItemKind::Credential, "credentials", "credential",
             "OAuth logins (oauth_<name>.json), including MCP server logins", true},
    KindInfo{ItemKind::Memory, "memories", "memory",
             "active project and global memories, one item per project", false},
};

[[nodiscard]] std::string kind_list() {
    return std::ranges::to<std::string>(kKinds | std::views::transform(&KindInfo::name)
                                        | std::views::join_with(std::string_view(", ")));
}

} // namespace

std::span<const KindInfo> item_kinds() noexcept { return kKinds; }

const KindInfo& kind_info(ItemKind kind) noexcept {
    // kKinds lists every ItemKind, so the lookup cannot miss.
    return *std::ranges::find(kKinds, kind, &KindInfo::kind);
}

std::optional<ItemKind> parse_kind(std::string_view token) noexcept {
    for (const auto& info : kKinds) {
        if (core::utils::ascii::iequals(token, info.name)
            || core::utils::ascii::iequals(token, info.singular)) {
            return info.kind;
        }
    }
    return std::nullopt;
}

bool Bundle::contains_secrets() const noexcept {
    return std::ranges::any_of(items, &BundleItem::secret);
}

std::string serialize_bundle(const Bundle& bundle) {
    JsonValue root = JsonValue::object();
    root.set("format", Bundle::kFormat);
    root.set("version", JsonValue::integer(bundle.version));
    root.set("created_at", bundle.created_at);
    root.set("filo_version", bundle.filo_version);
    root.set("source_home", bundle.source_home);
    root.set("contains_secrets", bundle.contains_secrets());
    JsonValue::Array items;
    items.reserve(bundle.items.size());
    for (const auto& item : bundle.items) {
        JsonValue entry = JsonValue::object();
        entry.set("kind", kind_info(item.kind).name);
        entry.set("name", item.name);
        if (item.secret) entry.set("secret", true);
        if (!item.redacted.empty()) {
            JsonValue::Array redacted;
            for (const auto& path : item.redacted) redacted.emplace_back(path);
            entry.set("redacted", JsonValue(std::move(redacted)));
        }
        entry.set("data", item.data);
        items.push_back(std::move(entry));
    }
    root.set("items", JsonValue(std::move(items)));
    std::string text = root.dump(2);
    text += '\n';
    return text;
}

std::expected<Bundle, std::string> parse_bundle(std::string_view text) {
    std::string error;
    auto root = JsonValue::parse(text, &error);
    if (!root) return std::unexpected(std::format("not valid JSON: {}", error));
    const auto* format = root->find("format");
    if (!format || format->as_string() != Bundle::kFormat) {
        return std::unexpected(std::format(
            "not a Filo configuration bundle (expected \"format\": \"{}\")", Bundle::kFormat));
    }
    Bundle bundle;
    const auto* version = root->find("version");
    const auto version_number = version ? version->as_integer() : std::nullopt;
    if (!version_number || *version_number < 1) {
        return std::unexpected("bundle has no valid \"version\"");
    }
    if (*version_number > Bundle::kVersion) {
        return std::unexpected(std::format(
            "bundle version {} was written by a newer Filo (this one reads up to {})",
            *version_number, Bundle::kVersion));
    }
    bundle.version = static_cast<int>(*version_number);
    if (const auto* value = root->find("created_at")) bundle.created_at = value->as_string();
    if (const auto* value = root->find("filo_version")) bundle.filo_version = value->as_string();
    if (const auto* value = root->find("source_home")) bundle.source_home = value->as_string();

    const auto* items = root->find("items");
    if (!items || !items->is_array()) return std::unexpected("bundle has no \"items\" array");
    for (std::size_t index = 0; index < items->items().size(); ++index) {
        const auto& entry = items->items()[index];
        const auto* kind = entry.find("kind");
        const auto* name = entry.find("name");
        const auto* data = entry.find("data");
        if (!kind || !name || !data || name->as_string().empty()) {
            return std::unexpected(std::format(
                "item {} needs \"kind\", \"name\" and \"data\"", index + 1));
        }
        const auto parsed_kind = parse_kind(kind->as_string());
        if (!parsed_kind) {
            return std::unexpected(std::format(
                "item {} has unknown kind '{}' (known: {})",
                index + 1, kind->as_string(), kind_list()));
        }
        BundleItem item{
            .kind = *parsed_kind,
            .name = std::string(name->as_string()),
            .data = *data,
        };
        if (const auto* secret = entry.find("secret")) item.secret = secret->as_bool();
        // A credential is a secret whatever the file claims.
        item.secret = item.secret || kind_info(item.kind).secret;
        if (const auto* redacted = entry.find("redacted"); redacted && redacted->is_array()) {
            for (const auto& path : redacted->items()) {
                if (path.is_string()) item.redacted.emplace_back(path.as_string());
            }
        }
        bundle.items.push_back(std::move(item));
    }
    return bundle;
}

BundleEncoding bundle_encoding_for(std::string_view file_name) noexcept {
    return core::utils::ascii::iends_with(file_name, kCompressedBundleSuffix)
        ? BundleEncoding::Gzip : BundleEncoding::Json;
}

BundleEncoding detect_bundle_encoding(std::string_view bytes) noexcept {
    // A JSON document cannot begin with the gzip magic byte 0x1F.
    return core::utils::gzip::is_compressed(bytes) ? BundleEncoding::Gzip : BundleEncoding::Json;
}

std::string encode_bundle(const Bundle& bundle, BundleEncoding encoding) {
    std::string text = serialize_bundle(bundle);
    if (encoding == BundleEncoding::Json) return text;
    return core::utils::gzip::compress(text, {.level = 9});
}

std::expected<Bundle, std::string> decode_bundle(std::string_view bytes) {
    if (detect_bundle_encoding(bytes) == BundleEncoding::Json) return parse_bundle(bytes);
    const auto text = core::utils::gzip::decompress(bytes);
    if (!text) return std::unexpected("compressed bundle is damaged: " + text.error());
    return parse_bundle(*text);
}

bool Selector::matches(ItemKind item_kind, std::string_view name) const {
    // POSIX fnmatch without FNM_PATHNAME: `*` also crosses '/'.
    return (!kind || *kind == item_kind)
        && ::fnmatch(pattern.c_str(), std::string(name).c_str(), 0) == 0;
}

std::expected<Selector, std::string> parse_selector(std::string_view token) {
    const std::string clean = core::utils::str::trim_ascii_copy(token);
    if (clean.empty()) return std::unexpected("empty selector");
    if (const auto colon = clean.find(':'); colon != std::string::npos) {
        const auto kind_token = std::string_view(clean).substr(0, colon);
        const auto kind = parse_kind(kind_token);
        if (!kind) {
            return std::unexpected(std::format(
                "unknown kind '{}' in selector '{}' (known: {})", kind_token, clean, kind_list()));
        }
        std::string pattern = core::utils::str::trim_ascii_copy(
            std::string_view(clean).substr(colon + 1));
        return Selector{.kind = kind, .pattern = pattern.empty() ? "*" : std::move(pattern)};
    }
    if (const auto kind = parse_kind(clean)) return Selector{.kind = kind};
    return Selector{.pattern = clean};
}

bool ItemFilter::matches(ItemKind kind, std::string_view name) const {
    const auto hit = [&](const Selector& selector) { return selector.matches(kind, name); };
    return (only.empty() || std::ranges::any_of(only, hit)) && std::ranges::none_of(except, hit);
}

std::expected<ItemFilter, std::string> ItemFilter::parse(
    std::span<const std::string> only_tokens,
    std::span<const std::string> except_tokens) {
    ItemFilter filter;
    for (const auto& [tokens, out] : {std::pair{only_tokens, &filter.only},
                                      std::pair{except_tokens, &filter.except}}) {
        for (const auto& token : tokens) {
            auto selector = parse_selector(token);
            if (!selector) return std::unexpected(selector.error());
            out->push_back(std::move(*selector));
        }
    }
    return filter;
}

} // namespace core::portability
