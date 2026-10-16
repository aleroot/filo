#include "ConfigPortability.hpp"

#include "core/auth/FileTokenStore.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/mcp/McpOAuth.hpp"
#include "core/memory/MemoryStore.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/JsonValue.hpp"
#include "core/utils/PathUtils.hpp"
#include "core/version/Version.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <iterator>
#include <map>
#include <ranges>
#include <set>

namespace core::portability {
namespace {

namespace fs = std::filesystem;
using core::auth::FileTokenStore;
using core::utils::JsonValue;
using core::utils::kPrivateFilePermissions;
using core::utils::path::abbreviate_user_path;
using core::utils::path::expand_user_path;

// ---------------------------------------------------------------------------
// Where things live. These tables are the whole on-disk model; nothing below
// hardcodes a file name or a section key.
// ---------------------------------------------------------------------------

/// User configuration files that may carry settings, providers or MCP servers.
constexpr std::array<std::string_view, 6> kConfigFiles{
    "config.json",
    "auth_defaults.json",
    "model_defaults.json",
    "profile_defaults.json",
    "settings.json",
    "mcp_servers.json",
};

/// Field names that hold a secret wherever they appear in a configuration item.
constexpr std::array<std::string_view, 6> kSecretKeys{
    "api_key", "oauth_client_secret", "client_secret",
    "access_token", "refresh_token", "password",
};

/// MCP servers additionally carry free-form headers (Authorization) and env.
constexpr std::array<std::string_view, 2> kMcpSecretKeys{"headers", "env"};

/// A keyed collection inside the configuration files that one kind owns.
struct SectionSpec {
    ItemKind kind;
    std::string_view key;
    bool named_array;  ///< [{"name": ...}] rather than {"<name>": {...}}.
    std::span<const std::string_view> extra_secret_keys;
};

constexpr std::array kSections{
    SectionSpec{ItemKind::Provider, "providers", false, {}},
    SectionSpec{ItemKind::Mcp, "mcp_servers", true, kMcpSecretKeys},
};

/// Files a token store needs besides its token file to be usable elsewhere.
struct CredentialCompanions {
    std::string_view store;
    std::array<std::string_view, 2> files;
};

/// Kimi tokens embed a device id that must match the X-Msh-Device-Id header,
/// which Filo reads from kimi_device_id (see KimiOAuthFlow.cpp).
constexpr std::array kCredentialCompanions{
    CredentialCompanions{"kimi", {"kimi_device_id", "kimi_oauth_host"}},
};

constexpr std::string_view kBackupDirectory = "backups";

/// The configuration file Filo creates with defaults on first start.
constexpr std::string_view kSeededConfigFile = kConfigFiles.front();

[[nodiscard]] const SectionSpec* section_for(ItemKind kind) noexcept {
    for (const auto& section : kSections) {
        if (section.kind == kind) return &section;
    }
    return nullptr;
}

[[nodiscard]] bool is_section_key(std::string_view key) noexcept {
    return std::ranges::any_of(kSections, [key](const auto& s) { return s.key == key; });
}

[[nodiscard]] bool is_config_file(std::string_view name) noexcept {
    return std::ranges::find(kConfigFiles, name) != kConfigFiles.end();
}

/// Companion files of a credential: its kCredentialCompanions entry, plus the
/// metadata store an MCP login refreshes through. Only those present are used.
[[nodiscard]] std::vector<std::string> companions_for(std::string_view store) {
    std::vector<std::string> files;
    if (const auto entry = std::ranges::find(kCredentialCompanions, store,
                                             &CredentialCompanions::store);
        entry != kCredentialCompanions.end()) {
        files.assign(entry->files.begin(), entry->files.end());
    }
    files.push_back(FileTokenStore::file_name(core::mcp::mcp_oauth_meta_id(store)));
    return files;
}

/// Companions are stored verbatim, except JSON files, which stay structured
/// so a bundle diff shows fields rather than an escaped string.
[[nodiscard]] bool is_json_file(std::string_view file) noexcept {
    return fs::path(file).extension() == ".json";
}

// ---------------------------------------------------------------------------
// Redaction. Redacted fields are recorded as RFC 6901 pointers into the
// item's data ("/<file>/<key>/..."), so any key survives the round trip.
// ---------------------------------------------------------------------------

[[nodiscard]] bool is_secret_key(std::string_view key, std::span<const std::string_view> extra) {
    return std::ranges::find(kSecretKeys, key) != kSecretKeys.end()
        || std::ranges::find(extra, key) != extra.end();
}

void redact_secrets(JsonValue& node, const std::string& path,
                    std::span<const std::string_view> extra,
                    std::vector<std::string>& redacted) {
    if (!node.is_object()) return;
    auto& members = node.members();
    for (auto it = members.begin(); it != members.end();) {
        const std::string child = path + "/" + JsonValue::pointer_token(it->first);
        if (is_secret_key(it->first, extra)) {
            redacted.push_back(child);
            it = members.erase(it);
            continue;
        }
        redact_secrets(it->second, child, extra, redacted);
        ++it;
    }
}

/// Copies the destination's value at each redacted pointer of `file` into
/// `incoming` (that file's fragment). Returns the pointers for which this
/// machine has no value either.
[[nodiscard]] std::vector<std::string> restore_redacted(
    JsonValue& incoming, const JsonValue* existing,
    std::string_view file, const std::vector<std::string>& redacted) {
    std::vector<std::string> missing;
    const std::string prefix = "/" + JsonValue::pointer_token(file);
    for (const auto& pointer : redacted) {
        if (!pointer.starts_with(prefix + "/")) continue;
        const std::string_view inner = std::string_view(pointer).substr(prefix.size());
        const JsonValue* value = existing ? existing->at_pointer(inner) : nullptr;
        if (!value) {
            missing.push_back(pointer);
            continue;
        }
        incoming.set_at_pointer(inner, *value);
    }
    return missing;
}

// ---------------------------------------------------------------------------
// Paths in memories. Roots under the exporting user's home are stored as
// `~/...` so they land in the importing user's home without any mapping.
// ---------------------------------------------------------------------------

[[nodiscard]] std::string memory_item_name(std::string_view root, std::string_view home) {
    return root.empty() ? std::string("global") : abbreviate_user_path(root, home);
}

[[nodiscard]] std::string resolve_memory_root(std::string_view stored,
                                              std::string_view source_home,
                                              std::string_view local_home,
                                              const std::vector<PathMapping>& mappings) {
    if (stored.empty()) return {};
    const fs::path source_abs = expand_user_path(stored, source_home.empty() ? local_home : source_home);
    for (const auto& mapping : mappings) {
        // A mapping may be spelled the way the bundle stores the path (~/...)
        // or the way it was on the exporting machine (/Users/...).
        for (const fs::path& candidate : {fs::path(stored), source_abs}) {
            const auto rest = core::utils::path::relative_within(candidate, mapping.from);
            if (!rest) continue;
            fs::path mapped = mapping.to;
            if (*rest != ".") mapped /= *rest;
            return expand_user_path(mapped.string(), local_home).lexically_normal().string();
        }
    }
    return expand_user_path(stored, local_home).string();
}

// ---------------------------------------------------------------------------
// Collection (export)
// ---------------------------------------------------------------------------

using ItemPredicate = std::function<bool(ItemKind, std::string_view)>;

class Collector {
public:
    Collector(const ItemPredicate& wanted, bool include_secrets)
        : wanted_(wanted), include_secrets_(include_secrets) {}

    /// Adds `fragment` as the content `file` holds for item (kind, name).
    void add_fragment(ItemKind kind, std::string_view name, std::string_view file,
                      JsonValue fragment, std::span<const std::string_view> extra_secrets) {
        if (!wanted_(kind, name)) return;
        auto& item = item_for(kind, name);
        if (!include_secrets_) {
            redact_secrets(fragment, "/" + JsonValue::pointer_token(file), extra_secrets,
                           item.redacted);
        }
        item.data.set(file, std::move(fragment));
    }

    BundleItem& item_for(ItemKind kind, std::string_view name) {
        const auto key = std::pair{kind, std::string(name)};
        if (const auto it = index_.find(key); it != index_.end()) return items_[it->second];
        index_.emplace(key, items_.size());
        items_.push_back(BundleItem{.kind = kind, .name = std::string(name)});
        return items_.back();
    }

    [[nodiscard]] bool wanted(ItemKind kind, std::string_view name) const {
        return wanted_(kind, name);
    }
    [[nodiscard]] bool include_secrets() const noexcept { return include_secrets_; }

    /// Items in kind-table order, then in discovery order.
    [[nodiscard]] std::vector<BundleItem> take() && {
        std::ranges::stable_sort(items_, {}, [](const BundleItem& item) {
            return static_cast<int>(item.kind);
        });
        return std::move(items_);
    }

private:
    const ItemPredicate& wanted_;
    bool include_secrets_;
    std::vector<BundleItem> items_;
    std::map<std::pair<ItemKind, std::string>, std::size_t> index_;
};

void collect_config(const PortabilityPaths& paths, Collector& collector, ExportResult& result) {
    for (const auto file : kConfigFiles) {
        const fs::path path = paths.config_dir / file;
        std::error_code ec;
        if (!fs::exists(path, ec)) continue;
        std::string error;
        auto root = JsonValue::load(path, &error);
        if (!root) result.warnings.push_back(error);
        if (!root || !root->is_object()) continue;

        for (const auto& [key, value] : root->members()) {
            if (!is_section_key(key)) {
                collector.add_fragment(ItemKind::Setting, key, file, value, {});
            }
        }
        for (const auto& section : kSections) {
            const auto* entries = root->find(section.key);
            if (!entries) continue;
            if (!section.named_array && entries->is_object()) {
                for (const auto& [name, value] : entries->members()) {
                    collector.add_fragment(section.kind, name, file, value,
                                           section.extra_secret_keys);
                }
            } else if (section.named_array && entries->is_array()) {
                for (const auto& value : entries->items()) {
                    const auto* name = value.find("name");
                    if (!name || name->as_string().empty()) continue;
                    collector.add_fragment(section.kind, name->as_string(), file, value,
                                           section.extra_secret_keys);
                }
            }
        }
    }
}

void collect_credentials(const PortabilityPaths& paths, Collector& collector,
                         ExportResult& result) {
    std::error_code ec;
    std::set<std::string> stores;
    for (const auto& entry : fs::directory_iterator(paths.config_dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        auto store = FileTokenStore::provider_id_from_file_name(entry.path().filename().string());
        if (store && FileTokenStore::is_valid_provider_id(*store)) stores.insert(std::move(*store));
    }
    // A file that is some store's companion travels with that store.
    for (const auto& store : std::set(stores)) {
        for (const auto& companion : companions_for(store)) {
            if (auto id = FileTokenStore::provider_id_from_file_name(companion)) stores.erase(*id);
        }
    }

    for (const auto& store : stores) {
        if (!collector.wanted(ItemKind::Credential, store)) continue;
        if (!collector.include_secrets()) {
            result.withheld.push_back(std::format("{}:{}", kind_info(ItemKind::Credential).name, store));
            continue;
        }
        const std::string token_file = FileTokenStore::file_name(store);
        std::string error;
        auto token = JsonValue::load(paths.config_dir / token_file, &error);
        if (!token || !token->is_object()) {
            result.warnings.push_back(error.empty()
                ? std::format("skipped credential {}: token file is not a JSON object", store)
                : std::format("skipped credential {}: {}", store, error));
            continue;
        }
        if (const auto* access = token->find("access_token"); !access || access->as_string().empty()) {
            result.warnings.push_back(std::format("skipped credential {}: no access_token", store));
            continue;
        }
        auto& item = collector.item_for(ItemKind::Credential, store);
        item.secret = true;
        item.data.set(token_file, std::move(*token));
        for (const auto& companion : companions_for(store)) {
            const fs::path path = paths.config_dir / companion;
            if (!fs::exists(path, ec)) continue;
            if (is_json_file(companion)) {
                if (auto json = JsonValue::load(path, &error)) item.data.set(companion, std::move(*json));
                else result.warnings.push_back(std::format("credential {}: {}", store, error));
            } else if (auto text = core::utils::read_file(path)) {
                item.data.set(companion, std::move(*text));
            }
        }
    }
}

void collect_memories(const PortabilityPaths& paths, Collector& collector,
                      ExportResult& result) {
    std::error_code ec;
    if (!fs::exists(paths.memory_file, ec)) return;
    std::string error;
    const auto entries = core::memory::MemoryStore(paths.memory_file).list(false, &error);
    if (!error.empty()) {
        result.warnings.push_back(std::format("memories not exported: {}", error));
        return;
    }
    std::size_t session_scoped = 0;
    for (const auto& entry : entries) {
        const std::string name = memory_item_name(entry.project_root, paths.home);
        if (!collector.wanted(ItemKind::Memory, name)) continue;
        // Session memories are keyed to one session id on this machine.
        if (entry.scope == "session") {
            ++session_scoped;
            continue;
        }
        auto portable = JsonValue::parse(core::memory::serialize_memory_entry(entry));
        if (!portable) continue;
        portable->set("project_root", abbreviate_user_path(entry.project_root, paths.home));
        portable->erase("supersedes");
        auto& item = collector.item_for(ItemKind::Memory, name);
        auto* list = item.data.find("entries");
        if (!list) list = &item.data.set("entries", JsonValue::array());
        list->items().push_back(std::move(*portable));
    }
    if (session_scoped > 0) {
        result.warnings.push_back(std::format(
            "{} session-scoped memory item(s) not exported: they belong to one session.",
            session_scoped));
    }
}

[[nodiscard]] ExportResult collect(const PortabilityPaths& paths,
                                   const ItemPredicate& wanted,
                                   bool include_secrets) {
    ExportResult result;
    result.bundle.created_at = core::memory::MemoryStore::now_iso8601();
    result.bundle.filo_version = std::string(core::version::value);
    result.bundle.source_home = paths.home;
    Collector collector(wanted, include_secrets);
    collect_config(paths, collector, result);
    collect_credentials(paths, collector, result);
    collect_memories(paths, collector, result);
    result.bundle.items = std::move(collector).take();
    return result;
}

// ---------------------------------------------------------------------------
// Application (import)
// ---------------------------------------------------------------------------

[[nodiscard]] ImportedItem failed(const BundleItem& item, std::string detail) {
    return {item.kind, item.name, ImportAction::Failed, std::move(detail)};
}

/// The existing fragment for (kind, name) inside one configuration file root.
[[nodiscard]] JsonValue* locate(JsonValue& root, ItemKind kind, std::string_view name) {
    if (kind == ItemKind::Setting) return root.find(name);
    const auto* section = section_for(kind);
    auto* entries = root.find(section->key);
    if (!entries) return nullptr;
    if (!section->named_array) return entries->is_object() ? entries->find(name) : nullptr;
    if (!entries->is_array()) return nullptr;
    for (auto& value : entries->items()) {
        if (const auto* id = value.find("name"); id && id->as_string() == name) return &value;
    }
    return nullptr;
}

/// Stores `fragment` as the content of (kind, name) inside `root`.
void place(JsonValue& root, ItemKind kind, std::string_view name, JsonValue fragment) {
    if (auto* existing = locate(root, kind, name)) {
        *existing = std::move(fragment);
        return;
    }
    if (kind == ItemKind::Setting) {
        root.set(name, std::move(fragment));
        return;
    }
    const auto* section = section_for(kind);
    auto* entries = root.find(section->key);
    if (!section->named_array) {
        if (!entries || !entries->is_object()) entries = &root.set(section->key, JsonValue::object());
        entries->set(name, std::move(fragment));
    } else {
        if (!entries || !entries->is_array()) entries = &root.set(section->key, JsonValue::array());
        entries->items().push_back(std::move(fragment));
    }
}

/// What a configuration file holds before anything is imported into it.
/// config.json is seeded with the defaults Filo would write on first start:
/// creating it with only the imported item would shadow them for good.
[[nodiscard]] JsonValue initial_content(std::string_view file) {
    if (file == kSeededConfigFile) {
        if (auto defaults = JsonValue::parse(core::config::default_user_config_json())) {
            return std::move(*defaults);
        }
    }
    return JsonValue::object();
}

[[nodiscard]] ImportedItem apply_config_item(const BundleItem& item,
                                             const PortabilityPaths& paths,
                                             bool dry_run) {
    if (!item.data.is_object() || item.data.members().empty()) {
        return failed(item, "item has no configuration data");
    }
    if (item.kind == ItemKind::Setting) {
        if (const auto owner = std::ranges::find(kSections, item.name, &SectionSpec::key);
            owner != kSections.end()) {
            return failed(item, std::format("'{}' is imported as individual {} items",
                                            item.name, kind_info(owner->kind).name));
        }
    }
    for (const auto& [file, _] : item.data.members()) {
        if (!is_config_file(file)) return failed(item, std::format("unknown configuration file '{}'", file));
    }

    bool created_everywhere = true;
    std::vector<std::string> changed_files;
    std::vector<std::string> missing_secrets;
    for (const auto& [file, fragment] : item.data.members()) {
        const fs::path path = paths.config_dir / file;
        std::string lock_error;
        auto lock = dry_run ? std::nullopt
            : core::utils::InterprocessFileLock::acquire(core::utils::lock_path_for(path), &lock_error);
        if (!dry_run && !lock) return failed(item, lock_error);

        std::string error;
        std::optional<JsonValue> root;
        std::error_code ec;
        if (fs::exists(path, ec)) {
            root = JsonValue::load(path, &error);
            if (!root) return failed(item, error);
        } else {
            root = initial_content(file);
        }
        if (!root->is_object()) return failed(item, std::format("{} is not a JSON object", path.string()));

        JsonValue incoming = fragment;
        if (item.kind == ItemKind::Mcp) {
            if (!incoming.is_object()) return failed(item, "MCP server entry must be an object");
            incoming.set("name", item.name);
        }
        const JsonValue* existing = locate(*root, item.kind, item.name);
        auto missing = restore_redacted(incoming, existing, file, item.redacted);
        std::ranges::move(missing, std::back_inserter(missing_secrets));
        if (existing) created_everywhere = false;
        if (existing && *existing == incoming) continue;

        place(*root, item.kind, item.name, std::move(incoming));
        changed_files.push_back(file);
        if (!dry_run && !core::utils::atomic_write_file(path, root->dump() + '\n', &error)) {
            return failed(item, error);
        }
    }

    const auto joined = [](const std::vector<std::string>& parts) {
        return std::ranges::to<std::string>(parts | std::views::join_with(std::string_view(", ")));
    };
    std::string detail = joined(changed_files);
    if (!missing_secrets.empty()) {
        if (!detail.empty()) detail += "; ";
        detail += "not in bundle and not set here: " + joined(missing_secrets);
    }
    const auto action = changed_files.empty() ? ImportAction::Unchanged
        : created_everywhere                  ? ImportAction::Created
                                              : ImportAction::Updated;
    return {item.kind, item.name, action, std::move(detail)};
}

[[nodiscard]] ImportedItem apply_credential(const BundleItem& item,
                                            const PortabilityPaths& paths,
                                            bool dry_run) {
    if (!FileTokenStore::is_valid_provider_id(item.name)) {
        return failed(item, "credential name is not a valid token store name");
    }
    const std::string token_file = FileTokenStore::file_name(item.name);
    const auto companions = companions_for(item.name);
    if (!item.data.is_object()) return failed(item, "credential data must be an object");
    for (const auto& [file, content] : item.data.members()) {
        const bool companion = std::ranges::contains(companions, file);
        if (file != token_file && !companion) {
            return failed(item, std::format("unexpected file '{}' in credential", file));
        }
        if (companion && !(is_json_file(file) ? content.is_object() : content.is_string())) {
            return failed(item, std::format("'{}' must be {}", file,
                                            is_json_file(file) ? "a JSON object" : "text"));
        }
    }
    const auto* token = item.data.find(token_file);
    const auto* access_token = token ? token->find("access_token") : nullptr;
    if (!access_token || access_token->as_string().empty()) {
        return failed(item, std::format("{} has no access_token", token_file));
    }

    // Hold the lock the token manager rotates refresh tokens under, so a
    // concurrent refresh cannot overwrite the imported login.
    std::unique_ptr<core::auth::ITokenStoreLock> lock;
    if (!dry_run) {
        try {
            lock = FileTokenStore(paths.config_dir.string()).acquire_refresh_lock(item.name);
        } catch (const std::exception& e) {
            return failed(item, e.what());
        }
    }

    const fs::path token_path = paths.config_dir / token_file;
    std::string error;
    std::error_code ec;
    const bool existed = fs::exists(token_path, ec);
    bool changed = false;
    for (const auto& [file, content] : item.data.members()) {
        const fs::path path = paths.config_dir / file;
        const bool unchanged = content.is_string()
            ? core::utils::read_file(path) == content.as_string()
            : JsonValue::load(path) == content;
        if (unchanged) continue;
        changed = true;
        const std::string text = content.is_string() ? std::string(content.as_string())
                                                     : content.dump() + '\n';
        if (!dry_run && !core::utils::atomic_write_file(path, text, &error, kPrivateFilePermissions)) {
            return failed(item, error);
        }
    }
    const auto action = !changed ? ImportAction::Unchanged
        : existed                ? ImportAction::Updated
                                 : ImportAction::Created;
    return {item.kind, item.name, action, {}};
}

[[nodiscard]] ImportedItem apply_memory(const BundleItem& item,
                                        const Bundle& bundle,
                                        const PortabilityPaths& paths,
                                        const ImportOptions& options) {
    const auto* list = item.data.find("entries");
    if (!list || !list->is_array()) return failed(item, "memory item has no entries");
    std::vector<core::memory::MemoryEntry> entries;
    std::set<std::string> roots;
    for (const auto& value : list->items()) {
        auto entry = core::memory::parse_memory_entry(value.dump(0));
        if (!entry || entry->scope == "session") continue;
        entry->project_root = resolve_memory_root(
            entry->project_root, bundle.source_home, paths.home, options.path_mappings);
        roots.insert(entry->project_root.empty() ? "global" : entry->project_root);
        entries.push_back(std::move(*entry));
    }
    const auto outcome = core::memory::MemoryStore(paths.memory_file)
        .import_entries(std::move(entries), options.dry_run);
    if (!outcome.ok) return failed(item, outcome.message);

    std::string detail = std::format("{} added, {} already present", outcome.added, outcome.duplicates);
    if (outcome.archived > 0) {
        detail += std::format(", {} automatic note(s) archived to stay within the limit",
                              outcome.archived);
    }
    if (roots.size() == 1 && *roots.begin() != item.name
        && *roots.begin() != expand_user_path(item.name, paths.home).string()) {
        detail += " → " + *roots.begin();
    }
    return {item.kind, item.name,
            outcome.added > 0 ? ImportAction::Merged : ImportAction::Unchanged,
            std::move(detail)};
}

[[nodiscard]] std::string backup_stamp() {
    std::string stamp = core::memory::MemoryStore::now_iso8601();
    std::erase_if(stamp, [](char c) { return c == '-' || c == ':'; });
    return stamp;
}

/// Saves the current state of every item the import is about to replace, as a
/// bundle `filo --import` can restore. Memories are merged additively and
/// created items had no previous state, so neither needs a snapshot.
[[nodiscard]] std::expected<std::optional<fs::path>, std::string> write_backup(
    const std::vector<ImportedItem>& plan, const PortabilityPaths& paths,
    std::vector<std::string>& warnings) {
    std::set<std::pair<ItemKind, std::string>> targets;
    for (const auto& planned : plan) {
        if (planned.action == ImportAction::Updated) targets.emplace(planned.kind, planned.name);
    }
    if (targets.empty()) return std::nullopt;
    const ItemPredicate wanted = [&](ItemKind kind, std::string_view name) {
        return targets.contains({kind, std::string(name)});
    };
    auto snapshot = collect(paths, wanted, /*include_secrets=*/true);
    for (auto& warning : snapshot.warnings) warnings.push_back("backup: " + std::move(warning));
    if (snapshot.bundle.items.empty()) return std::nullopt;

    const fs::path directory = paths.config_dir / kBackupDirectory;
    fs::path path = directory / std::format("import-{}.json", backup_stamp());
    std::error_code ec;
    for (int n = 2; fs::exists(path, ec); ++n) {
        path = directory / std::format("import-{}-{}.json", backup_stamp(), n);
    }
    fs::create_directories(directory, ec);
    fs::permissions(directory, fs::perms::owner_all, fs::perm_options::replace, ec);
    std::string error;
    if (!core::utils::atomic_write_file(path, serialize_bundle(snapshot.bundle), &error,
                                        kPrivateFilePermissions)) {
        return std::unexpected(error);
    }
    return path;
}

} // namespace

PortabilityPaths PortabilityPaths::current() {
    return {
        .config_dir = core::config::ConfigManager::get_instance().get_config_dir(),
        .memory_file = core::memory::MemoryStore::default_path(),
        .home = core::utils::path::home_directory().string(),
    };
}

ExportResult export_bundle(const PortabilityPaths& paths, const ExportOptions& options) {
    const ItemPredicate wanted = [&](ItemKind kind, std::string_view name) {
        return options.filter.matches(kind, name);
    };
    return collect(paths, wanted, options.include_secrets);
}

std::expected<PathMapping, std::string> parse_path_mapping(std::string_view spec) {
    const auto equals = spec.find('=');
    if (equals == std::string_view::npos || equals == 0 || equals + 1 == spec.size()) {
        return std::unexpected(std::format("--map-path expects FROM=TO, got '{}'", spec));
    }
    return PathMapping{std::string(spec.substr(0, equals)), std::string(spec.substr(equals + 1))};
}

std::string_view import_action_name(ImportAction action) noexcept {
    switch (action) {
    case ImportAction::Created: return "create";
    case ImportAction::Updated: return "update";
    case ImportAction::Unchanged: return "unchanged";
    case ImportAction::Merged: return "merge";
    case ImportAction::Skipped: return "skipped";
    case ImportAction::Failed: return "failed";
    }
    return "?";
}

bool ImportResult::ok() const noexcept {
    return std::ranges::none_of(items, [](const ImportedItem& item) {
        return item.action == ImportAction::Failed;
    });
}

ImportResult import_bundle(const Bundle& bundle,
                           const PortabilityPaths& paths,
                           const ImportOptions& options) {
    const auto apply = [&](const BundleItem& item, bool dry_run) -> ImportedItem {
        switch (item.kind) {
        case ItemKind::Setting:
        case ItemKind::Provider:
        case ItemKind::Mcp:
            return apply_config_item(item, paths, dry_run);
        case ItemKind::Credential:
            return apply_credential(item, paths, dry_run);
        case ItemKind::Memory: {
            auto memory_options = options;
            memory_options.dry_run = dry_run;
            return apply_memory(item, bundle, paths, memory_options);
        }
        }
        return failed(item, "unsupported item kind");
    };

    std::vector<const BundleItem*> selected;
    for (const auto& item : bundle.items) {
        if (options.filter.matches(item.kind, item.name)) selected.push_back(&item);
    }

    // Plan first: every item is validated against the current files before
    // anything is written, so a bad item cannot leave a half-applied import.
    ImportResult result;
    for (const auto* item : selected) result.items.push_back(apply(*item, /*dry_run=*/true));
    if (options.dry_run) return result;

    if (!result.ok()) {
        for (auto& planned : result.items) {
            if (planned.action != ImportAction::Failed) {
                planned = {planned.kind, planned.name, ImportAction::Skipped,
                           "not imported because another item failed"};
            }
        }
        return result;
    }

    auto backup = write_backup(result.items, paths, result.warnings);
    if (!backup) {
        for (auto& planned : result.items) {
            planned = {planned.kind, planned.name, ImportAction::Failed,
                       "not imported: backup failed: " + backup.error()};
        }
        return result;
    }
    result.backup = std::move(*backup);
    result.write_started = true;

    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (result.items[i].action == ImportAction::Unchanged) continue;
        result.items[i] = apply(*selected[i], /*dry_run=*/false);
    }
    return result;
}

} // namespace core::portability
