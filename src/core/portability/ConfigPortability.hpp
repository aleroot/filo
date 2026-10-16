#pragma once

// ---------------------------------------------------------------------------
// Export and import of Filo's user state.
//
// Where each kind of item lives on disk is described by the tables at the top
// of ConfigPortability.cpp (config files, sections, secret keys, credential
// companions); nothing else hardcodes a location:
//
//   settings     top-level keys of the user config files (minus the sections
//                owned by providers and mcp)
//   providers    `providers.<name>` in config.json, auth_defaults.json and
//                model_defaults.json, gathered into one item per name
//   mcp          `mcp_servers[name]` in config.json and mcp_servers.json
//   credentials  oauth_<name>.json (and companion files such as Kimi's
//                device id)
//   memories     memory.json entries, grouped by project
//
// Secrets never leave the machine by accident: unless include_secrets is set,
// credential items are withheld entirely and secret fields (api_key, client
// secrets, MCP headers/env) are cut out of the other items and listed in
// BundleItem::redacted. Import treats a redacted field as "keep what the
// destination already has", so a redacted bundle can never erase a key.
//
// Import replaces each selected configuration item as a unit, merges
// memories additively (deduplicated), validates every name that becomes a
// file path, and snapshots the items it is about to replace into a backup
// bundle that `filo --import` can restore.
// ---------------------------------------------------------------------------

#include "ConfigBundle.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace core::portability {

struct PortabilityPaths {
    std::filesystem::path config_dir;
    std::filesystem::path memory_file;
    /// This machine's home directory; `~/` in bundles expands to it.
    std::string home;

    /// The live locations: ConfigManager's config dir and MemoryStore's file.
    [[nodiscard]] static PortabilityPaths current();
};

struct ExportOptions {
    ItemFilter filter;
    bool include_secrets = false;
};

struct ExportResult {
    Bundle bundle;
    /// Secret items that matched the filter but were left out.
    std::vector<std::string> withheld;
    /// Unreadable files and other problems that did not stop the export.
    std::vector<std::string> warnings;
};

[[nodiscard]] ExportResult export_bundle(const PortabilityPaths& paths,
                                         const ExportOptions& options);

/// --map-path FROM=TO: rewrites memory project roots under FROM to TO.
struct PathMapping {
    std::string from;
    std::string to;
};

[[nodiscard]] std::expected<PathMapping, std::string> parse_path_mapping(std::string_view spec);

struct ImportOptions {
    ItemFilter filter;
    bool dry_run = false;
    std::vector<PathMapping> path_mappings;
};

enum class ImportAction { Created, Updated, Unchanged, Merged, Skipped, Failed };

[[nodiscard]] std::string_view import_action_name(ImportAction action) noexcept;

struct ImportedItem {
    ItemKind kind;
    std::string name;
    ImportAction action;
    std::string detail;
};

struct ImportResult {
    std::vector<ImportedItem> items;
    /// Where the replaced items were saved, when anything was replaced.
    /// Every item is validated before any is written: if one fails, the
    /// others come back Skipped and nothing changes on disk.
    std::optional<std::filesystem::path> backup;
    std::vector<std::string> warnings;
    /// Validation and the backup passed and files were being written. When
    /// false, a failed import changed nothing.
    bool write_started = false;

    [[nodiscard]] bool ok() const noexcept;
};

[[nodiscard]] ImportResult import_bundle(const Bundle& bundle,
                                         const PortabilityPaths& paths,
                                         const ImportOptions& options);

} // namespace core::portability
