#include "SkillLoader.hpp"
#ifdef FILO_ENABLE_PYTHON
#include "PythonTool.hpp"
#endif
#include "../logging/Logger.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace core::tools {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// Returns @p sv with leading and trailing ASCII whitespace removed.
std::string trim(std::string_view sv) {
    const auto start = sv.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    const auto end = sv.find_last_not_of(" \t\r\n");
    return std::string(sv.substr(start, end - start + 1));
}

/// If @p line starts with @c "key:", returns the trimmed value after the colon.
/// Otherwise returns an empty string.
std::string extract_yaml_value(const std::string& line, std::string_view key) {
    const std::string prefix = std::string(key) + ":";
    if (line.rfind(prefix, 0) != 0) return {};
    return trim(line.substr(prefix.size()));
}

/// Splits @p sv on @c ',' and returns a vector of trimmed, non-empty tokens.
std::vector<std::string> split_comma(std::string_view sv) {
    std::vector<std::string> result;
    std::size_t start = 0;
    while (start <= sv.size()) {
        const auto comma = sv.find(',', start);
        const auto end   = (comma == std::string_view::npos) ? sv.size() : comma;
        const auto token = trim(sv.substr(start, end - start));
        if (!token.empty()) result.push_back(token);
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return result;
}

} // namespace

// ---------------------------------------------------------------------------
// SkillLoader
// ---------------------------------------------------------------------------

std::vector<fs::path> SkillLoader::default_search_paths() {
    std::vector<fs::path> paths;
    if (const char* home = std::getenv("HOME")) {
        paths.push_back(fs::path(home) / ".config" / "filo" / "skills");
    }
    paths.push_back(fs::current_path() / ".filo" / "skills");
    return paths;
}

std::optional<SkillManifest>
SkillLoader::parse_manifest(const fs::path& skill_dir) {
    const fs::path manifest_path = skill_dir / "SKILL.md";
    if (!fs::exists(manifest_path)) return std::nullopt;

    std::ifstream file(manifest_path);
    if (!file.is_open()) {
        core::logging::warn("SkillLoader: cannot open '{}'", manifest_path.string());
        return std::nullopt;
    }

    // Skip lines until the opening "---" delimiter.
    std::string line;
    bool found_delimiter = false;
    while (std::getline(file, line)) {
        if (trim(line) == "---") {
            found_delimiter = true;
            break;
        }
    }
    if (!found_delimiter) {
        core::logging::warn("SkillLoader: no frontmatter delimiter in '{}'",
                            manifest_path.string());
        return std::nullopt;
    }

    SkillManifest manifest;
    manifest.enabled = true;
    // Canonicalize only when the directory exists; during tests we may pass a
    // freshly-created temp path that is already absolute and canonical.
    manifest.skill_dir = fs::exists(skill_dir) ? fs::canonical(skill_dir) : skill_dir;

    // Whether the 'type' field was set explicitly in frontmatter.
    bool explicit_type = false;

    // Read key-value pairs until the closing "---" delimiter (or EOF).
    while (std::getline(file, line)) {
        const std::string t = trim(line);
        if (t == "---") break;
        if (t.empty() || t.front() == '#') continue;

        if (const auto v = extract_yaml_value(t, "name"); !v.empty()) {
            manifest.name = v;
        } else if (const auto v2 = extract_yaml_value(t, "description"); !v2.empty()) {
            manifest.description = v2;
        } else if (const auto v3 = extract_yaml_value(t, "entry_point"); !v3.empty()) {
            manifest.entry_point = v3;
        } else if (const auto v4 = extract_yaml_value(t, "enabled"); !v4.empty()) {
            manifest.enabled = (v4 != "false" && v4 != "0" && v4 != "no");
        } else if (const auto v5 = extract_yaml_value(t, "type"); !v5.empty()) {
            manifest.type   = (v5 == "prompt") ? SkillType::Prompt : SkillType::Tool;
            explicit_type   = true;
        } else if (const auto v6 = extract_yaml_value(t, "model"); !v6.empty()) {
            manifest.model_hint = v6;
        } else if (const auto v7 = extract_yaml_value(t, "allowed-tools"); !v7.empty()) {
            manifest.allowed_tools = split_comma(v7);
        } else if (const auto v8 = extract_yaml_value(t, "allowed_tools"); !v8.empty()) {
            // Accept underscore variant as well for cross-ecosystem compatibility.
            manifest.allowed_tools = split_comma(v8);
        }
    }

    // ── Validate required fields ─────────────────────────────────────────────

    if (manifest.name.empty()) {
        core::logging::warn("SkillLoader: '{}' missing required 'name' field",
                            manifest_path.string());
        return std::nullopt;
    }
    if (manifest.description.empty()) {
        core::logging::warn("SkillLoader: skill '{}' missing required 'description' field",
                            manifest.name);
        return std::nullopt;
    }

    // ── Auto-detect skill type ───────────────────────────────────────────────
    //
    // Explicit 'type:' frontmatter takes precedence.  Otherwise the presence
    // of 'entry_point' determines the type.

    if (!explicit_type) {
        manifest.type = manifest.entry_point.empty() ? SkillType::Prompt : SkillType::Tool;
    }

    // Tool skills require entry_point; Prompt skills do not.
    if (manifest.type == SkillType::Tool && manifest.entry_point.empty()) {
        core::logging::warn("SkillLoader: Tool skill '{}' missing required 'entry_point' field",
                            manifest.name);
        return std::nullopt;
    }

    // ── Read body for Prompt skills ──────────────────────────────────────────
    //
    // Everything remaining in the file after the closing '---' is the body.
    // We trim one leading newline to avoid an empty first line.

    if (manifest.type == SkillType::Prompt) {
        std::ostringstream body_stream;
        bool first_line = true;
        while (std::getline(file, line)) {
            if (first_line) {
                // Skip a single blank line immediately after the closing delimiter.
                if (trim(line).empty()) {
                    first_line = false;
                    continue;
                }
                first_line = false;
            }
            body_stream << line << '\n';
        }
        manifest.body = body_stream.str();
        // Trim trailing whitespace from body.
        const auto last_nonws = manifest.body.find_last_not_of(" \t\r\n");
        if (last_nonws != std::string::npos) {
            manifest.body.resize(last_nonws + 1);
        } else {
            manifest.body.clear();
        }
    }

    return manifest;
}

#ifdef FILO_ENABLE_PYTHON

int SkillLoader::load_from_directory(const fs::path& root, ToolManager& tool_manager) {
    if (!fs::exists(root) || !fs::is_directory(root)) return 0;

    int count = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) continue;

        auto maybe_manifest = parse_manifest(entry.path());
        if (!maybe_manifest) continue;

        const auto& manifest = *maybe_manifest;

        if (!manifest.enabled) {
            core::logging::info("SkillLoader: skipping disabled skill '{}'", manifest.name);
            continue;
        }

        // Prompt skills are registered as slash commands by SkillCommandLoader, not here.
        if (manifest.type == SkillType::Prompt) {
            core::logging::info("SkillLoader: skipping Prompt skill '{}' "
                                "(registered by SkillCommandLoader)", manifest.name);
            continue;
        }

        const fs::path script_path = manifest.skill_dir / manifest.entry_point;
        if (!fs::exists(script_path)) {
            core::logging::warn("SkillLoader: entry_point '{}' not found for skill '{}'",
                                script_path.string(), manifest.name);
            continue;
        }

        // Python module name is the entry_point stem (filename without extension).
        const std::string module_name = script_path.stem().string();

        try {
            auto tool = std::make_shared<PythonTool>(script_path.string(), module_name);
            tool_manager.register_tool(std::move(tool));
            core::logging::info("SkillLoader: registered Tool skill '{}'", manifest.name);
            ++count;
        } catch (const std::exception& e) {
            core::logging::error("SkillLoader: failed to load skill '{}': {}",
                                 manifest.name, e.what());
        }
    }
    return count;
}

int SkillLoader::discover_and_register(ToolManager& tool_manager) {
    int total = 0;
    for (const auto& root : default_search_paths()) {
        total += load_from_directory(root, tool_manager);
    }
    if (total > 0) {
        core::logging::info("SkillLoader: {} Tool skill(s) loaded", total);
    }
    return total;
}

#else // FILO_ENABLE_PYTHON

int SkillLoader::load_from_directory(const fs::path&, ToolManager&) { return 0; }
int SkillLoader::discover_and_register(ToolManager&)                { return 0; }

#endif // FILO_ENABLE_PYTHON

} // namespace core::tools
