#include "SkillRegistry.hpp"
#include "../landrun/LandrunSettings.hpp"
#include "../logging/Logger.hpp"
#include "../utils/AsciiUtils.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace fs = std::filesystem;

namespace core::tools {

namespace {

[[nodiscard]] std::string trim(std::string_view sv) {
    const auto start = sv.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    const auto end = sv.find_last_not_of(" \t\r\n");
    return std::string(sv.substr(start, end - start + 1));
}

[[nodiscard]] std::string unquote(std::string value) {
    value = trim(value);
    if (value.size() >= 2
        && ((value.front() == '"' && value.back() == '"')
            || (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

[[nodiscard]] std::string strip_inline_comment(std::string_view value) {
    bool in_single = false;
    bool in_double = false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const char ch = value[i];
        if (ch == '\'' && !in_double) {
            in_single = !in_single;
            continue;
        }
        if (ch == '"' && !in_single) {
            in_double = !in_double;
            continue;
        }
        if (ch == '#' && !in_single && !in_double
            && (i == 0 || value[i - 1] == ' ' || value[i - 1] == '\t')) {
            return trim(value.substr(0, i));
        }
    }
    return trim(value);
}

[[nodiscard]] std::vector<std::string> split_tool_list(std::string_view sv) {
    std::vector<std::string> result;
    const bool comma_separated = sv.find(',') != std::string_view::npos;
    std::size_t start = 0;
    while (start < sv.size()) {
        const auto end = comma_separated
            ? sv.find(',', start)
            : sv.find_first_of(" \t\r\n", start);
        const auto token_end = end == std::string_view::npos ? sv.size() : end;
        const auto token = trim(sv.substr(start, token_end - start));
        if (!token.empty()) result.push_back(token);
        if (end == std::string_view::npos) break;
        start = end + 1;
        while (!comma_separated && start < sv.size()
               && (sv[start] == ' ' || sv[start] == '\t' || sv[start] == '\r' || sv[start] == '\n')) {
            ++start;
        }
    }
    return result;
}

[[nodiscard]] bool boolean_enabled(std::string value) {
    value = trim(value);
    std::ranges::transform(value, value.begin(), core::utils::ascii::to_lower);
    return value != "false" && value != "0" && value != "no";
}

[[nodiscard]] std::optional<std::string> read_text_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

[[nodiscard]] std::vector<std::string> split_lines(std::string_view text) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find('\n', start);
        auto line = end == std::string_view::npos
            ? text.substr(start)
            : text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        lines.emplace_back(line);
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return lines;
}

struct ParsedFrontmatter {
    std::unordered_map<std::string, std::string> fields;
    std::string body;
};

[[nodiscard]] std::optional<ParsedFrontmatter>
parse_frontmatter(std::string_view content, const fs::path& manifest_path) {
    const auto lines = split_lines(content);
    std::size_t open = 0;
    while (open < lines.size() && trim(lines[open]).empty()) {
        ++open;
    }
    if (open >= lines.size() || trim(lines[open]) != "---") {
        core::logging::warn("SkillRegistry: no frontmatter delimiter in '{}'",
                            manifest_path.string());
        return std::nullopt;
    }

    std::size_t close = open + 1;
    while (close < lines.size() && trim(lines[close]) != "---") {
        ++close;
    }
    if (close >= lines.size()) {
        core::logging::warn("SkillRegistry: missing closing frontmatter delimiter in '{}'",
                            manifest_path.string());
        return std::nullopt;
    }

    ParsedFrontmatter parsed;
    for (std::size_t i = open + 1; i < close; ++i) {
        const std::string line = trim(lines[i]);
        if (line.empty() || line.front() == '#') continue;

        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;

        const std::string key = trim(std::string_view(line).substr(0, colon));
        std::string value = strip_inline_comment(std::string_view(line).substr(colon + 1));

        if ((value == "|" || value == ">") && i + 1 < close) {
            std::ostringstream block;
            std::size_t base_indent = 0;
            for (std::size_t probe = i + 1; probe < close; ++probe) {
                if (trim(lines[probe]).empty()) continue;
                base_indent = lines[probe].find_first_not_of(" \t");
                break;
            }
            while (i + 1 < close) {
                const auto& next = lines[i + 1];
                const std::size_t indent = next.find_first_not_of(" \t") == std::string::npos
                    ? next.size()
                    : next.find_first_not_of(" \t");
                if (!trim(next).empty() && (base_indent == 0 || indent < base_indent)) break;
                std::string_view piece(next);
                if (piece.size() >= base_indent) piece.remove_prefix(base_indent);
                block << piece << '\n';
                ++i;
            }
            value = block.str();
            if (value.ends_with('\n')) value.pop_back();
        }

        if (!key.empty()) {
            parsed.fields[key] = unquote(std::move(value));
        }
    }

    std::ostringstream body;
    bool first_body_line = true;
    for (std::size_t i = close + 1; i < lines.size(); ++i) {
        if (first_body_line && trim(lines[i]).empty()) {
            first_body_line = false;
            continue;
        }
        first_body_line = false;
        body << lines[i];
        if (i + 1 < lines.size()) body << '\n';
    }
    parsed.body = body.str();
    const auto last_nonws = parsed.body.find_last_not_of(" \t\r\n");
    if (last_nonws == std::string::npos) {
        parsed.body.clear();
    } else {
        parsed.body.resize(last_nonws + 1);
    }
    return parsed;
}

[[nodiscard]] std::optional<std::string>
field(const ParsedFrontmatter& parsed, std::string_view key) {
    const auto it = parsed.fields.find(std::string(key));
    if (it == parsed.fields.end()) return std::nullopt;
    return it->second;
}

[[nodiscard]] bool is_resource_root(std::string_view name) {
    return name == "scripts" || name == "references" || name == "assets";
}

[[nodiscard]] bool safe_relative_resource(const fs::path& path) {
    if (path.empty() || path.is_absolute()) return false;
    for (const auto& part : path) {
        if (part == "..") return false;
    }
    return true;
}

[[nodiscard]] std::string xml_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '&':
            escaped += "&amp;";
            break;
        case '<':
            escaped += "&lt;";
            break;
        case '>':
            escaped += "&gt;";
            break;
        case '"':
            escaped += "&quot;";
            break;
        case '\'':
            escaped += "&apos;";
            break;
        default:
            escaped += ch;
            break;
        }
    }
    return escaped;
}

} // namespace

std::vector<SkillSearchRoot>
SkillRegistry::default_search_roots(const fs::path& project_root) {
    std::vector<SkillSearchRoot> roots;
    const bool include_user_roots = !core::landrun::LandrunSettings::instance().enabled();
    if (include_user_roots) {
        if (const char* home = std::getenv("HOME")) {
            const fs::path home_path(home);
            roots.push_back({
                .path = home_path / ".claude" / "skills",
                .scope = SkillScope::User,
                .label = "user-claude",
            });
            roots.push_back({
                .path = home_path / ".agents" / "skills",
                .scope = SkillScope::User,
                .label = "user-agent",
            });
        }
    }

    roots.push_back({
        .path = project_root / ".claude" / "skills",
        .scope = SkillScope::Project,
        .label = "project-claude",
    });
    roots.push_back({
        .path = project_root / ".agents" / "skills",
        .scope = SkillScope::Project,
        .label = "project-agent",
    });
    if (include_user_roots) {
        if (const char* home = std::getenv("HOME")) {
            const fs::path home_path(home);
            roots.push_back({
                .path = home_path / ".config" / "filo" / "skills",
                .scope = SkillScope::User,
                .label = "user-filo",
            });
        }
    }
    roots.push_back({
        .path = project_root / ".filo" / "skills",
        .scope = SkillScope::Project,
        .label = "project-filo",
    });
    return roots;
}

std::vector<fs::path>
SkillRegistry::default_search_paths(const fs::path& project_root) {
    std::vector<fs::path> paths;
    for (const auto& root : default_search_roots(project_root)) {
        paths.push_back(root.path);
    }
    return paths;
}

std::optional<SkillManifest>
SkillRegistry::parse_manifest(const fs::path& skill_dir) {
    const fs::path manifest_path = skill_dir / "SKILL.md";
    if (!fs::exists(manifest_path)) return std::nullopt;

    const auto content = read_text_file(manifest_path);
    if (!content.has_value()) {
        core::logging::warn("SkillRegistry: cannot open '{}'", manifest_path.string());
        return std::nullopt;
    }

    const auto parsed = parse_frontmatter(*content, manifest_path);
    if (!parsed.has_value()) return std::nullopt;

    SkillManifest manifest;
    manifest.enabled = true;
    manifest.skill_dir = fs::exists(skill_dir) ? fs::weakly_canonical(skill_dir) : skill_dir;
    manifest.manifest_path = fs::exists(manifest_path)
        ? fs::weakly_canonical(manifest_path)
        : manifest_path;

    if (const auto value = field(*parsed, "name")) manifest.name = *value;
    if (const auto value = field(*parsed, "description")) manifest.description = *value;
    if (const auto value = field(*parsed, "entry_point")) manifest.entry_point = *value;
    if (const auto value = field(*parsed, "entry-point")) manifest.entry_point = *value;
    if (const auto value = field(*parsed, "enabled")) manifest.enabled = boolean_enabled(*value);
    if (const auto value = field(*parsed, "user_invocable")) {
        manifest.user_invocable = boolean_enabled(*value);
    } else if (const auto value2 = field(*parsed, "user-invocable")) {
        manifest.user_invocable = boolean_enabled(*value2);
    }
    if (const auto value = field(*parsed, "model")) manifest.model_hint = *value;
    if (const auto value = field(*parsed, "license")) manifest.license = *value;
    if (const auto value = field(*parsed, "compatibility")) manifest.compatibility = *value;

    bool explicit_type = false;
    if (const auto value = field(*parsed, "type")) {
        auto type = trim(*value);
        std::ranges::transform(type, type.begin(), core::utils::ascii::to_lower);
        manifest.type = type == "prompt" || type == "instruction" || type == "agent"
            ? SkillType::Prompt
            : SkillType::Tool;
        explicit_type = true;
    }

    if (const auto value = field(*parsed, "allowed-tools")) {
        manifest.allowed_tools = split_tool_list(*value);
    } else if (const auto value2 = field(*parsed, "allowed_tools")) {
        manifest.allowed_tools = split_tool_list(*value2);
    }

    if (manifest.name.empty()) {
        core::logging::warn("SkillRegistry: '{}' missing required 'name' field",
                            manifest_path.string());
        return std::nullopt;
    }
    if (manifest.description.empty()) {
        core::logging::warn("SkillRegistry: skill '{}' missing required 'description' field",
                            manifest.name);
        return std::nullopt;
    }

    if (!explicit_type) {
        manifest.type = manifest.entry_point.empty() ? SkillType::Prompt : SkillType::Tool;
    }
    if (manifest.type == SkillType::Tool && manifest.entry_point.empty()) {
        core::logging::warn("SkillRegistry: Tool skill '{}' missing required 'entry_point' field",
                            manifest.name);
        return std::nullopt;
    }

    if (manifest.type == SkillType::Prompt) {
        manifest.body = parsed->body;
    }
    return manifest;
}

std::vector<SkillManifest>
SkillRegistry::discover_all(const fs::path& project_root) {
    std::vector<SkillManifest> ordered;
    std::unordered_map<std::string, std::size_t> by_name;

    for (const auto& root : default_search_roots(project_root)) {
        std::error_code ec;
        if (!fs::is_directory(root.path, ec)) continue;

        for (const auto& entry : fs::directory_iterator(root.path, ec)) {
            if (ec) break;
            if (!entry.is_directory(ec)) continue;

            auto manifest = parse_manifest(entry.path());
            if (!manifest.has_value() || !manifest->enabled) continue;

            if (auto it = by_name.find(manifest->name); it != by_name.end()) {
                ordered[it->second] = std::move(*manifest);
            } else {
                by_name[manifest->name] = ordered.size();
                ordered.push_back(std::move(*manifest));
            }
        }
    }

    return ordered;
}

std::vector<SkillManifest>
SkillRegistry::discover_instruction_skills(const fs::path& project_root) {
    auto skills = discover_all(project_root);
    std::erase_if(skills, [](const SkillManifest& skill) {
        return !skill.is_agent_instruction_skill();
    });
    std::ranges::sort(skills, {}, &SkillManifest::name);
    return skills;
}

std::optional<SkillManifest>
SkillRegistry::find_instruction_skill(std::string_view name, const fs::path& project_root) {
    for (auto& skill : discover_instruction_skills(project_root)) {
        if (skill.name == name) return skill;
    }
    return std::nullopt;
}

std::string SkillRegistry::build_catalog_prompt(const fs::path& project_root) {
    const auto skills = discover_instruction_skills(project_root);
    if (skills.empty()) return {};

    std::ostringstream out;
    out << "\n\n[Agent Skills]\n";
    out << "The following Agent Skills provide specialized instructions. "
           "When a task matches a skill description, call `activate_skill` with the skill name "
           "before proceeding. If activated instructions reference relative files, call "
           "`activate_skill` again with `resource_path` set to that relative path.\n";
    out << "<available_skills>\n";
    for (const auto& skill : skills) {
        out << "  <skill>\n"
            << "    <name>" << xml_escape(skill.name) << "</name>\n"
            << "    <description>" << xml_escape(skill.description) << "</description>\n"
            << "  </skill>\n";
    }
    out << "</available_skills>";
    return out.str();
}

std::vector<fs::path>
SkillRegistry::list_relative_resources(const SkillManifest& skill, std::size_t max_entries) {
    std::vector<fs::path> resources;
    std::error_code ec;
    if (!fs::is_directory(skill.skill_dir, ec)) return resources;

    for (const auto& root : fs::directory_iterator(skill.skill_dir, ec)) {
        if (ec) break;
        if (!root.is_directory(ec) || !is_resource_root(root.path().filename().string())) {
            continue;
        }
        for (const auto& entry : fs::recursive_directory_iterator(root.path(), ec)) {
            if (ec) break;
            if (!entry.is_regular_file(ec)) continue;
            auto relative = fs::relative(entry.path(), skill.skill_dir, ec);
            if (ec || !safe_relative_resource(relative)) continue;
            resources.push_back(relative);
            if (resources.size() >= max_entries) return resources;
        }
    }
    std::ranges::sort(resources);
    return resources;
}

} // namespace core::tools
