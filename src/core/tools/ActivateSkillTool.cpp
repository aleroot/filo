#include "ActivateSkillTool.hpp"
#include "SkillRegistry.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <simdjson.h>
#include <sstream>
#include <string>
#include <string_view>

namespace fs = std::filesystem;

namespace core::tools {

namespace {

constexpr std::size_t kMaxSkillContentBytes = 256 * 1024;
constexpr std::size_t kMaxResourceBytes = 512 * 1024;

[[nodiscard]] std::string read_limited(const fs::path& path,
                                       std::size_t limit,
                                       bool& truncated) {
    truncated = false;
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};

    std::string content;
    content.resize(limit + 1);
    input.read(content.data(), static_cast<std::streamsize>(content.size()));
    const auto read_count = input.gcount();
    if (read_count < 0) return {};
    content.resize(static_cast<std::size_t>(read_count));
    if (content.size() > limit) {
        content.resize(limit);
        truncated = true;
    }
    return content;
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

[[nodiscard]] std::string build_input_schema() {
    const auto skills = SkillRegistry::discover_instruction_skills();
    core::utils::JsonWriter writer(2048);
    {
        auto _root = writer.object();
        writer.kv_str("type", "object").comma();
        writer.key("properties");
        {
            auto _properties = writer.object();
            writer.key("name");
            {
                auto _name = writer.object();
                writer.kv_str("type", "string").comma()
                      .kv_str("description", "Name of the Agent Skill to activate.");
                if (!skills.empty()) {
                    writer.comma().key("enum");
                    {
                        auto _enum = writer.array();
                        bool first = true;
                        for (const auto& skill : skills) {
                            if (!first) writer.comma();
                            first = false;
                            writer.str(skill.name);
                        }
                    }
                }
            }
            writer.comma().key("resource_path");
            {
                auto _resource = writer.object();
                writer.kv_str("type", "string").comma()
                      .kv_str("description",
                              "Optional relative path to a listed skill resource to read, "
                              "for example references/REFERENCE.md or scripts/check.py.");
            }
        }
        writer.comma().key("required");
        {
            auto _required = writer.array();
            writer.str("name");
        }
        writer.comma().kv_bool("additionalProperties", false);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string format_skill_content(
    const SkillManifest& skill,
    std::string_view body,
    bool truncated) {
    std::ostringstream out;
    out << "<skill_content name=\"" << xml_escape(skill.name) << "\">\n";
    if (!skill.compatibility.empty()) {
        out << "Compatibility: " << skill.compatibility << "\n\n";
    }
    out << body;
    if (truncated) {
        out << "\n\n[Skill content truncated after "
            << kMaxSkillContentBytes
            << " bytes.]";
    }
    out << "\n\nSkill directory: " << skill.skill_dir.string() << "\n";
    out << "Relative paths in this skill are relative to the skill directory.\n";

    const auto resources = SkillRegistry::list_relative_resources(skill);
    if (!resources.empty()) {
        out << "<skill_resources>\n";
        for (const auto& resource : resources) {
            out << "  <file>" << resource.generic_string() << "</file>\n";
        }
        out << "</skill_resources>\n";
    }
    out << "</skill_content>";
    return out.str();
}

} // namespace

ToolDefinition ActivateSkillTool::get_definition() const {
    return {
        .name = std::string(names::kActivateSkill),
        .title = "Activate Agent Skill",
        .description =
            "Loads an Agent Skill's full instructions on demand. "
            "Call this when the user's task matches one of the available Agent Skills. "
            "If activated instructions reference a listed relative resource, call this tool "
            "again with the same name and resource_path to read that resource.",
        .input_schema = build_input_schema(),
        .output_schema =
            R"({"type":"object","properties":{"content":{"type":"string"},"skill_directory":{"type":"string"},"resource_path":{"type":"string"},"truncated":{"type":"boolean"},"resources":{"type":"array","items":{"type":"string"}}},"required":["content","skill_directory"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint = true,
            .idempotent_hint = true,
        },
    };
}

std::string ActivateSkillTool::execute(
    const std::string& json_args,
    const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return "{\"error\":\"Invalid JSON arguments provided to activate_skill.\"}";
    }

    if (const auto validation_error =
            detail::validate_object_arguments(doc, names::kActivateSkill, {"name", "resource_path"})) {
        return *validation_error;
    }

    std::string_view name;
    if (doc["name"].get(name) != simdjson::SUCCESS || name.empty()) {
        return "{\"error\":\"Missing or invalid 'name' argument.\"}";
    }

    const auto project_root = context.workspace_view().primary().empty()
        ? fs::current_path()
        : context.workspace_view().primary();
    const auto skill = SkillRegistry::find_instruction_skill(name, project_root);
    if (!skill.has_value()) {
        return std::format(
            "{{\"error\":\"Unknown or unavailable Agent Skill: {}\"}}",
            core::utils::escape_json_string(name));
    }

    std::string_view resource_path_raw;
    const bool wants_resource = doc["resource_path"].get(resource_path_raw) == simdjson::SUCCESS
        && !resource_path_raw.empty();

    core::utils::JsonWriter writer(4096);
    {
        auto _root = writer.object();
        writer.kv_str("skill_directory", skill->skill_dir.string()).comma();

        if (wants_resource) {
            const fs::path relative(resource_path_raw);
            if (!safe_relative_resource(relative)) {
                return "{\"error\":\"resource_path must be a safe relative path inside the skill directory.\"}";
            }

            std::error_code ec;
            const fs::path target = fs::weakly_canonical(skill->skill_dir / relative, ec);
            if (ec) {
                return std::format(
                    "{{\"error\":\"Skill resource not found or outside skill directory: {}\"}}",
                    core::utils::escape_json_string(resource_path_raw));
            }
            const fs::path base = fs::weakly_canonical(skill->skill_dir, ec);
            if (ec) {
                return "{\"error\":\"Skill directory is unavailable.\"}";
            }
            const auto checked_relative = fs::relative(target, base, ec);
            if (ec || !safe_relative_resource(checked_relative)
                || !fs::is_regular_file(target, ec)) {
                return std::format(
                    "{{\"error\":\"Skill resource not found or outside skill directory: {}\"}}",
                    core::utils::escape_json_string(resource_path_raw));
            }

            bool truncated = false;
            const auto content = read_limited(target, kMaxResourceBytes, truncated);
            writer.kv_str("resource_path", relative.generic_string()).comma()
                  .kv_str("content", content).comma()
                  .kv_bool("truncated", truncated).comma()
                  .key("resources");
        } else {
            bool truncated = false;
            const auto body = read_limited(skill->manifest_path, kMaxSkillContentBytes, truncated);
            writer.kv_str("content", format_skill_content(*skill, body, truncated)).comma()
                  .kv_bool("truncated", truncated).comma()
                  .key("resources");
        }

        {
            auto _resources = writer.array();
            bool first = true;
            for (const auto& resource : SkillRegistry::list_relative_resources(*skill)) {
                if (!first) writer.comma();
                first = false;
                writer.str(resource.generic_string());
            }
        }
    }
    return std::move(writer).take();
}

} // namespace core::tools
