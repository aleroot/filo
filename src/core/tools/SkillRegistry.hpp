#pragma once

#include "SkillManifest.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools {

enum class SkillScope {
    User,
    Project,
};

struct SkillSearchRoot {
    std::filesystem::path path;
    SkillScope scope = SkillScope::Project;
    std::string label;
};

class SkillRegistry {
public:
    [[nodiscard]] static std::vector<SkillSearchRoot>
    default_search_roots(const std::filesystem::path& project_root = std::filesystem::current_path());

    [[nodiscard]] static std::vector<std::filesystem::path>
    default_search_paths(const std::filesystem::path& project_root = std::filesystem::current_path());

    [[nodiscard]] static std::optional<SkillManifest>
    parse_manifest(const std::filesystem::path& skill_dir);

    [[nodiscard]] static std::vector<SkillManifest>
    discover_all(const std::filesystem::path& project_root = std::filesystem::current_path());

    [[nodiscard]] static std::vector<SkillManifest>
    discover_instruction_skills(
        const std::filesystem::path& project_root = std::filesystem::current_path());

    [[nodiscard]] static std::optional<SkillManifest>
    find_instruction_skill(
        std::string_view name,
        const std::filesystem::path& project_root = std::filesystem::current_path());

    [[nodiscard]] static std::string
    build_catalog_prompt(
        const std::filesystem::path& project_root = std::filesystem::current_path());

    [[nodiscard]] static std::vector<std::filesystem::path>
    list_relative_resources(const SkillManifest& skill, std::size_t max_entries = 200);
};

} // namespace core::tools
