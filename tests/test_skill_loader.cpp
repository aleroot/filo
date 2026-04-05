#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/tools/SkillLoader.hpp"
#include "core/tools/ToolManager.hpp"
#include "TestSessionContext.hpp"

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <string>

namespace fs = std::filesystem;
using namespace core::tools;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static fs::path make_temp_skill_root(std::string_view tag) {
    fs::path tmp = fs::temp_directory_path()
                 / ("filo_skill_test_" + std::string(tag));
    fs::create_directories(tmp);
    return tmp;
}

static void write_file(const fs::path& p, std::string_view content) {
    std::ofstream f(p);
    f << content;
}

// ---------------------------------------------------------------------------
// parse_manifest — valid inputs
// ---------------------------------------------------------------------------

TEST_CASE("SkillLoader::parse_manifest returns manifest for valid SKILL.md",
          "[skill_loader]") {
    auto root = make_temp_skill_root("valid");
    auto skill_dir = root / "my_skill";
    fs::create_directories(skill_dir);

    write_file(skill_dir / "SKILL.md", R"(---
name: my_tool
description: Does something useful
entry_point: skill.py
enabled: true
---

## Full docs ignored by parser
)");

    auto result = SkillLoader::parse_manifest(skill_dir);

    REQUIRE(result.has_value());
    CHECK(result->name        == "my_tool");
    CHECK(result->description == "Does something useful");
    CHECK(result->entry_point == "skill.py");
    CHECK(result->enabled     == true);

    fs::remove_all(root);
}

TEST_CASE("SkillLoader::parse_manifest defaults enabled to true when field absent",
          "[skill_loader]") {
    auto root = make_temp_skill_root("enabled_default");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);

    write_file(skill_dir / "SKILL.md", R"(---
name: no_enabled_field
description: No enabled key here
entry_point: tool.py
---
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK(result->enabled == true);

    fs::remove_all(root);
}

TEST_CASE("SkillLoader::parse_manifest parses enabled: false", "[skill_loader]") {
    auto root = make_temp_skill_root("disabled");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);

    write_file(skill_dir / "SKILL.md", R"(---
name: disabled_tool
description: This skill is disabled
entry_point: skill.py
enabled: false
---
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK(result->enabled == false);

    fs::remove_all(root);
}

TEST_CASE("SkillLoader::parse_manifest treats '0' and 'no' as disabled",
          "[skill_loader]") {
    auto root = make_temp_skill_root("disabled_variants");

    for (const auto& [tag, value] : std::vector<std::pair<std::string, std::string>>{
             {"zero", "0"}, {"no", "no"}}) {
        auto skill_dir = root / tag;
        fs::create_directories(skill_dir);
        write_file(skill_dir / "SKILL.md",
                   "---\nname: t\ndescription: d\nentry_point: t.py\nenabled: " + value + "\n---\n");
        auto result = SkillLoader::parse_manifest(skill_dir);
        REQUIRE(result.has_value());
        CHECK(result->enabled == false);
    }

    fs::remove_all(root);
}

// ---------------------------------------------------------------------------
// parse_manifest — invalid / missing inputs
// ---------------------------------------------------------------------------

TEST_CASE("SkillLoader::parse_manifest returns nullopt when SKILL.md absent",
          "[skill_loader]") {
    auto root = make_temp_skill_root("no_md");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);

    CHECK(!SkillLoader::parse_manifest(skill_dir).has_value());

    fs::remove_all(root);
}

TEST_CASE("SkillLoader::parse_manifest returns nullopt when no frontmatter delimiter",
          "[skill_loader]") {
    auto root = make_temp_skill_root("no_delim");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", "Just plain text, no YAML frontmatter\n");

    CHECK(!SkillLoader::parse_manifest(skill_dir).has_value());

    fs::remove_all(root);
}

TEST_CASE("SkillLoader::parse_manifest returns nullopt when 'name' is missing",
          "[skill_loader]") {
    auto root = make_temp_skill_root("no_name");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md",
               "---\ndescription: Missing name\nentry_point: s.py\n---\n");

    CHECK(!SkillLoader::parse_manifest(skill_dir).has_value());

    fs::remove_all(root);
}

TEST_CASE("SkillLoader::parse_manifest returns nullopt when 'description' is missing",
          "[skill_loader]") {
    auto root = make_temp_skill_root("no_desc");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md",
               "---\nname: t\nentry_point: s.py\n---\n");

    CHECK(!SkillLoader::parse_manifest(skill_dir).has_value());

    fs::remove_all(root);
}

TEST_CASE("SkillLoader::parse_manifest with no 'entry_point' yields a Prompt skill",
          "[skill_loader]") {
    // A SKILL.md without entry_point is valid — it becomes a SkillType::Prompt
    // skill registered by SkillCommandLoader, not a Python Tool skill.
    auto root = make_temp_skill_root("no_ep");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md",
               "---\nname: t\ndescription: d\n---\n");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK(result->entry_point.empty());
    CHECK(result->type == SkillType::Prompt);

    fs::remove_all(root);
}

// ---------------------------------------------------------------------------
// load_from_directory — directory-level behaviour
// ---------------------------------------------------------------------------

TEST_CASE("SkillLoader::load_from_directory returns 0 for nonexistent path",
          "[skill_loader]") {
    ToolManager& tm = ToolManager::get_instance();
    int count = SkillLoader::load_from_directory("/nonexistent/path/xyz", tm);
    CHECK(count == 0);
}

TEST_CASE("SkillLoader::load_from_directory returns 0 for empty directory",
          "[skill_loader]") {
    auto root = make_temp_skill_root("empty_root");
    ToolManager& tm = ToolManager::get_instance();
    int count = SkillLoader::load_from_directory(root, tm);
    CHECK(count == 0);
    fs::remove_all(root);
}

TEST_CASE("SkillLoader::load_from_directory skips subdirs without SKILL.md",
          "[skill_loader]") {
    auto root = make_temp_skill_root("no_manifest_subdir");
    fs::create_directories(root / "not_a_skill");  // directory but no SKILL.md

    ToolManager& tm = ToolManager::get_instance();
    int count = SkillLoader::load_from_directory(root, tm);
    CHECK(count == 0);

    fs::remove_all(root);
}

TEST_CASE("SkillLoader::load_from_directory skips skill when entry_point file absent",
          "[skill_loader]") {
    auto root = make_temp_skill_root("missing_ep_file");
    auto skill_dir = root / "ghost";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md",
               "---\nname: ghost\ndescription: d\nentry_point: ghost.py\n---\n");
    // ghost.py is deliberately NOT created

    ToolManager& tm = ToolManager::get_instance();
    int count = SkillLoader::load_from_directory(root, tm);
    CHECK(count == 0);

    fs::remove_all(root);
}

// ---------------------------------------------------------------------------
// Python integration (requires FILO_ENABLE_PYTHON)
// ---------------------------------------------------------------------------

#ifdef FILO_ENABLE_PYTHON
TEST_CASE("SkillLoader registers and executes a minimal Python skill",
          "[skill_loader][python]") {
    auto root = make_temp_skill_root("py_skill");
    auto skill_dir = root / "echo_skill";
    fs::create_directories(skill_dir);

    write_file(skill_dir / "SKILL.md", R"(---
name: echo_skill
description: Returns the input string prefixed with 'echo:'.
entry_point: echoskill.py
enabled: true
---
)");

    write_file(skill_dir / "echoskill.py", R"(
import json

def get_schema():
    return {
        "name": "echo_skill",
        "description": "Returns the input string prefixed with 'echo:'.",
        "parameters": [
            {
                "name": "input",
                "type": "string",
                "description": "Text to echo.",
                "required": True,
            }
        ],
    }

def execute(json_args):
    args = json.loads(json_args)
    return json.dumps({"output": "echo:" + args.get("input", "")})
)");

    ToolManager& tm = ToolManager::get_instance();
    int count = SkillLoader::load_from_directory(root, tm);
    REQUIRE(count == 1);
    REQUIRE(tm.has_tool("echo_skill"));

    const std::string result = tm.execute_tool(
        "echo_skill",
        R"({"input": "hello"})",
        test_support::make_workspace_session_context());
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("echo:hello"));

    fs::remove_all(root);
}

TEST_CASE("SkillLoader skips disabled skill (Python)", "[skill_loader][python]") {
    auto root = make_temp_skill_root("disabled_py");
    auto skill_dir = root / "off_skill";
    fs::create_directories(skill_dir);

    write_file(skill_dir / "SKILL.md", R"(---
name: off_skill
description: This skill should be skipped.
entry_point: off.py
enabled: false
---
)");
    write_file(skill_dir / "off.py", R"(
def get_schema():
    return {"name": "off_skill", "description": "d", "parameters": []}

def execute(json_args):
    return '{"output": "should never run"}'
)");

    ToolManager& tm = ToolManager::get_instance();
    int count = SkillLoader::load_from_directory(root, tm);
    CHECK(count == 0);

    fs::remove_all(root);
}
#endif // FILO_ENABLE_PYTHON
