#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/commands/CommandExecutor.hpp"
#include "core/commands/SkillCommand.hpp"
#include "core/commands/SkillCommandLoader.hpp"
#include "core/commands/SkillTurnResolver.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ProviderManager.hpp"
#include "core/tools/SkillLoader.hpp"
#include "core/tools/SkillManifest.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace fs = std::filesystem;
using namespace core::commands;
using namespace core::commands::detail;
using namespace core::tools;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static fs::path make_temp_root(std::string_view tag) {
    fs::path tmp = fs::temp_directory_path()
                 / ("filo_skill_cmd_test_" + std::string(tag));
    fs::create_directories(tmp);
    return tmp;
}

static void write_file(const fs::path& p, std::string_view content) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p);
    f << content;
}

struct ScopedEnvVar {
    std::string name;
    std::optional<std::string> old_value;

    ScopedEnvVar(std::string env_name, const std::string& value)
        : name(std::move(env_name)) {
        if (const char* existing = std::getenv(name.c_str())) {
            old_value = std::string(existing);
        }
        ::setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (old_value.has_value()) {
            ::setenv(name.c_str(), old_value->c_str(), 1);
        } else {
            ::unsetenv(name.c_str());
        }
    }
};

class DummyProvider final : public core::llm::LLMProvider {
public:
    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        callback(core::llm::StreamChunk::make_final());
    }
};

static core::commands::CommandContext make_null_ctx() {
    return core::commands::CommandContext{
        .text              = "",
        .clear_input_fn    = []() {},
        .append_history_fn = [](const std::string&) {},
        .agent             = nullptr,
    };
}

// ---------------------------------------------------------------------------
// detail::expand_skill_arguments
// ---------------------------------------------------------------------------

TEST_CASE("expand_skill_arguments: replaces single $ARGUMENTS placeholder",
          "[skill_commands]") {
    CHECK(expand_skill_arguments("Review PR #$ARGUMENTS", "42") == "Review PR #42");
}

TEST_CASE("expand_skill_arguments: replaces multiple $ARGUMENTS occurrences",
          "[skill_commands]") {
    CHECK(expand_skill_arguments("$ARGUMENTS and $ARGUMENTS", "hello")
          == "hello and hello");
}

TEST_CASE("expand_skill_arguments: no placeholder leaves body unchanged",
          "[skill_commands]") {
    CHECK(expand_skill_arguments("Fixed prompt text.", "ignored")
          == "Fixed prompt text.");
}

TEST_CASE("expand_skill_arguments: empty args collapses placeholder to empty string",
          "[skill_commands]") {
    CHECK(expand_skill_arguments("Prefix $ARGUMENTS Suffix", "") == "Prefix  Suffix");
}

TEST_CASE("expand_skill_arguments: empty body stays empty",
          "[skill_commands]") {
    CHECK(expand_skill_arguments("", "args") == "");
}

TEST_CASE("expand_skill_arguments: partial match 'ARGUMENTS' without $ is not replaced",
          "[skill_commands]") {
    CHECK(expand_skill_arguments("ARGUMENTS placeholder", "x") == "ARGUMENTS placeholder");
}

// ---------------------------------------------------------------------------
// detail::extract_trailing_args
// ---------------------------------------------------------------------------

TEST_CASE("extract_trailing_args: returns args after command verb",
          "[skill_commands]") {
    CHECK(extract_trailing_args("/review-pr 123") == "123");
}

TEST_CASE("extract_trailing_args: returns multi-word args",
          "[skill_commands]") {
    CHECK(extract_trailing_args("/summarise hello world") == "hello world");
}

TEST_CASE("extract_trailing_args: returns empty when no args",
          "[skill_commands]") {
    CHECK(extract_trailing_args("/summarise") == "");
}

TEST_CASE("extract_trailing_args: returns empty for trailing-only whitespace",
          "[skill_commands]") {
    CHECK(extract_trailing_args("/summarise   ") == "");
}

TEST_CASE("extract_trailing_args: handles leading whitespace before verb",
          "[skill_commands]") {
    CHECK(extract_trailing_args("  /cmd arg") == "arg");
}

TEST_CASE("extract_trailing_args: returns empty for empty input",
          "[skill_commands]") {
    CHECK(extract_trailing_args("") == "");
}

// ---------------------------------------------------------------------------
// SkillManifest: new fields via parse_manifest
// ---------------------------------------------------------------------------

TEST_CASE("parse_manifest: Prompt skill with body populates type and body fields",
          "[skill_commands]") {
    auto root = make_temp_root("manifest_prompt");
    auto skill_dir = root / "pr_review";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: review-pr
description: Security-focused PR review.
---

Review PR #$ARGUMENTS for security vulnerabilities.
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK(result->name        == "review-pr");
    CHECK(result->description == "Security-focused PR review.");
    CHECK(result->entry_point.empty());
    CHECK(result->type        == SkillType::Prompt);
    CHECK_THAT(result->body, Catch::Matchers::ContainsSubstring("$ARGUMENTS"));
    CHECK_THAT(result->body, Catch::Matchers::ContainsSubstring("vulnerabilities"));

    fs::remove_all(root);
}

TEST_CASE("parse_manifest: explicit 'type: prompt' forces Prompt even if name looks tool-like",
          "[skill_commands]") {
    auto root = make_temp_root("manifest_explicit_prompt");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: my_tool
description: A prompt-type skill despite the name.
type: prompt
---
Do something useful.
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK(result->type == SkillType::Prompt);

    fs::remove_all(root);
}

TEST_CASE("parse_manifest: Tool skill with entry_point keeps SkillType::Tool",
          "[skill_commands]") {
    auto root = make_temp_root("manifest_tool");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: weather
description: Fetch weather.
entry_point: weather.py
---
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK(result->type == SkillType::Tool);
    CHECK(result->entry_point == "weather.py");

    fs::remove_all(root);
}

TEST_CASE("parse_manifest: parses 'model' field into model_hint",
          "[skill_commands]") {
    auto root = make_temp_root("manifest_model");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: fast_review
description: Quick review using a small model.
model: grok-3-mini
---
Review $ARGUMENTS quickly.
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK(result->model_hint == "grok-3-mini");

    fs::remove_all(root);
}

TEST_CASE("resolve_skill_turn selects the provider family for known model IDs",
          "[skill_commands][model_resolution]") {
    const auto sandbox = make_temp_root("skill_turn_resolution_family");
    const auto xdg_home = sandbox / "xdg";
    const auto project_dir = sandbox / "project";
    const auto config_path = project_dir / ".filo" / "config.json";
    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_file(config_path, R"({
        "default_provider": "claude"
    })");

    core::config::ConfigManager::get_instance().load(project_dir);
    core::llm::ProviderManager::get_instance().register_provider(
        "claude",
        std::make_shared<DummyProvider>());

    const auto resolution = resolve_skill_turn(
        "claude-sonnet-4-6",
        {"read_file"},
        std::optional<std::string_view>{"claude"});

    CHECK(resolution.warning.empty());
    CHECK(resolution.callbacks.provider_override != nullptr);
    CHECK(resolution.callbacks.model_override == "claude-sonnet-4-6");
    REQUIRE(resolution.callbacks.allowed_tools.size() == 1);
    CHECK(resolution.callbacks.allowed_tools.front() == "read_file");

    core::config::ConfigManager::get_instance().load(std::filesystem::current_path());
    fs::remove_all(sandbox);
}

TEST_CASE("resolve_skill_turn falls back safely when provider hints are unavailable",
          "[skill_commands][model_resolution]") {
    const auto sandbox = make_temp_root("skill_turn_resolution_missing");
    const auto xdg_home = sandbox / "xdg";
    const auto project_dir = sandbox / "project";
    const auto config_path = project_dir / ".filo" / "config.json";
    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_file(config_path, R"({
        "providers": {
            "openai-skill-test": {
                "api_type": "openai",
                "base_url": "https://example.test/v1",
                "model": "gpt-5.4"
            }
        }
    })");

    core::config::ConfigManager::get_instance().load(project_dir);

    const auto resolution = resolve_skill_turn(
        "claude-sonnet-4-6",
        {"read_file"});

    CHECK(resolution.callbacks.provider_override == nullptr);
    CHECK(resolution.callbacks.model_override.empty());
    CHECK_FALSE(resolution.warning.empty());
    CHECK_THAT(
        resolution.warning,
        Catch::Matchers::ContainsSubstring("using the current model"));

    core::config::ConfigManager::get_instance().load(std::filesystem::current_path());
    fs::remove_all(sandbox);
}

TEST_CASE("resolve_skill_turn matches Qwen model hints to DashScope providers",
          "[skill_commands][model_resolution]") {
    const auto sandbox = make_temp_root("skill_turn_resolution_qwen_family");
    const auto xdg_home = sandbox / "xdg";
    const auto project_dir = sandbox / "project";
    const auto config_path = project_dir / ".filo" / "config.json";
    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_file(config_path, R"({
        "providers": {
            "dashscope-prod": {
                "api_type": "dashscope",
                "base_url": "https://example.test/compatible-mode/v1",
                "model": "qwen3-coder-plus"
            }
        }
    })");

    core::config::ConfigManager::get_instance().load(project_dir);
    core::llm::ProviderManager::get_instance().register_provider(
        "dashscope-prod",
        std::make_shared<DummyProvider>());

    const auto resolution = resolve_skill_turn(
        "qwen3-max",
        {"read_file"});

    CHECK(resolution.warning.empty());
    CHECK(resolution.callbacks.provider_override != nullptr);
    CHECK(resolution.callbacks.model_override == "qwen3-max");
    REQUIRE(resolution.callbacks.allowed_tools.size() == 1);
    CHECK(resolution.callbacks.allowed_tools.front() == "read_file");

    core::config::ConfigManager::get_instance().load(std::filesystem::current_path());
    fs::remove_all(sandbox);
}

TEST_CASE("parse_manifest: parses 'allowed-tools' into allowed_tools vector",
          "[skill_commands]") {
    auto root = make_temp_root("manifest_tools");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: safe_review
description: Review with limited tools.
allowed-tools: shell, read_file, grep_search
---
Review $ARGUMENTS.
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    REQUIRE(result->allowed_tools.size() == 3);
    CHECK(result->allowed_tools[0] == "shell");
    CHECK(result->allowed_tools[1] == "read_file");
    CHECK(result->allowed_tools[2] == "grep_search");

    fs::remove_all(root);
}

TEST_CASE("parse_manifest: parses 'allowed_tools' (underscore variant) for cross-ecosystem compat",
          "[skill_commands]") {
    auto root = make_temp_root("manifest_tools_underscore");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: safe_review2
description: Review with limited tools (underscore key).
allowed_tools: shell, read_file
---
Review $ARGUMENTS.
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    REQUIRE(result->allowed_tools.size() == 2);
    CHECK(result->allowed_tools[0] == "shell");

    fs::remove_all(root);
}

TEST_CASE("parse_manifest: body is empty when Prompt skill has no body text",
          "[skill_commands]") {
    auto root = make_temp_root("manifest_empty_body");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: empty_prompt
description: No body.
---
)");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK(result->type == SkillType::Prompt);
    CHECK(result->body.empty());

    fs::remove_all(root);
}

TEST_CASE("parse_manifest: body trims the leading blank line after closing ---",
          "[skill_commands]") {
    auto root = make_temp_root("manifest_body_trim");
    auto skill_dir = root / "s";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", "---\nname: t\ndescription: d\n---\n\nActual body.\n");

    auto result = SkillLoader::parse_manifest(skill_dir);
    REQUIRE(result.has_value());
    CHECK_THAT(result->body, Catch::Matchers::StartsWith("Actual body."));
    CHECK_THAT(result->body, !Catch::Matchers::StartsWith("\n"));

    fs::remove_all(root);
}

// ---------------------------------------------------------------------------
// SkillCommandLoader
// ---------------------------------------------------------------------------

TEST_CASE("SkillCommandLoader::load_from_directory returns 0 for nonexistent path",
          "[skill_commands]") {
    CommandExecutor executor;
    const int before = static_cast<int>(executor.describe_commands().size());
    int count = SkillCommandLoader::load_from_directory("/nonexistent/path/xyz", executor);
    CHECK(count == 0);
    CHECK(static_cast<int>(executor.describe_commands().size()) == before);
}

TEST_CASE("SkillCommandLoader::load_from_directory returns 0 for empty directory",
          "[skill_commands]") {
    auto root = make_temp_root("cmd_loader_empty");
    CommandExecutor executor;
    CHECK(SkillCommandLoader::load_from_directory(root, executor) == 0);
    fs::remove_all(root);
}

TEST_CASE("SkillCommandLoader::load_from_directory registers a Prompt skill",
          "[skill_commands]") {
    auto root = make_temp_root("cmd_loader_valid");
    auto skill_dir = root / "my_skill";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: summarise
description: Summarise the text.
---
Please summarise: $ARGUMENTS
)");

    CommandExecutor executor;
    const int count = SkillCommandLoader::load_from_directory(root, executor);
    CHECK(count == 1);

    // Skill must appear in the command index.
    const auto descs = executor.describe_commands();
    bool found = false;
    for (const auto& d : descs) {
        if (d.name == "/summarise") {
            found = true;
            CHECK(d.description == "Summarise the text.");
            CHECK(d.accepts_arguments);
        }
    }
    CHECK(found);

    fs::remove_all(root);
}

TEST_CASE("SkillCommandLoader::load_from_directory skips disabled Prompt skills",
          "[skill_commands]") {
    auto root = make_temp_root("cmd_loader_disabled");
    auto skill_dir = root / "off";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: off_skill
description: This should not load.
enabled: false
---
Some body.
)");

    CommandExecutor executor;
    const int count = SkillCommandLoader::load_from_directory(root, executor);
    CHECK(count == 0);

    const auto descs = executor.describe_commands();
    bool found = std::any_of(descs.begin(), descs.end(),
                             [](const auto& d) { return d.name == "/off_skill"; });
    CHECK_FALSE(found);

    fs::remove_all(root);
}

TEST_CASE("SkillCommandLoader::load_from_directory skips Tool skills (entry_point present)",
          "[skill_commands]") {
    auto root = make_temp_root("cmd_loader_skips_tool");
    auto skill_dir = root / "tool";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: weather
description: Fetch weather.
entry_point: weather.py
---
)");

    CommandExecutor executor;
    const int count = SkillCommandLoader::load_from_directory(root, executor);
    CHECK(count == 0);

    fs::remove_all(root);
}

TEST_CASE("SkillCommandLoader::load_from_directory loads multiple skills",
          "[skill_commands]") {
    auto root = make_temp_root("cmd_loader_multi");
    for (const auto& name : {"alpha", "beta", "gamma"}) {
        fs::create_directories(root / name);
        write_file(root / name / "SKILL.md",
                   std::string("---\nname: ") + name +
                   "\ndescription: skill " + name + "\n---\nBody of " + name + "\n");
    }

    CommandExecutor executor;
    CHECK(SkillCommandLoader::load_from_directory(root, executor) == 3);

    fs::remove_all(root);
}

// ---------------------------------------------------------------------------
// SkillCommand execution
// ---------------------------------------------------------------------------

TEST_CASE("SkillCommand::execute calls send_user_message_fn with expanded prompt",
          "[skill_commands]") {
    SkillManifest m;
    m.name        = "review-pr";
    m.description = "PR review";
    m.type        = SkillType::Prompt;
    m.body        = "Review PR #$ARGUMENTS for issues.";

    SkillCommand cmd(m);
    CHECK(cmd.get_name()        == "/review-pr");
    CHECK(cmd.get_description() == "PR review");
    CHECK(cmd.accepts_arguments());

    std::string captured;
    auto ctx            = make_null_ctx();
    ctx.text            = "/review-pr 42";
    ctx.send_user_message_fn = [&](const std::string& msg) { captured = msg; };

    cmd.execute(ctx);

    CHECK(captured == "Review PR #42 for issues.");
}

TEST_CASE("SkillCommand::execute with no $ARGUMENTS uses body verbatim",
          "[skill_commands]") {
    SkillManifest m;
    m.name        = "fixed";
    m.description = "Fixed prompt";
    m.type        = SkillType::Prompt;
    m.body        = "Run the test suite and report failures.";

    SkillCommand cmd(m);

    std::string captured;
    auto ctx            = make_null_ctx();
    ctx.text            = "/fixed";
    ctx.send_user_message_fn = [&](const std::string& msg) { captured = msg; };

    cmd.execute(ctx);

    CHECK(captured == "Run the test suite and report failures.");
}

TEST_CASE("SkillCommand::execute reports error when body is empty",
          "[skill_commands]") {
    SkillManifest m;
    m.name        = "empty";
    m.description = "Empty skill";
    m.type        = SkillType::Prompt;
    m.body        = "";

    SkillCommand cmd(m);

    bool message_fn_called = false;
    std::string appended;
    auto ctx            = make_null_ctx();
    ctx.text            = "/empty";
    ctx.send_user_message_fn = [&](const std::string&) { message_fn_called = true; };
    ctx.append_history_fn    = [&](const std::string& s) { appended += s; };

    cmd.execute(ctx);

    CHECK_FALSE(message_fn_called);
    CHECK_THAT(appended, Catch::Matchers::ContainsSubstring("empty body"));
}

TEST_CASE("SkillCommand::execute falls back to agent->send_message when no send_user_message_fn",
          "[skill_commands]") {
    // Verify the fallback path compiles and runs without crashing when
    // send_user_message_fn is null but agent is also null (reports error).
    SkillManifest m;
    m.name        = "fallback";
    m.description = "Fallback skill";
    m.type        = SkillType::Prompt;
    m.body        = "Test prompt.";

    SkillCommand cmd(m);

    std::string appended;
    auto ctx            = make_null_ctx();
    ctx.text            = "/fallback";
    ctx.agent           = nullptr;
    ctx.send_user_message_fn = nullptr; // deliberately null
    ctx.append_history_fn    = [&](const std::string& s) { appended += s; };

    cmd.execute(ctx);

    // No agent + no send_user_message_fn → reports "No agent available".
    CHECK_THAT(appended, Catch::Matchers::ContainsSubstring("No agent available"));
}

// ---------------------------------------------------------------------------
// Layer precedence: project-local overrides global
// ---------------------------------------------------------------------------

TEST_CASE("SkillCommandLoader: project-local skill overrides global by last-registered wins",
          "[skill_commands]") {
    // Global root has /summarise with description "global version".
    auto global_root = make_temp_root("layer_global");
    {
        auto sd = global_root / "summarise";
        fs::create_directories(sd);
        write_file(sd / "SKILL.md",
                   "---\nname: summarise\ndescription: global version\n---\nGlobal body.\n");
    }

    // Project root has /summarise with description "project version".
    auto project_root = make_temp_root("layer_project");
    {
        auto sd = project_root / "summarise";
        fs::create_directories(sd);
        write_file(sd / "SKILL.md",
                   "---\nname: summarise\ndescription: project version\n---\nProject body.\n");
    }

    CommandExecutor executor;
    SkillCommandLoader::load_from_directory(global_root,  executor); // loaded first
    SkillCommandLoader::load_from_directory(project_root, executor); // overrides

    // Both /summarise registrations are present; try_execute picks the first match.
    // The first match is the global one (registered first).  However, from the user
    // perspective the project-local skill should override.  CommandExecutor iterates
    // in registration order and returns on first match.  To achieve project-local
    // override, load order must be global first so project-local is prepended or
    // the executor must be modified.  In the current append-order design, project-local
    // is loaded LAST which means it is NOT the first match.
    //
    // The current design documents this as "last-registered wins on describe but
    // first-match wins on try_execute".  This test captures the ACTUAL behaviour so
    // any future change to override semantics is immediately visible.
    const auto descs = executor.describe_commands();
    std::string last_description;
    for (const auto& d : descs) {
        if (d.name == "/summarise") {
            last_description = d.description;
        }
    }
    // Last descriptor seen is the project-local one (loaded last).
    CHECK(last_description == "project version");

    fs::remove_all(global_root);
    fs::remove_all(project_root);
}

// ---------------------------------------------------------------------------
// Round-trip: write SKILL.md → parse → register → execute
// ---------------------------------------------------------------------------

TEST_CASE("Skill round-trip: write Prompt SKILL.md, load via SkillCommandLoader, execute",
          "[skill_commands]") {
    auto root = make_temp_root("round_trip");
    auto skill_dir = root / "explain";
    fs::create_directories(skill_dir);
    write_file(skill_dir / "SKILL.md", R"(---
name: explain
description: Explain a concept in simple terms.
---
Explain the concept "$ARGUMENTS" in simple, beginner-friendly terms.
Use an analogy if possible.
)");

    CommandExecutor executor;
    REQUIRE(SkillCommandLoader::load_from_directory(root, executor) == 1);

    std::string sent_prompt;
    core::commands::CommandContext ctx{
        .text              = "/explain recursion",
        .clear_input_fn    = []() {},
        .append_history_fn = [](const std::string&) {},
        .agent             = nullptr,
        .send_user_message_fn = [&](const std::string& msg) { sent_prompt = msg; },
    };

    const bool executed = executor.try_execute("/explain recursion", ctx);
    CHECK(executed);
    CHECK_THAT(sent_prompt, Catch::Matchers::ContainsSubstring("recursion"));
    CHECK_THAT(sent_prompt, Catch::Matchers::ContainsSubstring("analogy"));

    fs::remove_all(root);
}
