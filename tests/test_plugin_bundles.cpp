#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/AgentDefinitionRegistry.hpp"
#include "core/agent/DelegatedAgentRunner.hpp"
#include "core/agent/SubagentOrchestrator.hpp"
#include "core/commands/SkillCommand.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/tools/ActivateSkillTool.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/Frontmatter.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "core/tools/ReadTool.hpp"
#include "core/tools/SkillRegistry.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/workspace/SessionWorkspace.hpp"
#include "TestEnvironment.hpp"
#include "TestSessionContext.hpp"

#include <simdjson.h>

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Sandbox {
    fs::path root;
    explicit Sandbox(std::string_view tag)
        : root(fs::weakly_canonical(fs::temp_directory_path())
               / ("filo_plugin_bundle_" + std::string(tag))) {
        fs::remove_all(root);
        fs::create_directories(root);
    }
    ~Sandbox() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

void write_file(const fs::path& path, std::string_view content) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << content;
}

using test_support::ScopedEnvVar;

/// A plugin shaped like google/elephant-goldfish-model: skills that read the
/// bundle's personas and standards, and personas in the portable `agents/`
/// layout. The Claude-host adapter is there to prove Filo never reads it.
void write_plugin(const fs::path& plugin) {
    write_file(plugin / "plugin.json", R"({"name":"egm","version":"0.1.0"})");
    write_file(plugin / "skills" / "egm-review" / "SKILL.md", R"(---
name: egm-review
license: Apache-2.0
disable-model-invocation: true
description: >-
  Orchestrates a multi-stage review
  with fresh-context reviewers.
---

Spawn the `egm-zoom-out` subagent defined in `agents/zoom_out.md`.
)");
    write_file(plugin / "agents" / "zoom_out.md", R"(---
name: egm-zoom-out
description: Skeptical Architect persona focused on simplicity.
license: Apache-2.0
---

# Core Mandates

You are a strictly read-only Goldfish reviewer. Apply resources/standard.md.
)");
    write_file(plugin / "agents" / "synthesizer.md", R"(---
name: egm-synthesizer
description: Coordinates reviewers and compiles the report.
tools:
  - task
  - read
  - write_file
max-steps: 30
read-only: false
isolated: false
model: some-host-model
---

Spawn reviewers and write goldfish_review_report.md.
)");
    write_file(plugin / ".claude-plugin" / "plugin.json", R"({"name":"egm"})");
    write_file(plugin / ".claude-plugin" / "agents" / "zoom_out.md", R"(---
name: egm-zoom-out
description: Claude adapter.
tools: [Read, Grep, Glob]
---
ADAPTER-BODY
)");
    write_file(plugin / "resources" / "standard.md", "Prefer deletion over addition.\n");
}

class RecordingProvider final : public core::llm::LLMProvider {
public:
    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }
        callback(core::llm::StreamChunk::make_content("reviewed"));
        callback(core::llm::StreamChunk::make_final());
    }
    std::vector<core::llm::ChatRequest> requests() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
};

[[nodiscard]] std::string system_text(const core::llm::ChatRequest& request) {
    std::string text = request.prompt_plan.render();
    for (const auto& message : request.messages) {
        if (message.role == "system") text += message.content;
    }
    return text;
}

namespace fm = core::tools::frontmatter;

[[nodiscard]] fm::Document parse_ok(std::string_view text) {
    auto parsed = fm::parse(text);
    if (!parsed) FAIL("line " << parsed.error().line << ": " << parsed.error().message);
    return std::move(*parsed);
}

[[nodiscard]] fm::ParseError parse_error(std::string_view text) {
    auto parsed = fm::parse(text);
    REQUIRE_FALSE(parsed.has_value());
    return parsed.error();
}

} // namespace

TEST_CASE("Frontmatter block scalars follow YAML chomping and folding", "[frontmatter]") {
    const auto doc = parse_ok("---\n"
                              "clip: |\n  a\n  b\n\n"
                              "strip: |-\n  a\n\n"
                              "keep: |+\n  a\n\n\n"
                              "folded: >\n  one\n  two\n\n  three\n"
                              "spaced: >-\n  text\n    code\n  more\n"
                              "spaced_gap: >2-\n    code\n\n  text\n"
                              "leading: |-\n\n  after blank\n"
                              "explicit: |2-\n     deeper\n"
                              "header_comment: >- # trailing comment\n  x\n"
                              "hash_inside: |-\n  # not a comment\n"
                              "empty_keep: |+\n\n"
                              "next: done\n"
                              "---\n");
    CHECK(doc.get("clip") == "a\nb\n");
    CHECK(doc.get("strip") == "a");
    CHECK(doc.get("keep") == "a\n\n\n");
    CHECK(doc.get("folded") == "one two\nthree\n");
    CHECK(doc.get("spaced") == "text\n  code\nmore");
    CHECK(doc.get("spaced_gap") == "  code\n\ntext");
    CHECK(doc.get("leading") == "\nafter blank");
    CHECK(doc.get("explicit") == "   deeper");
    CHECK(doc.get("header_comment") == "x");
    CHECK(doc.get("hash_inside") == "# not a comment");
    CHECK(doc.get("next") == "done");
}

TEST_CASE("Frontmatter quoted scalars decode escapes and fold lines", "[frontmatter]") {
    const auto doc = parse_ok("---\n"
                              "dq: \"tab\\there \\\"q\\\" \\\\ \\x41\\u00e9\\U0001F600 \\/\"\n"
                              "sq: 'it''s \"raw\" \\n'\n"
                              "multi: \"first\n  second\n\n  third\"\n"
                              "joined: \"abc\\\n    def\"\n"
                              "escaped_space: \"a\\ \n  b\"\n"
                              "colon: \"a: b\" # comment\n"
                              "hash: 'a #b'\n"
                              "---\n");
    CHECK(doc.get("dq") == "tab\there \"q\" \\ A\xC3\xA9\xF0\x9F\x98\x80 /");
    CHECK(doc.get("sq") == "it's \"raw\" \\n");
    CHECK(doc.get("multi") == "first second\nthird");
    CHECK(doc.get("joined") == "abcdef");
    CHECK(doc.get("escaped_space") == "a  b");
    CHECK(doc.get("colon") == "a: b");
    CHECK(doc.get("hash") == "a #b");
}

TEST_CASE("Frontmatter plain scalars fold and drop comments", "[frontmatter]") {
    const auto doc = parse_ok("---\n"
                              "plain: a value that\n  continues here\n\n  and a new line\n"
                              "comment: kept # dropped\n"
                              "apostrophe: Don't stop # dropped\n"
                              "hash_in_word: C#  and a#b\n"
                              "colon: Use when: the review is done\n"
                              "url: https://egm.dev/x\n"
                              "empty:\n"
                              "---\n");
    CHECK(doc.get("plain") == "a value that continues here\nand a new line");
    CHECK(doc.get("comment") == "kept");
    CHECK(doc.get("apostrophe") == "Don't stop");
    CHECK(doc.get("hash_in_word") == "C#  and a#b");
    CHECK(doc.get("colon") == "Use when: the review is done");
    CHECK(doc.get("url") == "https://egm.dev/x");
    CHECK(doc.get("empty") == "");
}

TEST_CASE("Frontmatter sequences are typed and nested structure stays nested", "[frontmatter]") {
    const auto doc = parse_ok("---\n"
                              "name: real\n"
                              "flow: [read, \"grep_search\", 'file_search', ]\n"
                              "flow_lines: [a,\n  b, # note\n  c]\n"
                              "block:\n  - task\n  - 'write_file'  # comment\n  -\n"
                              "compact:\n- one\n# between\n- two\n"
                              "spaced_list: read grep_search\n"
                              "metadata:\n  name: nested\n  author: someone\n"
                              "inline_map: {a: 1, b: [2, 3]}\n"
                              "list_of_maps:\n  - name: x\n    kind: y\n"
                              "nested_flow: [[a], b]\n"
                              "---\n");
    CHECK(doc.get("name") == "real");
    CHECK(doc.find("author") == nullptr);
    CHECK(doc.list("flow") == std::vector<std::string>{"read", "grep_search", "file_search"});
    CHECK(doc.list("flow_lines") == std::vector<std::string>{"a", "b", "c"});
    CHECK(doc.list("block") == std::vector<std::string>{"task", "write_file"});
    CHECK(doc.find("block")->items.size() == 3);  // the empty entry is kept in the value
    CHECK(doc.list("compact") == std::vector<std::string>{"one", "two"});
    CHECK(doc.list("spaced_list") == std::vector<std::string>{"read", "grep_search"});
    CHECK_FALSE(doc.get("flow").has_value());
    for (const auto* key : {"metadata", "inline_map", "list_of_maps", "nested_flow"}) {
        INFO(key);
        REQUIRE(doc.find(key) != nullptr);
        CHECK(doc.find(key)->kind == fm::Value::Kind::Nested);
        CHECK_FALSE(doc.list(key).has_value());
    }
}

TEST_CASE("Frontmatter delimiters, line endings and body", "[frontmatter]") {
    const auto crlf = parse_ok("\xEF\xBB\xBF---\r\nname: x\r\ndescription: >-\r\n  a\r\n  b\r\n...\r\n\r\n\r\nBody line\r\n\r\n");
    CHECK(crlf.get("name") == "x");
    CHECK(crlf.get("description") == "a b");
    CHECK(crlf.body == "Body line");

    const auto rule_in_body = parse_ok("---\na: 1\n---\nText\n---\nMore\n");
    CHECK(rule_in_body.body == "Text\n---\nMore");
}

TEST_CASE("Frontmatter accepts what libyaml accepts: properties, aliases, loose continuations",
          "[frontmatter]") {
    const auto doc = parse_ok("---\n"
                              "base: &shared read\n"
                              "copy: *shared\n"
                              "tagged: !!str 42\n"
                              "list: &tools [a, &one b]\n"
                              "again: *tools\n"
                              "flow_alias: [*one, c]\n"
                              "description: \"A long description that a tool wrapped\n"
                              "at column zero, as libyaml and js-yaml accept.\"\n"
                              "metadata:\n"
                              "  patterns:\n"
                              "#    - commented out at column 0\n"
                              "    - 'x'\n"
                              "    - '[^\\n]*\n"
                              "wrapped'\n"
                              "  docs:\n"
                              "  - compact\n"
                              "  - entries\n"
                              "  owner: someone\n"
                              "seq_of_maps:\n"
                              "  - name: a\n"
                              "    kind: b\n"
                              "  - &m name: c\n"
                              "&k !!str keyed: value\n"
                              "anchored_empty: [&e, x]\n"
                              "name: real\n"
                              "---\n");
    CHECK(doc.get("keyed") == "value");
    CHECK(doc.find("anchored_empty")->items == std::vector<std::string>{"", "x"});
    CHECK(doc.get("base") == "read");
    CHECK(doc.get("copy") == "read");
    CHECK(doc.get("tagged") == "42");
    CHECK(doc.list("again") == std::vector<std::string>{"a", "b"});
    CHECK(doc.list("flow_alias") == std::vector<std::string>{"b", "c"});
    CHECK(doc.get("description")
          == "A long description that a tool wrapped at column zero, as libyaml and js-yaml accept.");
    CHECK(doc.find("metadata")->kind == fm::Value::Kind::Nested);
    CHECK(doc.find("seq_of_maps")->kind == fm::Value::Kind::Nested);
    CHECK(doc.get("name") == "real");
    CHECK(doc.find("owner") == nullptr);
}

TEST_CASE("Frontmatter rejects malformed input with the line number", "[frontmatter]") {
    struct Case {
        std::string_view text;
        std::size_t line;
        std::string_view message;
    };
    const Case cases[] = {
        {"no frontmatter\n", 1, "opening"},
        {"---\na: 1\n", 1, "closing"},
        {"---\na: 1\na: 2\n---\n", 3, "duplicate key 'a'"},
        {"---\na:\n\t- x\n---\n", 3, "tab"},
        {"---\n  a: 1\n---\n", 2, "unexpected indentation"},
        {"---\na: \"open\n---\n", 2, "unterminated double-quoted"},
        {"---\na: 'open\n---\n", 2, "unterminated single-quoted"},
        {"---\na: [x, y\n---\n", 2, "unterminated flow sequence"},
        {"---\na: [x,, y]\n---\n", 2, "empty entry"},
        {"---\na: \"bad \\q\"\n---\n", 2, "invalid escape"},
        {"---\na: \"\\ud800\"\n---\n", 2, "Unicode scalar"},
        {"---\na: \"x\" trailing\n---\n", 2, "after a quoted scalar"},
        {"---\na: |x\n  y\n---\n", 2, "block scalar header"},
        {"---\na: |\n    deep\n  shallow\n---\n", 4, "indented less"},
        {"---\na: >-\n    detected from here\n\n  shallower\n---\n", 5, "indented less"},
        {"---\na: *nowhere\n---\n", 2, "undefined alias"},
        {"---\na: &x 1\nb: [*y]\n---\n", 3, "undefined alias"},
        {"---\n*a: v\n---\n", 2, "aliases as keys"},
        {"---\n]name: x\n---\n", 2, "indicator character"},
        {"---\nm:\n  |k: v\n---\n", 3, "indicator character"},
        {"---\na: @mention\n---\n", 2, "indicator character"},
        {"---\na: [x, : y]\n---\n", 2, "indicator character"},
        {"---\na:\n  b: 1\n  b: 2\n---\n", 4, "duplicate key 'b'"},
        {"---\na:\n  - k: v\n   x: y\n---\n", 4, "unexpected indentation"},
        {"---\na: \"x\"\n  b: y\n---\n", 3, "unexpected indentation"},
        {"---\n? complex\n---\n", 2, "complex"},
        {"---\na: - x\n---\n", 2, "line after its key"},
        {"---\n- x\n---\n", 2, "not a sequence"},
        {"---\njust text\n---\n", 2, "expected 'key: value'"},
        {"---\na: x # c\n  more\n---\n", 3, "unexpected indentation"},
        {"---\na:\n  - x\n  y\n---\n", 4, "expected a '- ' sequence entry"},
    };
    for (const auto& c : cases) {
        INFO(c.text);
        const auto error = parse_error(c.text);
        CHECK(error.line == c.line);
        CHECK_THAT(error.message, Catch::Matchers::ContainsSubstring(std::string(c.message)));
    }
}

TEST_CASE("Frontmatter bounds how deeply block collections nest", "[frontmatter]") {
    const auto ladder = [](std::size_t levels) {
        std::string text = "---\nx:\n";
        for (std::size_t i = 0; i < levels; ++i) {
            text += std::string(2 * (i + 1), ' ') + "k" + std::to_string(i) + ":\n";
        }
        text += std::string(2 * (levels + 1), ' ') + "leaf\n---\n";
        return text;
    };
    CHECK(fm::parse(ladder(32)).has_value());
    CHECK_THAT(parse_error(ladder(5000)).message, Catch::Matchers::ContainsSubstring("nesting deeper"));

    std::string dashes = "---\nx:\n  ";
    for (int i = 0; i < 100000; ++i) dashes += "- ";
    dashes += "leaf\n---\n";
    const auto error = parse_error(dashes);
    CHECK(error.line == 3);
    CHECK_THAT(error.message, Catch::Matchers::ContainsSubstring("nesting deeper"));
}

TEST_CASE("Frontmatter booleans and lists", "[frontmatter]") {
    CHECK(fm::parse_bool(" True ") == true);
    CHECK(fm::parse_bool("off") == false);
    CHECK_FALSE(fm::parse_bool("maybe").has_value());
    CHECK(fm::parse_bool("1") == true);
    CHECK(fm::parse_bool("0") == false);
    CHECK(fm::split_list("a, b ,c") == std::vector<std::string>{"a", "b", "c"});
    CHECK(fm::split_list("Bash(git add:*) read") == std::vector<std::string>{"Bash(git", "add:*)", "read"});
    CHECK(fm::split_list("Bash(git add:*), read") == std::vector<std::string>{"Bash(git add:*)", "read"});
    CHECK(fm::split_list("  ").empty());
}

TEST_CASE("Every real-world manifest shape in a skill still loads", "[frontmatter][skills]") {
    Sandbox sandbox("manifest_shapes");
    write_file(sandbox.root / "s" / "SKILL.md", "---\n"
                                              "name: shaped\n"
                                              "description: |\n  Multi-line\n  description.\n"
                                              "allowed-tools:\n  - read\n  - grep_search\n"
                                              "user-invocable: no\n"
                                              "enabled: maybe\n"
                                              "metadata:\n  name: hijack\n"
                                              "---\nBody\n");
    const auto manifest = core::tools::SkillRegistry::parse_manifest(sandbox.root / "s");
    REQUIRE(manifest.has_value());
    CHECK(manifest->name == "shaped");
    CHECK(manifest->description == "Multi-line\ndescription.");
    CHECK(manifest->allowed_tools == std::vector<std::string>{"read", "grep_search"});
    CHECK_FALSE(manifest->user_invocable);
    CHECK(manifest->enabled);  // not a boolean: the default stands
}

TEST_CASE("An installed plugin contributes skills that know their bundle",
          "[skills][plugins]") {
    Sandbox sandbox("installed");
    const auto home = sandbox.root / "home";
    const auto project = sandbox.root / "project";
    fs::create_directories(project);
    const auto plugin = home / ".config" / "filo" / "plugins" / "egm";
    write_plugin(plugin);
    ScopedEnvVar scoped_home("HOME", home.string());
    const std::vector<fs::path> roots{project};

    const auto all = core::tools::SkillRegistry::discover_all(roots);
    const auto review = std::ranges::find(all, std::string("egm-review"), &core::tools::SkillManifest::name);
    REQUIRE(review != all.end());
    CHECK(review->description == "Orchestrates a multi-stage review with fresh-context reviewers.");
    CHECK(review->plugin_root == fs::weakly_canonical(plugin));
    CHECK_FALSE(review->model_invocable);

    SECTION("disable-model-invocation keeps it out of the model's catalog") {
        CHECK(core::tools::SkillRegistry::discover_instruction_skills(roots).empty());
        CHECK(core::tools::SkillRegistry::build_catalog_prompt(roots).empty());
    }

    SECTION("its slash command names the skill directory and plugin root") {
        core::commands::SkillCommand command(*review);
        std::string sent;
        core::commands::CommandContext context{
            .text = "/egm-review design.md",
            .clear_input_fn = [] {},
            .append_history_fn = [](const std::string&) {},
            .agent = nullptr,
        };
        context.send_user_message_fn = [&](const std::string& message) { sent = message; };
        command.execute(context);
        CHECK_THAT(sent, Catch::Matchers::StartsWith("Spawn the `egm-zoom-out`"));
        CHECK_THAT(sent, Catch::Matchers::ContainsSubstring(
                             "Plugin root: " + fs::weakly_canonical(plugin).string()));
        CHECK_THAT(sent, Catch::Matchers::ContainsSubstring("resolve against the plugin root"));
    }

    SECTION("its bundle is readable but not writable") {
        auto snapshot = core::workspace::WorkspaceSnapshot{.primary = project, .enforce = true};
        snapshot.extensions = core::workspace::FileAccessScope(
            core::tools::SkillRegistry::package_read_roots(roots), {});
        const core::workspace::SessionWorkspace workspace(snapshot);
        CHECK(workspace.allows_read(plugin / "agents" / "zoom_out.md"));
        CHECK(workspace.allows_read(plugin / "resources" / "standard.md"));
        CHECK_FALSE(workspace.allows_write(plugin / "agents" / "zoom_out.md"));
        CHECK_FALSE(workspace.allows_read(home / ".ssh" / "id_rsa"));
    }
}

TEST_CASE("A skill linked from a plugin checkout finds its bundle", "[skills][plugins]") {
    Sandbox sandbox("linked");
    const auto home = sandbox.root / "home";
    const auto project = sandbox.root / "project";
    const auto checkout = sandbox.root / "checkout" / "egm";
    write_plugin(checkout);
    fs::create_directories(project / ".agents" / "skills");
    fs::create_directory_symlink(checkout / "skills" / "egm-review",
                                 project / ".agents" / "skills" / "egm-review");
    fs::create_directories(home);
    ScopedEnvVar scoped_home("HOME", home.string());
    const std::vector<fs::path> roots{project};

    const auto all = core::tools::SkillRegistry::discover_all(roots);
    REQUIRE(all.size() == 1);
    CHECK(all.front().plugin_root == fs::weakly_canonical(checkout));

    const auto definitions = core::agent::AgentDefinitionRegistry::discover(roots);
    CHECK(std::ranges::contains(definitions, std::string("egm-zoom-out"),
                                &core::agent::AgentDefinition::name));

    const auto readable = core::tools::SkillRegistry::package_read_roots(roots);
    CHECK(std::ranges::contains(readable, fs::weakly_canonical(checkout)));
}

TEST_CASE("activate_skill reads plugin resources but never escapes the bundle",
          "[skills][plugins][agent_skills]") {
    Sandbox sandbox("activate");
    const auto home = sandbox.root / "home";
    const auto project = sandbox.root / "project";
    fs::create_directories(project);
    const auto plugin = project / ".filo" / "plugins" / "egm";
    write_plugin(plugin);
    // Model-invocable so activate_skill can see it.
    write_file(plugin / "skills" / "egm-draft" / "SKILL.md", R"(---
name: egm-draft
description: Draft a spec.
---
Read resources/standard.md.
)");
    write_file(sandbox.root / "secret.txt", "secret");
    fs::create_directories(home);
    ScopedEnvVar scoped_home("HOME", home.string());

    const auto context = test_support::make_session_context({.primary = project, .enforce = true});
    core::tools::ActivateSkillTool tool({project});
    simdjson::dom::parser parser;

    const auto inside = tool.execute(
        R"({"name":"egm-draft","resource_path":"../../resources/standard.md"})", context);
    simdjson::dom::element doc;
    REQUIRE(parser.parse(inside).get(doc) == simdjson::SUCCESS);
    std::string_view content;
    REQUIRE(doc["content"].get(content) == simdjson::SUCCESS);
    CHECK(content == "Prefer deletion over addition.\n");

    const auto outside = tool.execute(
        R"({"name":"egm-draft","resource_path":"../../../../../../secret.txt"})", context);
    CHECK_THAT(outside, Catch::Matchers::ContainsSubstring("outside skill directory"));

    const auto hidden = tool.execute(R"({"name":"egm-review"})", context);
    CHECK_THAT(hidden, Catch::Matchers::ContainsSubstring("Unknown or unavailable"));
}

TEST_CASE("activate_skill keeps withheld steering out of a plugin root",
          "[skills][plugins][agent_skills][steering]") {
    // A checkout that is both a project and a plugin: its skill, linked into
    // .agents/skills, has the project root as its plugin root.
    Sandbox sandbox("activate_steering");
    const auto home = sandbox.root / "home";
    const auto project = sandbox.root / "project";
    write_file(project / "plugin.json", R"({"name":"self"})");
    write_file(project / "AGENTS.md", "WITHHELD-STEERING\n");
    write_file(project / "notes.md", "Plain notes.\n");
    write_file(project / "skills" / "self-draft" / "SKILL.md", R"(---
name: self-draft
description: Draft a spec.
---
Read notes.md.
)");
    fs::create_directories(project / ".agents" / "skills");
    fs::create_directory_symlink(project / "skills" / "self-draft",
                                 project / ".agents" / "skills" / "self-draft");
    fs::create_directories(home);
    ScopedEnvVar scoped_home("HOME", home.string());

    auto context = test_support::make_session_context({.primary = project, .enforce = true});
    context.steering_policy = core::context::SteeringPolicy{.mode = core::context::SteeringMode::None};
    core::tools::ActivateSkillTool tool({project});

    const auto steering = tool.execute(
        R"({"name":"self-draft","resource_path":"../../AGENTS.md"})", context);
    CHECK_THAT(steering, Catch::Matchers::ContainsSubstring("Access denied"));
    CHECK_THAT(steering, !Catch::Matchers::ContainsSubstring("WITHHELD-STEERING"));

    const auto notes = tool.execute(
        R"({"name":"self-draft","resource_path":"../../notes.md"})", context);
    CHECK_THAT(notes, Catch::Matchers::ContainsSubstring("Plain notes."));

    context.steering_policy = core::context::SteeringPolicy{.mode = core::context::SteeringMode::Default};
    const auto allowed = tool.execute(
        R"({"name":"self-draft","resource_path":"../../AGENTS.md"})", context);
    CHECK_THAT(allowed, Catch::Matchers::ContainsSubstring("WITHHELD-STEERING"));
}

TEST_CASE("A plugin persona can narrow its capabilities but never widen them",
          "[agent][plugins]") {
    Sandbox sandbox("definitions");
    const auto plugin = sandbox.root / "egm";
    write_plugin(plugin);

    const auto zoom = core::agent::AgentDefinitionRegistry::parse_file(
        plugin / "agents" / "zoom_out.md", plugin);
    REQUIRE(zoom.has_value());
    CHECK(zoom->name == "egm-zoom-out");
    CHECK(zoom->description == "Skeptical Architect persona focused on simplicity.");
    CHECK_FALSE(zoom->tools.has_value());
    CHECK(zoom->read_only);
    CHECK(zoom->isolated);
    CHECK_FALSE(zoom->allow_task_tool);
    CHECK_THAT(zoom->prompt, Catch::Matchers::StartsWith("# Core Mandates"));
    CHECK_THAT(zoom->prompt, Catch::Matchers::ContainsSubstring(
                                 "plugin bundle at " + plugin.string()));
    CHECK_THAT(zoom->prompt, !Catch::Matchers::ContainsSubstring("ADAPTER-BODY"));

    // The synthesizer asks for delegation, writes, project context and a
    // model; a bundle gets none of that from its own file.
    const auto synthesizer = core::agent::AgentDefinitionRegistry::parse_file(
        plugin / "agents" / "synthesizer.md", plugin);
    REQUIRE(synthesizer.has_value());
    CHECK(*synthesizer->tools == std::vector<std::string>{"read", "write_file"});
    CHECK_FALSE(synthesizer->allow_task_tool);
    CHECK(synthesizer->read_only);
    CHECK(synthesizer->isolated);
    CHECK(synthesizer->model.empty());
    CHECK(synthesizer->max_steps == 30);
}

TEST_CASE("The user's own agent definitions configure every setting", "[agent][plugins]") {
    Sandbox sandbox("own_definitions");
    write_file(sandbox.root / "helper.md", R"(---
name: Helper
description: >
  The user's own
  helper.
tools: [read, task, run_terminal_command, read]
read-only: yes
isolated: true
max-steps: 7
provider: openai
model: my-model
color: blue
---
Help.
)");
    const auto own = core::agent::AgentDefinitionRegistry::parse_file(sandbox.root / "helper.md");
    REQUIRE(own.has_value());
    CHECK(own->name == "helper");
    CHECK(own->description == "The user's own helper.");
    CHECK(*own->tools == std::vector<std::string>{"read", "run_terminal_command"});
    CHECK(own->allow_task_tool);
    CHECK(own->read_only);
    CHECK(own->isolated);
    CHECK(own->max_steps == 7);
    CHECK(own->provider == "openai");
    CHECK(own->model == "my-model");
    CHECK(own->prompt == "Help.");

    const auto rejected = [&](std::string_view name, std::string_view content) {
        write_file(sandbox.root / name, content);
        return !core::agent::AgentDefinitionRegistry::parse_file(sandbox.root / name).has_value();
    };
    CHECK(rejected("disabled.md", "---\nname: off\ndescription: d\nenabled: false\n---\nBody\n"));
    CHECK(rejected("spaced.md", "---\nname: two words\ndescription: d\n---\nBody\n"));
    CHECK(rejected("at.md", "---\nname: \"@x\"\ndescription: d\n---\nBody\n"));
    CHECK(rejected("nodesc.md", "---\nname: x\n---\nBody\n"));
    CHECK(rejected("nobody.md", "---\nname: x\ndescription: d\n---\n"));
    CHECK(rejected("broken.md", "---\nname: x\nname: y\ndescription: d\n---\nBody\n"));

    write_file(sandbox.root / "lenient.md",
               "---\nname: x\ndescription: d\nmax-steps: lots\nisolated: maybe\n---\nBody\n");
    const auto lenient = core::agent::AgentDefinitionRegistry::parse_file(sandbox.root / "lenient.md");
    REQUIRE(lenient.has_value());
    CHECK_FALSE(lenient->max_steps.has_value());
    CHECK_FALSE(lenient->isolated);
}

TEST_CASE("Plugin personas become isolated, read-only subagent profiles",
          "[agent][orchestration][plugins]") {
    Sandbox sandbox("profiles");
    const auto home = sandbox.root / "home";
    const auto project = sandbox.root / "project";
    write_plugin(project / ".filo" / "plugins" / "egm");
    write_file(project / "AGENTS.md", "PROJECT-STEERING-MARKER\n");
    write_file(project / ".filo" / "agents" / "general.md", R"(---
name: general
description: Hijacked.
---
Hijack.
)");
    fs::create_directories(home);
    ScopedEnvVar scoped_home("HOME", home.string());

    // A private registry: these tools must not outlive the test in the
    // process-wide one other test cases see.
    core::tools::ToolManager tools;
    tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
    tools.register_tool(std::make_shared<core::tools::ReadTool>());
    tools.register_tool(std::make_shared<core::tools::FileSearchTool>());
    tools.register_tool(std::make_shared<core::tools::GrepSearchTool>());

    core::config::AppConfig config;
    core::config::SubagentConfig pinned;
    pinned.max_steps = 5;
    config.subagents.emplace("egm-zoom-out", std::move(pinned));
    core::agent::SubagentOrchestrator orchestrator(tools, &config);
    const auto context = test_support::make_session_context({.primary = project, .enforce = true});

    const auto plan_for = [&](std::string_view worker) {
        return orchestrator.build_execution_plan({
            .worker = std::string(worker),
            .inherited_provider = std::make_shared<RecordingProvider>(),
            .inherited_model = "test-model",
            .prefer_inherited_provider = true,
            .workspace_roots = {project},
        });
    };
    const auto plan = plan_for("egm-zoom-out");
    REQUIRE(plan.has_value());
    CHECK(plan->read_only);
    CHECK(plan->isolated);
    CHECK(plan->max_steps == 5);  // subagents.<name> still wins over the file
    CHECK_THAT(plan->worker_prompt, Catch::Matchers::ContainsSubstring("Goldfish reviewer"));
    const auto zoom_tools = plan->allowed_tool_names();
    CHECK(std::ranges::contains(zoom_tools, std::string("read")));
    CHECK_FALSE(std::ranges::contains(zoom_tools, std::string("write_file")));

    // The synthesizer's own request for writes and delegation was refused...
    const auto synthesizer = plan_for("egm-synthesizer");
    REQUIRE(synthesizer.has_value());
    CHECK(synthesizer->read_only);
    CHECK_FALSE(std::ranges::contains(synthesizer->allowed_tool_names(), std::string("write_file")));
    CHECK_FALSE(std::ranges::contains(synthesizer->allowed_tool_names(), std::string("task")));

    // ...and the user grants it explicitly.
    core::config::SubagentConfig grant;
    grant.read_only = false;
    grant.allow_task_tool = true;
    config.subagents.emplace("egm-synthesizer", std::move(grant));
    orchestrator.reload_profiles(config);
    const auto granted = plan_for("egm-synthesizer");
    REQUIRE(granted.has_value());
    CHECK_FALSE(granted->read_only);
    CHECK(std::ranges::contains(granted->allowed_tool_names(), std::string("write_file")));
    CHECK(std::ranges::contains(granted->allowed_tool_names(), std::string("task")));
    CHECK(granted->isolated);

    const auto general = orchestrator.build_execution_plan({
        .worker = "general",
        .inherited_provider = std::make_shared<RecordingProvider>(),
        .prefer_inherited_provider = true,
        .workspace_roots = {project},
    });
    REQUIRE(general.has_value());
    CHECK_THAT(general->worker_prompt, !Catch::Matchers::ContainsSubstring("Hijack"));

    auto run = [&](std::string_view worker) {
        auto provider = std::make_shared<RecordingProvider>();
        const auto result = orchestrator.execute_task(
            std::format(R"({{"description":"review","prompt":"review design.md","subagent_type":"{}"}})",
                        worker),
            provider,
            {.active_model = "test-model", .parent_mode = "BUILD", .session_context = context,
             .permission_check = {}});
        CHECK_FALSE(result.contains("\"error\""));
        const auto requests = provider->requests();
        REQUIRE_FALSE(requests.empty());
        return system_text(requests.front());
    };
    CHECK_THAT(run("egm-zoom-out"), !Catch::Matchers::ContainsSubstring("PROJECT-STEERING-MARKER"));
    CHECK_THAT(run("general"), Catch::Matchers::ContainsSubstring("PROJECT-STEERING-MARKER"));
}

TEST_CASE("An additional workspace root's personas are restricted like a plugin's",
          "[agent][plugins]") {
    Sandbox sandbox("secondary_agents");
    const auto home = sandbox.root / "home";
    const auto primary = sandbox.root / "primary";
    const auto secondary = sandbox.root / "secondary";
    const auto persona = [](std::string_view name) {
        return std::format(R"(---
name: {}
description: A helper.
tools: [task, read, write_file]
read-only: false
isolated: false
provider: openai
model: my-model
---
Help.
)", name);
    };
    write_file(primary / ".filo" / "agents" / "own.md", persona("own"));
    write_file(secondary / ".filo" / "agents" / "guest.md", persona("guest"));
    fs::create_directories(home);
    ScopedEnvVar scoped_home("HOME", home.string());

    const auto definitions = core::agent::AgentDefinitionRegistry::discover({primary, secondary});
    const auto own = std::ranges::find(definitions, std::string("own"), &core::agent::AgentDefinition::name);
    const auto guest = std::ranges::find(definitions, std::string("guest"), &core::agent::AgentDefinition::name);
    REQUIRE(own != definitions.end());
    REQUIRE(guest != definitions.end());

    CHECK(own->allow_task_tool);
    CHECK_FALSE(own->read_only);
    CHECK(own->model == "my-model");

    CHECK_FALSE(guest->allow_task_tool);
    CHECK(guest->read_only);
    CHECK(guest->isolated);
    CHECK(guest->provider.empty());
    CHECK(guest->model.empty());
    CHECK(*guest->tools == std::vector<std::string>{"read", "write_file"});
    CHECK(guest->prompt == "Help.");

    // The same directory as the primary is trusted again.
    const auto promoted = core::agent::AgentDefinitionRegistry::discover({secondary});
    REQUIRE(promoted.size() == 1);
    CHECK(promoted.front().allow_task_tool);
}

TEST_CASE("The task tool lists the agent files of the session's own workspace",
          "[agent][orchestration][plugins]") {
    Sandbox sandbox("task_description");
    const auto home = sandbox.root / "home";
    const auto mine = sandbox.root / "mine";
    const auto theirs = sandbox.root / "theirs";
    write_file(mine / ".filo" / "agents" / "mine.md",
               "---\nname: mine-helper\ndescription: Mine.\nread-only: true\n---\nHelp.\n");
    write_file(theirs / ".filo" / "agents" / "theirs.md",
               "---\nname: their-helper\ndescription: Theirs.\n---\nHelp.\n");
    fs::create_directories(home);
    ScopedEnvVar scoped_home("HOME", home.string());

    core::tools::ToolManager tools;
    tools.register_tool(std::make_shared<core::tools::ReadTool>());
    tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
    core::agent::SubagentOrchestrator orchestrator(tools);
    const auto description = orchestrator.task_tool_definition({mine}).function.description;
    CHECK_THAT(description, Catch::Matchers::ContainsSubstring("mine-helper"));
    CHECK_THAT(description, !Catch::Matchers::ContainsSubstring("their-helper"));

    const auto other = orchestrator.task_tool_definition({theirs}).function.description;
    CHECK_THAT(other, Catch::Matchers::ContainsSubstring("their-helper"));
    CHECK_THAT(other, !Catch::Matchers::ContainsSubstring("mine-helper"));

    // Read-only classification (which skips the permission prompt) resolves
    // the persona from the same roots.
    CHECK(orchestrator.task_is_read_only(R"({"subagent_type":"mine-helper"})", "BUILD", {mine}));
    CHECK_FALSE(orchestrator.task_is_read_only(R"({"subagent_type":"mine-helper"})", "BUILD", {theirs}));
}

TEST_CASE("Nested delegation stops at the depth limit", "[agent][orchestration]") {
    core::agent::SubagentOrchestrator orchestrator(core::tools::ToolManager::get_instance());
    auto context = test_support::make_workspace_session_context();
    context.delegation_depth = core::agent::SubagentOrchestrator::kMaxDelegationDepth;
    auto provider = std::make_shared<RecordingProvider>();

    const auto result = orchestrator.execute_task(
        R"({"description":"nest","prompt":"go deeper","subagent_type":"general"})",
        provider,
        {.active_model = "m", .parent_mode = "BUILD", .session_context = context,
         .permission_check = {}});
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("Delegation depth limit"));
    CHECK(provider->requests().empty());
}
