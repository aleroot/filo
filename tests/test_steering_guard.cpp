#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_contains.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/ContextBuilder.hpp"
#include "core/context/SessionContext.hpp"
#include "core/context/SteeringGuard.hpp"
#include "core/context/SteeringLoader.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/PathVisibilityToolDecorator.hpp"
#include "core/tools/ReadTool.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/SteeringEnforcement.hpp"
#include "core/tools/ToolArgumentUtils.hpp"
#include "core/landrun/LandrunPolicy.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/workspace/SessionWorkspace.hpp"
#include "core/workspace/Workspace.hpp"

#ifdef FILO_ENABLE_PYTHON
#include "core/tools/PythonInterpreterTool.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

class TempDir {
public:
    explicit TempDir(std::string_view label)
        : path_(fs::temp_directory_path()
                / std::format(
                    "filo_steering_guard_{}_{}",
                    label,
                    std::chrono::steady_clock::now().time_since_epoch().count())) {
        fs::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

class ScopedWorkspace {
public:
    explicit ScopedWorkspace(const fs::path& root) {
        core::workspace::Workspace::get_instance().initialize(root, {}, true);
    }

    ~ScopedWorkspace() {
        core::workspace::Workspace::get_instance().initialize({}, {}, false);
    }
};

void write_text(const fs::path& path, std::string_view content) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
}

[[nodiscard]] core::context::SessionContext make_context(
    const fs::path& primary,
    core::context::SteeringPolicy policy,
    std::vector<fs::path> additional = {}) {
    auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = primary,
            .additional = std::move(additional),
            .enforce = true,
            .version = 1,
        },
        core::context::SessionTransport::cli,
        "steering-guard-test");
    context.steering_policy = std::move(policy);
    return context;
}

/// The loader resolves roots through weakly_canonical(), so a macOS /var/...
/// temp path only compares equal after the same normalization (/var ->
/// /private/var). Expectations must go through this.
[[nodiscard]] fs::path resolved(const fs::path& path) {
    return core::workspace::SessionWorkspace::normalize_path(path);
}

[[nodiscard]] core::context::SteeringPolicy no_steering() {
    return core::context::SteeringPolicy{.mode = core::context::SteeringMode::None};
}

[[nodiscard]] core::context::SteeringPolicy default_steering() {
    return core::context::SteeringPolicy{.mode = core::context::SteeringMode::Default};
}

/// A workspace carrying every flavour of steering file: two root-level names, a
/// `.filo/steering` directory, and an unrelated markdown file that must stay
/// readable throughout.
struct SteeringWorkspace {
    TempDir dir{"project"};

    SteeringWorkspace() {
        write_text(dir.path() / "AGENTS.md", "needle agents steering\n");
        write_text(dir.path() / "CLAUDE.md", "needle claude steering\n");
        write_text(dir.path() / ".filo" / "steering" / "style.md", "needle style steering\n");
        write_text(dir.path() / "README.md", "needle readme\n");
    }

    [[nodiscard]] fs::path path(std::string_view relative) const {
        return dir.path() / fs::path(relative);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// The name tables and the I/O-free candidate filter
// ---------------------------------------------------------------------------

TEST_CASE("steering filenames are recognised case-insensitively", "[steering][guard]") {
    using core::context::is_steering_filename;

    CHECK(is_steering_filename("AGENTS.md"));
    CHECK(is_steering_filename("agents.md"));
    CHECK(is_steering_filename("AGENTS.override.md"));
    CHECK(is_steering_filename("CLAUDE.md"));
    CHECK(is_steering_filename("FILO.md"));
    CHECK(is_steering_filename("GEMINI.md"));
    CHECK(is_steering_filename("SYSTEM.md"));
    CHECK(is_steering_filename("CURSOR.md"));
    CHECK(is_steering_filename("COPILOT.md"));

    CHECK_FALSE(is_steering_filename("README.md"));
    CHECK_FALSE(is_steering_filename("MYAGENTS.md"));
    CHECK_FALSE(is_steering_filename("AGENTS.md.bak"));
    CHECK_FALSE(is_steering_filename(""));
}

TEST_CASE("is_steering_candidate filters without touching the filesystem",
          "[steering][guard]") {
    using core::context::is_steering_candidate;

    CHECK(is_steering_candidate("docs/AGENTS.md"));
    CHECK(is_steering_candidate("docs/agents.md"));
    CHECK(is_steering_candidate("/tmp/project/.filo/steering"));
    CHECK(is_steering_candidate("/tmp/project/.filo/steering/style.md"));

    CHECK_FALSE(is_steering_candidate("docs/README.md"));
    CHECK_FALSE(is_steering_candidate("/tmp/project/.filo"));
    CHECK_FALSE(is_steering_candidate("/tmp/project/.filo/settings.json"));
    CHECK_FALSE(is_steering_candidate("/tmp/project/.filo/steering/notes.txt"));
    CHECK_FALSE(is_steering_candidate("src/steering_loader.cpp"));
}

// ---------------------------------------------------------------------------
// The derived rule: blocked == discovered - loaded
// ---------------------------------------------------------------------------

TEST_CASE("SteeringGuard blocks exactly what the policy withholds", "[steering][guard]") {
    SteeringWorkspace workspace;
    const std::vector<fs::path> roots{resolved(workspace.dir.path())};

    SECTION("discovery sees every steering file, whichever policy is active") {
        const auto discovered = core::context::discover_steering_files(roots);
        REQUIRE(discovered.size() == 3);
        CHECK(discovered[0].label == "AGENTS.md");
        CHECK(discovered[0].content.empty());

        CHECK(core::context::select_steering_files(roots, no_steering()).empty());
        const auto selected = core::context::select_steering_files(roots, default_steering());
        REQUIRE(selected.files.size() == 3);
        CHECK(std::ranges::all_of(selected.files, [](const auto& file) { return file.enabled; }));
    }

    SECTION("none blocks all of them, plus the directory that holds them") {
        const auto guard = core::context::SteeringGuard::for_roots(roots, no_steering());
        REQUIRE_FALSE(guard.empty());

        const auto labels = guard.blocked_labels();
        CHECK_THAT(labels, Catch::Matchers::Contains("AGENTS.md"));
        CHECK_THAT(labels, Catch::Matchers::Contains("CLAUDE.md"));
        CHECK_THAT(labels, Catch::Matchers::Contains(".filo/steering/style.md"));
        CHECK(labels.size() == 3);

        CHECK(guard.blocked_reason(resolved(workspace.path("AGENTS.md"))).has_value());
        CHECK(guard.blocked_reason(resolved(workspace.path(".filo/steering/style.md"))).has_value());
        CHECK(guard.blocked_reason(resolved(workspace.path(".filo/steering"))).has_value());

        // Ordinary project files are untouched.
        CHECK_FALSE(guard.blocked_reason(resolved(workspace.path("README.md"))).has_value());
        CHECK_FALSE(guard.blocked_reason(resolved(workspace.path(".filo"))).has_value());
        CHECK_FALSE(guard.blocked_reason(resolved(workspace.path("src/main.cpp"))).has_value());
    }

    SECTION("default on a single root blocks nothing") {
        const auto guard = core::context::SteeringGuard::for_roots(roots, default_steering());
        CHECK(guard.empty());
        CHECK_FALSE(core::context::SteeringGuard::may_block(default_steering()));
        CHECK_FALSE(guard.blocked_reason(resolved(workspace.path("AGENTS.md"))).has_value());
    }

    SECTION("an individually unloaded source becomes unreadable") {
        auto policy = default_steering();
        policy.disable_source("AGENTS.md");

        CHECK(core::context::SteeringGuard::may_block(policy));
        const auto guard = core::context::SteeringGuard::for_roots(roots, policy);
        const auto labels = guard.blocked_labels();
        REQUIRE(labels.size() == 1);
        CHECK(labels.front() == "AGENTS.md");

        CHECK(guard.blocked_reason(resolved(workspace.path("AGENTS.md"))).has_value());
        CHECK_FALSE(guard.blocked_reason(resolved(workspace.path("CLAUDE.md"))).has_value());
        // A single file is still loaded from the directory, so it stays listed.
        CHECK_FALSE(guard.blocked_reason(resolved(workspace.path(".filo/steering"))).has_value());
    }

    SECTION("a custom file replaces project steering and blocks it") {
        const auto custom = workspace.path("custom-instructions.md");
        write_text(custom, "needle custom\n");

        const auto policy = core::context::parse_steering_policy(custom.string());
        REQUIRE(policy.mode == core::context::SteeringMode::CustomFile);

        const auto guard = core::context::SteeringGuard::for_roots(roots, policy);
        CHECK(guard.blocked_reason(resolved(workspace.path("AGENTS.md"))).has_value());
        CHECK(guard.blocked_reason(resolved(workspace.path("CLAUDE.md"))).has_value());
        // The file that *is* loaded stays readable.
        CHECK_FALSE(guard.blocked_reason(resolved(custom)).has_value());
    }

    SECTION("the denial explains itself") {
        const auto guard = core::context::SteeringGuard::for_roots(roots, no_steering());
        const auto reason = guard.blocked_reason(resolved(workspace.path("AGENTS.md")));
        REQUIRE(reason.has_value());
        REQUIRE_THAT(*reason, Catch::Matchers::ContainsSubstring("Access denied"));
        REQUIRE_THAT(*reason, Catch::Matchers::ContainsSubstring("AGENTS.md"));
        REQUIRE_THAT(*reason, Catch::Matchers::ContainsSubstring("none (disabled)"));
    }
}

/**
 * A shadowed steering file is still on disk, so it is still withheld.
 *
 * AGENTS.override.md outranks a sibling AGENTS.md and the loader reads only the
 * stronger one. Discovery used to report that as "AGENTS.md is not there", so the
 * guard's subtraction never saw it: under `--no-steering` the most important
 * instruction file in the project stayed readable through every tool while its
 * weaker siblings were denied. The guard therefore subtracts from
 * enumerate_steering_files() (presence) and not discover_steering_files()
 * (loadable), and this pins the difference.
 */
TEST_CASE("a shadowed AGENTS.md is withheld like any other steering file",
          "[steering][guard]") {
    SteeringWorkspace workspace;
    write_text(workspace.path("AGENTS.override.md"), "needle override steering\n");
    const std::vector<fs::path> roots{resolved(workspace.dir.path())};

    const auto labels_of = [](const auto& files) {
        std::vector<std::string> labels;
        for (const auto& file : files) {
            labels.push_back(file.label);
        }
        return labels;
    };

    SECTION("the loader still reads only the stronger name") {
        const auto loadable = labels_of(core::context::discover_steering_files(roots));
        CHECK_THAT(loadable, Catch::Matchers::Contains("AGENTS.override.md"));
        CHECK_THAT(loadable, !Catch::Matchers::Contains("AGENTS.md"));
        CHECK(loadable.size() == 3);
    }

    SECTION("the presence set reports both") {
        const auto present = labels_of(core::context::enumerate_steering_files(roots));
        CHECK_THAT(present, Catch::Matchers::Contains("AGENTS.override.md"));
        CHECK_THAT(present, Catch::Matchers::Contains("AGENTS.md"));
        CHECK(present.size() == 4);
    }

    SECTION("none blocks the shadowed file as well as the one that wins") {
        const auto guard = core::context::SteeringGuard::for_roots(roots, no_steering());
        const auto labels = guard.blocked_labels();
        CHECK_THAT(labels, Catch::Matchers::Contains("AGENTS.md"));
        CHECK_THAT(labels, Catch::Matchers::Contains("AGENTS.override.md"));
        CHECK(labels.size() == 4);

        CHECK(guard.blocked_reason(resolved(workspace.path("AGENTS.md"))).has_value());
        CHECK(guard.blocked_reason(resolved(workspace.path("AGENTS.override.md"))).has_value());
        // Still not a blanket denial of the project.
        CHECK_FALSE(guard.blocked_reason(resolved(workspace.path("README.md"))).has_value());
    }

    SECTION("default on a single root still blocks nothing") {
        const auto guard = core::context::SteeringGuard::for_roots(roots, default_steering());
        CHECK(guard.empty());
        CHECK_FALSE(guard.blocked_reason(resolved(workspace.path("AGENTS.md"))).has_value());
    }

    SECTION("unloading the override does not un-shadow AGENTS.md") {
        // The loader never falls back to a shadowed sibling, so unloading the
        // override leaves *both* files out of the prompt — and the guard blocks
        // exactly what the prompt lacks. Steering the policy still loads stays
        // readable, so this is a subtraction and not a blanket denial.
        auto policy = default_steering();
        policy.disable_source("AGENTS.override.md");

        const auto guard = core::context::SteeringGuard::for_roots(roots, policy);
        const auto labels = guard.blocked_labels();
        CHECK_THAT(labels, Catch::Matchers::Contains("AGENTS.override.md"));
        CHECK_THAT(labels, Catch::Matchers::Contains("AGENTS.md"));
        CHECK_THAT(labels, !Catch::Matchers::Contains("CLAUDE.md"));
        CHECK_THAT(labels, !Catch::Matchers::Contains(".filo/steering/style.md"));
    }
}

TEST_CASE("SteeringGuard matches aliases, not spellings", "[steering][guard]") {
    SteeringWorkspace workspace;
    const std::vector<fs::path> roots{resolved(workspace.dir.path())};
    const auto guard = core::context::SteeringGuard::for_roots(roots, no_steering());
    REQUIRE_FALSE(guard.empty());

    std::error_code ec;
    fs::create_symlink(workspace.path("AGENTS.md"), workspace.path("NOTES.md"), ec);
    REQUIRE_FALSE(ec);

    // A symlink with an innocent name is the same file, so it is the same denial.
    CHECK(guard.blocked_reason(resolved(workspace.path("NOTES.md"))).has_value());
    CHECK(guard.blocked_text_reason("cat NOTES.md", workspace.dir.path()).has_value());

    // Case-insensitive filesystems must not become a way around the rule.
    CHECK(guard.blocked_reason(resolved(workspace.path("agents.md"))).has_value());
    CHECK(guard.blocked_text_reason("cat agents.md", workspace.dir.path()).has_value());
}

TEST_CASE("automatic steering selection does not restrict multi-root tools",
          "[steering][guard][enforcement][regression]") {
    TempDir primary{"primary"};
    TempDir secondary{"secondary"};
    TempDir tertiary{"tertiary"};
    ScopedWorkspace scoped{primary.path()};
    write_text(secondary.path() / "AGENTS.md", "needle secondary\n");
    write_text(tertiary.path() / "AGENTS.md", "needle tertiary\n");

    SECTION("primary has steering including a shadowed file") {
        write_text(primary.path() / "AGENTS.md", "needle shadowed\n");
        write_text(primary.path() / "AGENTS.override.md", "needle primary\n");
    }
    SECTION("primary has no steering") {}

    for (const auto mode : {core::context::SteeringMode::Default,
                            core::context::SteeringMode::Fallback}) {
        const core::context::SteeringPolicy policy{.mode = mode};
        INFO("policy: " << policy.format());
        const auto context = make_context(primary.path(), policy,
                                          {secondary.path(), tertiary.path()});
        const auto roots = context.workspace_view().ordered_roots();
        const auto guard = core::context::SteeringGuard::for_context(context);
        CHECK(guard.empty());
        CHECK_FALSE(core::context::SteeringGuard::may_block(policy));
        CHECK(core::context::SteeringGuard::for_roots(roots, policy).empty());
        CHECK(guard.prompt_notice().empty());

        // Prompt discovery still selects just one root, without prohibiting
        // tools from reading instruction files elsewhere in the workspace.
        const auto prompt = core::context::ContextBuilder(context).with_mode("BUILD").build();
        CHECK_THAT(prompt, !Catch::Matchers::ContainsSubstring("needle tertiary"));
        CHECK_THAT(prompt, !Catch::Matchers::ContainsSubstring("NOT accessible through any tool"));
        const auto read = core::tools::with_path_visibility(
            std::make_shared<core::tools::ReadTool>())->execute(
                std::format(R"({{"path":"{}"}})", (tertiary.path() / "AGENTS.md").string()),
                context);
        CHECK_THAT(read, Catch::Matchers::ContainsSubstring("needle tertiary"));

        const auto enforcement = core::tools::SteeringEnforcement::for_context(context);
        CHECK_FALSE(enforcement.blocks_anything());
        CHECK_FALSE(enforcement.in_process_code_error().has_value());
        CHECK_FALSE(enforcement.child_policy({}).enabled());

        // An explicit --sandbox remains in effect, without added steering bans.
        core::landrun::LandrunPolicy sandbox{
            .mode = core::landrun::LandrunMode::workspace_write};
        core::landrun::add_writable_root(sandbox, primary.path());
        CHECK(enforcement.child_policy(sandbox) == sandbox);

#ifdef FILO_ENABLE_PYTHON
        core::tools::PythonInterpreterTool python;
        CHECK_THAT(python.execute(R"JSON({"code":"print(1 + 1)"})JSON", context),
                   Catch::Matchers::ContainsSubstring(R"("success":true)"));
#endif
    }
}

TEST_CASE("explicit steering restrictions still apply across workspace roots",
          "[steering][guard][enforcement][regression]") {
    SteeringWorkspace primary;
    SteeringWorkspace secondary;
    const auto context = make_context(primary.dir.path(), no_steering(),
                                      {secondary.dir.path()});
    const auto guard = core::context::SteeringGuard::for_context(context);
    CHECK(guard.blocked_reason(resolved(primary.path("AGENTS.md"))).has_value());
    CHECK(guard.blocked_reason(resolved(secondary.path("AGENTS.md"))).has_value());
    const auto enforcement = core::tools::SteeringEnforcement::for_guard(
        guard, core::tools::SteeringEnforcement::Tier::kernel);
    CHECK(enforcement.in_process_code_error().has_value());
    const auto child_policy = enforcement.child_policy({});
    CHECK(child_policy.enabled());
    CHECK_FALSE(child_policy.confines());
    CHECK(child_policy.protects_paths());
}

TEST_CASE("steering outside the workspace is nobody's business", "[steering][guard]") {
    SteeringWorkspace workspace;
    TempDir elsewhere{"elsewhere"};
    write_text(elsewhere.path() / "AGENTS.md", "needle elsewhere\n");

    const std::vector<fs::path> roots{resolved(workspace.dir.path())};
    const auto guard = core::context::SteeringGuard::for_roots(roots, no_steering());
    REQUIRE_FALSE(guard.empty());

    // Only files discovery found under the workspace roots are blocked; an
    // unrelated AGENTS.md elsewhere is a plain file.
    CHECK_FALSE(guard.blocked_reason(resolved(elsewhere.path() / "AGENTS.md")).has_value());
    CHECK_FALSE(guard.blocked_text_reason(
        "cat " + (elsewhere.path() / "AGENTS.md").string(), workspace.dir.path()).has_value());
}

// ---------------------------------------------------------------------------
// Free-form text: the shell and interpreter escape hatch
// ---------------------------------------------------------------------------

TEST_CASE("SteeringGuard reads a command the way a shell would", "[steering][guard]") {
    SteeringWorkspace workspace;
    const std::vector<fs::path> roots{resolved(workspace.dir.path())};
    const auto guard = core::context::SteeringGuard::for_roots(roots, no_steering());
    const auto& cwd = workspace.dir.path();
    REQUIRE_FALSE(guard.empty());

    const auto blocked = [&](std::string_view command) {
        CAPTURE(command);
        return guard.blocked_text_reason(command, cwd).has_value();
    };

    // Direct references, however they are spelled.
    CHECK(blocked("cat AGENTS.md"));
    CHECK(blocked("cat ./AGENTS.md"));
    CHECK(blocked("cat ../" + workspace.dir.path().filename().string() + "/AGENTS.md"));
    CHECK(blocked("head -n 20 CLAUDE.md | tail -n 5"));
    CHECK(blocked("grep -n rule AGENTS.md"));
    CHECK(blocked("sed -n '1,40p' AGENTS.md"));
    CHECK(blocked("sort < AGENTS.md"));
    CHECK(blocked(("cat " + (cwd / "AGENTS.md").string()).c_str()));
    CHECK(blocked("cat $PWD/AGENTS.md"));
    CHECK(blocked("cat ${PWD}/AGENTS.md"));
    CHECK(blocked("cat .filo/steering/style.md"));
    CHECK(blocked("ls .filo/steering"));

    // Wildcards and brace alternation cannot be used to reach the same bytes.
    CHECK(blocked("cat *.md"));
    CHECK(blocked("cat A*.md"));
    CHECK(blocked("cat .filo/steering/*"));
    CHECK(blocked("cat **/AGENTS.md"));
    CHECK(blocked("cat {AGENTS,CLAUDE}.md"));

    // Interpreters invoked from the shell are still just text here.
    CHECK(blocked("python3 -c \"print(open('AGENTS.md').read())\""));
    CHECK(blocked("node -e 'console.log(require(\"fs\").readFileSync(\"CLAUDE.md\"))'"));

    // Ordinary work is not held hostage.
    CHECK_FALSE(blocked("ls -la"));
    CHECK_FALSE(blocked("git status"));
    CHECK_FALSE(blocked("git log --oneline -n 20"));
    CHECK_FALSE(blocked("cat README.md"));
    CHECK_FALSE(blocked("grep -rn SteeringGuard src/"));
    CHECK_FALSE(blocked("cmake -S . -B cmake-build-debug && ninja -C cmake-build-debug"));
    CHECK_FALSE(blocked("cat docs/notes.md"));
    CHECK_FALSE(blocked("echo 'no steering here'"));
    CHECK_FALSE(blocked(""));
}

/**
 * The shell must not hand withheld steering back, but *how* it is stopped is a
 * host property: where the OS can subtract paths the command runs and the
 * kernel denies the file, and everywhere else Filo refuses the command after
 * reading its text. Both are correct, so the assertion is on the outcome that
 * matters — no withheld content in the result — plus a refusal that is
 * visible rather than silent.
 */
TEST_CASE("the shell cannot cat withheld steering back in", "[steering][guard][shell]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};

    core::tools::ShellTool shell;
    const auto run = [&](std::string_view command,
                         const core::context::SteeringPolicy& policy) {
        return shell.execute(
            std::format(R"({{"command":"{}"}})", command),
            make_context(workspace.dir.path(), policy));
    };

    for (const auto* command : {"cat AGENTS.md",
                                "cat CLAUDE.md",
                                "cat .filo/steering/style.md",
                                "cat *.md"}) {
        const auto result = run(command, no_steering());
        INFO("command: " << command);
        CHECK_THAT(result, !Catch::Matchers::ContainsSubstring("needle agents steering"));
        CHECK_THAT(result, !Catch::Matchers::ContainsSubstring("needle claude steering"));
        CHECK_THAT(result, !Catch::Matchers::ContainsSubstring("needle style steering"));
        // Refused up front, or run and denied by the kernel. A command that
        // merely produced no output would hide a policy that stopped working.
        CHECK((Catch::Matchers::ContainsSubstring("Access denied").match(result)
               || Catch::Matchers::ContainsSubstring("Operation not permitted").match(result)
               || Catch::Matchers::ContainsSubstring("No such file").match(result)));
    }

    // Withheld steering costs nothing else: unrelated files stay readable, and
    // everything comes back once the policy loads the steering again.
    const auto allowed = run("cat README.md", no_steering());
    REQUIRE_THAT(allowed, Catch::Matchers::ContainsSubstring("needle readme"));
    CHECK_THAT(allowed, !Catch::Matchers::ContainsSubstring("Access denied"));

    const auto steering_enabled = run("cat AGENTS.md", default_steering());
    REQUIRE_THAT(steering_enabled, Catch::Matchers::ContainsSubstring("needle agents steering"));
}

#ifdef FILO_ENABLE_PYTHON
/**
 * The interpreter runs inside Filo, which no sandbox confines, so there is no
 * tier in which its source can be trusted: while steering is withheld the tool
 * is refused outright and the sandboxed shell is the way to run code.
 */
TEST_CASE("the embedded interpreter is refused while steering is withheld",
          "[steering][guard][python]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};

    core::tools::PythonInterpreterTool python;
    const auto run = [&](std::string_view code,
                         const core::context::SteeringPolicy& policy) {
        return python.execute(
            std::format(R"JSON({{"code":"{}"}})JSON", code),
            make_context(workspace.dir.path(), policy));
    };

    const auto denied = run("print(open('AGENTS.md').read())", no_steering());
    REQUIRE_THAT(denied, Catch::Matchers::ContainsSubstring("Access denied"));
    REQUIRE_THAT(denied, !Catch::Matchers::ContainsSubstring("needle agents steering"));

    // Even code naming no file at all is refused: it shares a process with an
    // open() that no path gate can see.
    const auto innocent = run("print(1 + 1)", no_steering());
    CHECK_THAT(innocent, Catch::Matchers::ContainsSubstring("Access denied"));
    CHECK_THAT(innocent, Catch::Matchers::ContainsSubstring("run_terminal_command"));

    const auto allowed = run("print(1 + 1)", default_steering());
    CHECK_THAT(allowed, !Catch::Matchers::ContainsSubstring("Access denied"));
    CHECK_THAT(allowed, Catch::Matchers::ContainsSubstring(R"("success":true)"));
}
#endif

// ---------------------------------------------------------------------------
// Enforcement strategy: what the host can confine decides the mechanism
// ---------------------------------------------------------------------------

namespace {

using Enforcement = core::tools::SteeringEnforcement;
using Tier = Enforcement::Tier;

/// Tier injected explicitly, so the strategy is tested on every host rather
/// than only on the ones whose backend happens to support subtraction.
[[nodiscard]] Enforcement enforcement_for(const SteeringWorkspace& workspace,
                                          const core::context::SteeringPolicy& policy,
                                          Tier tier) {
    auto context = make_context(workspace.dir.path(), policy);
    return Enforcement::for_guard(
        core::context::SteeringGuard::for_context(context), tier);
}

[[nodiscard]] bool denies(const core::landrun::LandrunPolicy& policy,
                          std::string_view filename) {
    return std::ranges::any_of(policy.protected_read_paths, [&](const auto& path) {
        return path.filename() == filename;
    });
}

} // namespace

TEST_CASE("the enforcement tier follows what the policy withholds",
          "[steering][guard][enforcement]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};

    // Nothing withheld, so no tier survives construction — not even a requested
    // one. Callers rely on `none` meaning "no enforcement needed anywhere".
    const auto open = enforcement_for(workspace, default_steering(), Tier::kernel);
    CHECK(open.tier() == Tier::none);
    CHECK_FALSE(open.blocks_anything());
    CHECK_FALSE(open.command_error("cat AGENTS.md", workspace.dir.path()).has_value());
    CHECK_FALSE(open.in_process_code_error().has_value());
    CHECK(open.child_policy(core::landrun::LandrunPolicy{})
          == core::landrun::LandrunPolicy{});

    const auto blocking = enforcement_for(workspace, no_steering(), Tier::heuristic);
    CHECK(blocking.tier() == Tier::heuristic);
    CHECK(blocking.blocks_anything());
}

TEST_CASE("kernel-enforced steering leaves command text alone and confines the child",
          "[steering][guard][enforcement]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};
    const auto kernel = enforcement_for(workspace, no_steering(), Tier::kernel);

    // The point of the tier: no textual matching, so no false positives either.
    CHECK_FALSE(kernel.command_error("cat AGENTS.md", workspace.dir.path()).has_value());
    CHECK_FALSE(kernel.command_error("cat *.md", workspace.dir.path()).has_value());
    CHECK_FALSE(kernel.command_error("grep -l needle *.md", workspace.dir.path()).has_value());
    CHECK_FALSE(kernel.command_error(
        "base64 -d <<< YWdlbnRz | cat $(echo QUdFTlRTLm1k)",
        workspace.dir.path()).has_value());

    // The blocked set becomes protected paths, which is what the OS denies.
    core::landrun::LandrunPolicy policy{
        .mode = core::landrun::LandrunMode::workspace_write};
    core::landrun::add_writable_root(policy, workspace.dir.path());
    const auto confined = kernel.child_policy(policy);
    CHECK(confined.protects_paths());
    CHECK(confined.enabled());
    CHECK(denies(confined, "AGENTS.md"));
    CHECK(denies(confined, "CLAUDE.md"));
    CHECK(denies(confined, "style.md"));

    // Confinement the user asked for is preserved, not replaced.
    CHECK(confined.confines());
    CHECK(confined.writable_roots == policy.writable_roots);

    // In-process code has no kernel behind it, so it stays refused.
    CHECK(kernel.in_process_code_error().has_value());
}

TEST_CASE("heuristic-enforced steering refuses commands and leaves the policy alone",
          "[steering][guard][enforcement]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};
    const auto heuristic = enforcement_for(workspace, no_steering(), Tier::heuristic);

    const auto denied = heuristic.command_error("cat AGENTS.md", workspace.dir.path());
    REQUIRE(denied.has_value());
    CHECK_THAT(*denied, Catch::Matchers::ContainsSubstring("Access denied"));
    CHECK_THAT(*denied, Catch::Matchers::ContainsSubstring("AGENTS.md"));
    CHECK_FALSE(heuristic.command_error("cat README.md", workspace.dir.path()).has_value());

    // A backend that cannot subtract paths fails closed on them, so composing
    // protected paths here would break every command instead of protecting
    // anything. The fallback must hand back the policy exactly as compiled.
    core::landrun::LandrunPolicy policy{
        .mode = core::landrun::LandrunMode::workspace_write};
    core::landrun::add_writable_root(policy, workspace.dir.path());
    CHECK(heuristic.child_policy(policy) == policy);

    CHECK(heuristic.in_process_code_error().has_value());
}


// ---------------------------------------------------------------------------
// The tool gate: every path-taking tool, and tree traversal
// ---------------------------------------------------------------------------

TEST_CASE("check_workspace_access denies withheld steering for reads and writes",
          "[steering][guard][tools]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};

    const auto check = [&](std::string_view relative,
                           std::string_view tool,
                           const core::context::SteeringPolicy& policy) {
        const auto context = make_context(workspace.dir.path(), policy);
        return core::tools::detail::check_workspace_access(
            fs::path(relative), std::string(relative), context, nullptr, tool);
    };

    using core::tools::names::kRead;
    using core::tools::names::kWriteFile;

    CHECK(check("AGENTS.md", kRead, no_steering()).has_value());
    CHECK(check("AGENTS.md", kWriteFile, no_steering()).has_value());
    CHECK(check(".filo/steering/style.md", kRead, no_steering()).has_value());
    CHECK_FALSE(check("README.md", kRead, no_steering()).has_value());
    CHECK_FALSE(check("AGENTS.md", kRead, default_steering()).has_value());

    const auto denial = check("AGENTS.md", kRead, no_steering());
    REQUIRE(denial.has_value());
    REQUIRE_THAT(*denial, Catch::Matchers::ContainsSubstring("\"error\""));
}

TEST_CASE("no-steering hides withheld files from every reading tool",
          "[steering][guard][tools]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};
    const auto context = make_context(workspace.dir.path(), no_steering());

    const auto read_tool =
        core::tools::with_path_visibility(std::make_shared<core::tools::ReadTool>());
    const auto read = read_tool->execute(R"({"path":"AGENTS.md"})", context);
    REQUIRE_THAT(read, Catch::Matchers::ContainsSubstring("Access denied"));
    REQUIRE_THAT(read, !Catch::Matchers::ContainsSubstring("needle agents"));

    const auto write_tool =
        core::tools::with_path_visibility(std::make_shared<core::tools::WriteFileTool>());
    const auto write = write_tool->execute(
        R"({"file_path":"AGENTS.md","content":"rewritten"})", context);
    REQUIRE_THAT(write, Catch::Matchers::ContainsSubstring("Access denied"));
    REQUIRE_THAT(
        core::tools::with_path_visibility(std::make_shared<core::tools::ReadTool>())
            ->execute(R"({"path":"AGENTS.md"})",
                      make_context(workspace.dir.path(), default_steering())),
        Catch::Matchers::ContainsSubstring("needle agents steering"));

    const auto listing =
        core::tools::with_path_visibility(std::make_shared<core::tools::ListDirectoryTool>())
            ->execute(R"({"path":"."})", context);
    REQUIRE_THAT(listing, Catch::Matchers::ContainsSubstring("README.md"));
    CHECK_THAT(listing, !Catch::Matchers::ContainsSubstring("AGENTS.md"));
    CHECK_THAT(listing, !Catch::Matchers::ContainsSubstring("CLAUDE.md"));

    const auto steering_listing =
        core::tools::with_path_visibility(std::make_shared<core::tools::ListDirectoryTool>())
            ->execute(R"({"path":".filo/steering"})", context);
    CHECK_THAT(steering_listing, Catch::Matchers::ContainsSubstring("Access denied"));

    // Traversal is the bulk leak: grep returns matching lines, glob returns
    // names, and neither asks the per-path gate about each entry.
    const auto matches =
        core::tools::with_path_visibility(std::make_shared<core::tools::GrepSearchTool>())
            ->execute(R"({"pattern":"needle","path":"."})", context);
    REQUIRE_THAT(matches, Catch::Matchers::ContainsSubstring("needle readme"));
    CHECK_THAT(matches, !Catch::Matchers::ContainsSubstring("needle agents"));
    CHECK_THAT(matches, !Catch::Matchers::ContainsSubstring("needle style"));

    const auto files =
        core::tools::with_path_visibility(std::make_shared<core::tools::FileSearchTool>())
            ->execute(R"({"pattern":"*.md","path":"."})", context);
    REQUIRE_THAT(files, Catch::Matchers::ContainsSubstring("README.md"));
    CHECK_THAT(files, !Catch::Matchers::ContainsSubstring("AGENTS.md"));
    CHECK_THAT(files, !Catch::Matchers::ContainsSubstring("style.md"));

    // A grep aimed straight at the file is refused rather than silently empty.
    const auto targeted =
        core::tools::with_path_visibility(std::make_shared<core::tools::GrepSearchTool>())
            ->execute(R"({"pattern":"needle","path":"AGENTS.md"})", context);
    CHECK_THAT(targeted, Catch::Matchers::ContainsSubstring("Access denied"));
}

TEST_CASE("the same tools read steering normally when the policy loads it",
          "[steering][guard][tools]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};
    const auto context = make_context(workspace.dir.path(), default_steering());

    const auto read =
        core::tools::with_path_visibility(std::make_shared<core::tools::ReadTool>())
            ->execute(R"({"path":"AGENTS.md"})", context);
    REQUIRE_THAT(read, Catch::Matchers::ContainsSubstring("needle agents steering"));

    const auto listing =
        core::tools::with_path_visibility(std::make_shared<core::tools::ListDirectoryTool>())
            ->execute(R"({"path":"."})", context);
    REQUIRE_THAT(listing, Catch::Matchers::ContainsSubstring("AGENTS.md"));

    const auto matches =
        core::tools::with_path_visibility(std::make_shared<core::tools::GrepSearchTool>())
            ->execute(R"({"pattern":"needle","path":"."})", context);
    REQUIRE_THAT(matches, Catch::Matchers::ContainsSubstring("needle agents"));
}

// ---------------------------------------------------------------------------
// The model is told, so it does not spend turns finding out
// ---------------------------------------------------------------------------

TEST_CASE("the prompt states which steering files are unreadable", "[steering][guard][context]") {
    SteeringWorkspace workspace;
    ScopedWorkspace scoped{workspace.dir.path()};

    const auto notice =
        core::context::SteeringGuard::for_context(make_context(workspace.dir.path(), no_steering()))
            .prompt_notice();
    REQUIRE_FALSE(notice.empty());
    REQUIRE_THAT(notice, Catch::Matchers::ContainsSubstring("AGENTS.md"));
    REQUIRE_THAT(notice, Catch::Matchers::ContainsSubstring(".filo/steering/style.md"));
    REQUIRE_THAT(notice, Catch::Matchers::ContainsSubstring("NOT accessible through any tool"));

    CHECK(core::context::SteeringGuard::for_context(
              make_context(workspace.dir.path(), default_steering()))
              .prompt_notice()
              .empty());

    const auto prompt = core::context::ContextBuilder(
                            make_context(workspace.dir.path(), no_steering()))
                            .with_mode("BUILD")
                            .build();
    REQUIRE_THAT(prompt, Catch::Matchers::ContainsSubstring("[Project Steering]"));
    REQUIRE_THAT(prompt, Catch::Matchers::ContainsSubstring("NOT accessible through any tool"));
    // The withheld content itself must not appear anywhere in the prompt.
    CHECK_THAT(prompt, !Catch::Matchers::ContainsSubstring("needle agents"));

    const auto enabled_prompt = core::context::ContextBuilder(
                                    make_context(workspace.dir.path(), default_steering()))
                                    .with_mode("BUILD")
                                    .build();
    REQUIRE_THAT(enabled_prompt, Catch::Matchers::ContainsSubstring("needle agents steering"));
    CHECK_THAT(
        enabled_prompt,
        !Catch::Matchers::ContainsSubstring("NOT accessible through any tool"));
}
