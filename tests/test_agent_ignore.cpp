#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/ContextMentions.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/PathVisibilityToolDecorator.hpp"
#include "core/tools/ReadFileTool.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/workspace/AgentIgnore.hpp"
#include "core/workspace/PathVisibility.hpp"
#include "core/workspace/Workspace.hpp"
#include "tui/Autocomplete.hpp"
#include "TestSessionContext.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <ranges>
#include <string>

namespace fs = std::filesystem;

namespace {

class TempDir {
public:
    explicit TempDir(std::string_view label)
        : path_(fs::temp_directory_path()
                / std::format(
                    "{}_{}",
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
    std::ofstream out(path);
    out << content;
}

[[nodiscard]] core::context::SessionContext make_context() {
    return test_support::make_workspace_session_context(
        core::context::SessionTransport::cli);
}

} // namespace

TEST_CASE("PathVisibility composes custom decorators",
          "[agentignore][visibility]") {
    class SuffixVisibilityDecorator final
        : public core::workspace::PathVisibilityDecorator {
    public:
        SuffixVisibilityDecorator(
            std::shared_ptr<const core::workspace::IPathVisibilityPolicy> wrapped,
            std::string suffix)
            : core::workspace::PathVisibilityDecorator(std::move(wrapped))
            , suffix_(std::move(suffix)) {}

        [[nodiscard]] bool empty() const noexcept override {
            return suffix_.empty() && PathVisibilityDecorator::empty();
        }

        [[nodiscard]] std::optional<std::string> hidden_reason(
            std::string_view display_path,
            const std::filesystem::path& path,
            core::workspace::PathVisibilityKind kind) const override {
            if (const auto upstream =
                    PathVisibilityDecorator::hidden_reason(display_path, path, kind)) {
                return upstream;
            }
            if (kind != core::workspace::PathVisibilityKind::File
                || !path.filename().string().ends_with(suffix_)) {
                return std::nullopt;
            }
            return std::format("Path '{}' is excluded by test visibility policy", display_path);
        }

    private:
        std::string suffix_;
    };

    const core::workspace::PathVisibility visibility(
        std::make_shared<SuffixVisibilityDecorator>(
            std::make_shared<core::workspace::AllowAllPathVisibilityPolicy>(),
            ".blocked"));

    CHECK_FALSE(visibility.is_hidden("visible.txt"));
    CHECK(visibility.is_hidden("secret.blocked"));
    REQUIRE(visibility.hidden_reason("secret.blocked", "secret.blocked").has_value());
}

TEST_CASE(".agentignore matcher supports gitignore-style path rules",
          "[agentignore]") {
    TempDir tmp{"filo_agentignore_matcher"};
    write_text(
        tmp.path() / ".agentignore",
        R"(# comments are ignored
*.env
secrets/
!secrets/public.env
build-output/
/root-only.txt
docs/**/draft?.md
\#literal
)");

    write_text(tmp.path() / ".env", "secret");
    write_text(tmp.path() / "src" / "app.env", "secret");
    write_text(tmp.path() / "secrets" / "private.txt", "secret");
    write_text(tmp.path() / "secrets" / "public.env", "public");
    write_text(tmp.path() / "build-output" / "generated.txt", "generated");
    write_text(tmp.path() / "root-only.txt", "secret");
    write_text(tmp.path() / "nested" / "root-only.txt", "ok");
    write_text(tmp.path() / "docs" / "draft1.md", "draft");
    write_text(tmp.path() / "docs" / "deep" / "draft2.md", "draft");
    write_text(tmp.path() / "#literal", "secret");

    const auto matcher = core::workspace::AgentIgnoreMatcher::load_for_root(tmp.path());

    CHECK(matcher.is_ignored(tmp.path() / ".env", core::workspace::AgentIgnorePathKind::File));
    CHECK(matcher.is_ignored(
        tmp.path() / "src" / "app.env",
        core::workspace::AgentIgnorePathKind::File));
    CHECK(matcher.is_ignored(
        tmp.path() / "secrets" / "private.txt",
        core::workspace::AgentIgnorePathKind::File));
    CHECK_FALSE(matcher.is_ignored(
        tmp.path() / "secrets" / "public.env",
        core::workspace::AgentIgnorePathKind::File));
    CHECK_FALSE(matcher.can_prune_directory(tmp.path() / "secrets"));
    CHECK(matcher.can_prune_directory(tmp.path() / "build-output"));
    CHECK(matcher.is_ignored(
        tmp.path() / "root-only.txt",
        core::workspace::AgentIgnorePathKind::File));
    CHECK_FALSE(matcher.is_ignored(
        tmp.path() / "nested" / "root-only.txt",
        core::workspace::AgentIgnorePathKind::File));
    CHECK(matcher.is_ignored(
        tmp.path() / "docs" / "draft1.md",
        core::workspace::AgentIgnorePathKind::File));
    CHECK(matcher.is_ignored(
        tmp.path() / "docs" / "deep" / "draft2.md",
        core::workspace::AgentIgnorePathKind::File));
    CHECK(matcher.is_ignored(
        tmp.path() / "#literal",
        core::workspace::AgentIgnorePathKind::File));
}

TEST_CASE("absent .agentignore leaves standard workspace access unchanged",
          "[agentignore][regression]") {
    TempDir tmp{"filo_agentignore_absent"};
    ScopedWorkspace workspace{tmp.path()};

    write_text(tmp.path() / "visible.txt", "needle visible\n");
    write_text(tmp.path() / "secrets" / "hidden.txt", "needle hidden\n");
    write_text(tmp.path() / ".env", "needle env\n");

    const auto read_tool =
        core::tools::with_path_visibility(std::make_shared<core::tools::ReadFileTool>());
    const auto read_hidden = read_tool->execute(
        R"({"path":"secrets/hidden.txt"})",
        make_context());
    REQUIRE_THAT(read_hidden, Catch::Matchers::ContainsSubstring("needle hidden"));

    const auto file_search =
        core::tools::with_path_visibility(std::make_shared<core::tools::FileSearchTool>());
    const auto files = file_search->execute(
        R"({"pattern":"*.txt","path":"."})",
        make_context());
    REQUIRE_THAT(files, Catch::Matchers::ContainsSubstring("visible.txt"));
    REQUIRE_THAT(files, Catch::Matchers::ContainsSubstring("hidden.txt"));

    const auto grep_search =
        core::tools::with_path_visibility(std::make_shared<core::tools::GrepSearchTool>());
    const auto matches = grep_search->execute(
        R"({"pattern":"needle","path":"."})",
        make_context());
    REQUIRE_THAT(matches, Catch::Matchers::ContainsSubstring("needle visible"));
    REQUIRE_THAT(matches, Catch::Matchers::ContainsSubstring("needle hidden"));
    REQUIRE_THAT(matches, Catch::Matchers::ContainsSubstring("needle env"));

    const auto list_directory =
        core::tools::with_path_visibility(std::make_shared<core::tools::ListDirectoryTool>());
    const auto listing = list_directory->execute(
        R"({"path":"."})",
        make_context());
    REQUIRE_THAT(listing, Catch::Matchers::ContainsSubstring("visible.txt"));
    REQUIRE_THAT(listing, Catch::Matchers::ContainsSubstring("secrets"));
    REQUIRE_THAT(listing, Catch::Matchers::ContainsSubstring(".env"));

    const auto expanded_hidden = core::context::expand_mentions(
        "Inspect @secrets/hidden.txt",
        tmp.path());
    REQUIRE_THAT(expanded_hidden, Catch::Matchers::ContainsSubstring("needle hidden"));

    const auto suggestions = tui::build_mention_index(tmp.path());
    const auto hidden = std::ranges::any_of(suggestions, [](const auto& suggestion) {
        return suggestion.display_path.find("hidden.txt") != std::string::npos;
    });
    const auto env = std::ranges::any_of(suggestions, [](const auto& suggestion) {
        return suggestion.display_path == ".env";
    });
    CHECK(hidden);
    CHECK(env);
}

TEST_CASE(".agentignore blocks direct reads and filters search tools",
          "[agentignore][tools]") {
    TempDir tmp{"filo_agentignore_tools"};
    TempDir external{"filo_agentignore_external"};
    ScopedWorkspace workspace{tmp.path()};

    write_text(tmp.path() / ".agentignore", "secrets/\n!secrets/public.txt\n*.env\n");
    write_text(tmp.path() / "visible.txt", "needle visible\n");
    write_text(tmp.path() / "secrets" / "hidden.txt", "needle hidden\n");
    write_text(tmp.path() / "secrets" / "public.txt", "needle public\n");
    write_text(tmp.path() / ".env", "needle env\n");
    write_text(external.path() / "outside.txt", "needle outside\n");

    std::error_code symlink_ec;
    std::filesystem::create_symlink(
        tmp.path() / "secrets" / "hidden.txt",
        tmp.path() / "visible-link.txt",
        symlink_ec);
    if (!symlink_ec) {
        std::filesystem::create_symlink(
            external.path() / "outside.txt",
            tmp.path() / "outside-link.txt",
            symlink_ec);
    }

    const auto read_tool =
        core::tools::with_path_visibility(std::make_shared<core::tools::ReadFileTool>());
    const auto read_hidden = read_tool->execute(
        R"({"path":"secrets/hidden.txt"})",
        make_context());
    REQUIRE_THAT(read_hidden, Catch::Matchers::ContainsSubstring(".agentignore"));

    const auto read_visible = read_tool->execute(
        R"({"path":"visible.txt"})",
        make_context());
    REQUIRE_THAT(read_visible, Catch::Matchers::ContainsSubstring("needle visible"));

    const auto file_search =
        core::tools::with_path_visibility(std::make_shared<core::tools::FileSearchTool>());
    const auto files = file_search->execute(
        R"({"pattern":"*.txt","path":"."})",
        make_context());
    REQUIRE_THAT(files, Catch::Matchers::ContainsSubstring("visible.txt"));
    REQUIRE_THAT(files, Catch::Matchers::ContainsSubstring("public.txt"));
    REQUIRE_THAT(files, !Catch::Matchers::ContainsSubstring("hidden.txt"));
    REQUIRE_THAT(files, !Catch::Matchers::ContainsSubstring("visible-link.txt"));

    const auto grep_search =
        core::tools::with_path_visibility(std::make_shared<core::tools::GrepSearchTool>());
    const auto matches = grep_search->execute(
        R"({"pattern":"needle","path":"."})",
        make_context());
    REQUIRE_THAT(matches, Catch::Matchers::ContainsSubstring("needle visible"));
    REQUIRE_THAT(matches, Catch::Matchers::ContainsSubstring("needle public"));
    REQUIRE_THAT(matches, !Catch::Matchers::ContainsSubstring("needle hidden"));
    REQUIRE_THAT(matches, !Catch::Matchers::ContainsSubstring("needle env"));
    REQUIRE_THAT(matches, !Catch::Matchers::ContainsSubstring("needle outside"));
}

TEST_CASE(".agentignore does not disable open-world local execution tools",
          "[agentignore][tools]") {
    TempDir tmp{"filo_agentignore_open_world"};
    ScopedWorkspace workspace{tmp.path()};

    write_text(tmp.path() / ".agentignore", "secrets/\n");
    write_text(tmp.path() / "secrets" / "hidden.txt", "needle hidden\n");

    core::tools::ShellTool shell_tool;
    const auto shell_result = shell_tool.execute(
        R"({"command":"cat secrets/hidden.txt","working_dir":")"
            + tmp.path().string()
            + R"("})",
        make_context());
    REQUIRE_THAT(shell_result, Catch::Matchers::ContainsSubstring("needle hidden"));
}

TEST_CASE(".agentignore blocks mutating tools for ignored paths",
          "[agentignore][tools]") {
    TempDir tmp{"filo_agentignore_mutating"};
    ScopedWorkspace workspace{tmp.path()};

    write_text(tmp.path() / ".agentignore", "secrets/\n");
    write_text(tmp.path() / "secrets" / "hidden.txt", "original secret\n");

    const auto write_file =
        core::tools::with_path_visibility(std::make_shared<core::tools::WriteFileTool>());
    const auto write_result = write_file->execute(
        R"({"file_path":"secrets/hidden.txt","content":"new secret\n"})",
        make_context());

    REQUIRE_THAT(write_result, Catch::Matchers::ContainsSubstring(".agentignore"));
    REQUIRE_THAT(write_result, !Catch::Matchers::ContainsSubstring("original secret"));

    const auto read_tool =
        core::tools::with_path_visibility(std::make_shared<core::tools::ReadFileTool>());
    const auto visible_from_raw_fs = [&] {
        std::ifstream in(tmp.path() / "secrets" / "hidden.txt");
        std::string content;
        std::getline(in, content);
        return content;
    }();
    CHECK(visible_from_raw_fs == "original secret");
    REQUIRE_THAT(
        read_tool->execute(R"({"path":"secrets/hidden.txt"})", make_context()),
        Catch::Matchers::ContainsSubstring(".agentignore"));
}

TEST_CASE(".agentignore filters directory listings, mentions, and autocomplete",
          "[agentignore][context]") {
    TempDir tmp{"filo_agentignore_context"};
    ScopedWorkspace workspace{tmp.path()};

    write_text(tmp.path() / ".agentignore", "secrets/\n*.env\n");
    write_text(tmp.path() / "visible.txt", "visible context\n");
    write_text(tmp.path() / "secrets" / "hidden.txt", "hidden context\n");
    write_text(tmp.path() / ".env", "env context\n");

    const auto list_directory =
        core::tools::with_path_visibility(std::make_shared<core::tools::ListDirectoryTool>());
    const auto listing = list_directory->execute(
        R"({"path":"."})",
        make_context());
    REQUIRE_THAT(listing, Catch::Matchers::ContainsSubstring("visible.txt"));
    REQUIRE_THAT(listing, !Catch::Matchers::ContainsSubstring("secrets"));
    REQUIRE_THAT(listing, !Catch::Matchers::ContainsSubstring(".env"));

    const auto expanded_hidden = core::context::expand_mentions(
        "Inspect @secrets/hidden.txt",
        tmp.path());
    REQUIRE_THAT(expanded_hidden, Catch::Matchers::ContainsSubstring(".agentignore"));
    REQUIRE_THAT(expanded_hidden, !Catch::Matchers::ContainsSubstring("hidden context"));

    const auto expanded_visible = core::context::expand_mentions(
        "Inspect @visible.txt",
        tmp.path());
    REQUIRE_THAT(expanded_visible, Catch::Matchers::ContainsSubstring("visible context"));

    const auto suggestions = tui::build_mention_index(tmp.path());
    const auto visible = std::ranges::any_of(suggestions, [](const auto& suggestion) {
        return suggestion.display_path == "visible.txt";
    });
    const auto hidden = std::ranges::any_of(suggestions, [](const auto& suggestion) {
        return suggestion.display_path.find("hidden.txt") != std::string::npos
            || suggestion.display_path == ".env";
    });
    CHECK(visible);
    CHECK_FALSE(hidden);
}
