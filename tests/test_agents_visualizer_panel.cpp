#include "core/context/SteeringLoader.hpp"
#include "tui/AgentsVisualizerPanel.hpp"
#include "tui/PromptComponents.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <ftxui/dom/elements.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

class TempDir {
public:
    explicit TempDir(std::filesystem::path path)
        : path_(std::move(path)) {
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void write_text(const std::filesystem::path& path, std::string_view content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << content;
}

[[nodiscard]] TempDir make_temp_workspace(std::string_view name) {
    const auto base = std::filesystem::temp_directory_path()
        / std::format("{}_{}", name, std::chrono::steady_clock::now().time_since_epoch().count());
    return TempDir{base};
}

} // namespace

TEST_CASE("SteeringLoader: populates individual SteeringFile entries", "[context][steering]") {
    auto workspace = make_temp_workspace("filo_steering_files");
    write_text(workspace.path() / "AGENTS.md", "# Project Rules\nAlways write clean C++.\n");
    write_text(workspace.path() / "FILO.md", "# Filo Rules\nPreserve comments.\n");
    write_text(workspace.path() / ".filo" / "steering" / "rules.md", "## Extra Rules\nUse RAII.\n");

    const auto result = core::context::load_project_steering_context(workspace.path());

    REQUIRE(result.files.size() == 3);
    CHECK(result.files[0].label == "AGENTS.md");
    CHECK(std::filesystem::equivalent(result.files[0].path, workspace.path() / "AGENTS.md"));
    CHECK_THAT(result.files[0].content, Catch::Matchers::ContainsSubstring("Always write clean C++"));

    CHECK(result.files[1].label == "FILO.md");
    CHECK_THAT(result.files[1].content, Catch::Matchers::ContainsSubstring("Preserve comments"));

    CHECK(result.files[2].label == ".filo/steering/rules.md");
    CHECK_THAT(result.files[2].content, Catch::Matchers::ContainsSubstring("Use RAII"));
}

TEST_CASE("AgentsVisualizerPanel: renders empty state when no files exist", "[agents_visualizer]") {
    std::vector<core::context::SteeringFile> files;
    std::vector<ftxui::Box> tab_hitboxes;

    auto element = tui::render_agents_visualizer_panel(files, 0, 0, &tab_hitboxes);
    CHECK(element != nullptr);
    CHECK(tab_hitboxes.empty());
}

TEST_CASE("AgentsVisualizerPanel: renders single file without multi-tabs", "[agents_visualizer]") {
    std::vector<core::context::SteeringFile> files = {
        core::context::SteeringFile{
            .path = "/test/AGENTS.md",
            .label = "AGENTS.md",
            .content = "# Guidelines\n- Rule 1\n- Rule 2\n",
        }
    };
    std::vector<ftxui::Box> tab_hitboxes;

    auto element = tui::render_agents_visualizer_panel(files, 0, 0, &tab_hitboxes);
    CHECK(element != nullptr);
    CHECK(tab_hitboxes.empty());
}

TEST_CASE("AgentsVisualizerPanel: renders multiple files and sets tab hitboxes", "[agents_visualizer]") {
    std::vector<core::context::SteeringFile> files = {
        core::context::SteeringFile{
            .path = "/test/AGENTS.md",
            .label = "AGENTS.md",
            .content = "# Guidelines\n- Rule 1\n",
        },
        core::context::SteeringFile{
            .path = "/test/.filo/steering/backend.md",
            .label = ".filo/steering/backend.md",
            .content = "# Backend Guidelines\n- C++26\n",
        }
    };
    std::vector<ftxui::Box> tab_hitboxes;

    auto element = tui::render_agents_visualizer_panel(files, 1, 0, &tab_hitboxes);
    CHECK(element != nullptr);
    CHECK(tab_hitboxes.size() == 2);
}

TEST_CASE("AgentsVisualizerPanel: handles scroll offsets gracefully", "[agents_visualizer]") {
    std::string long_content;
    for (int i = 1; i <= 50; ++i) {
        long_content += std::format("Line {}\n", i);
    }

    std::vector<core::context::SteeringFile> files = {
        core::context::SteeringFile{
            .path = "/test/AGENTS.md",
            .label = "AGENTS.md",
            .content = long_content,
        }
    };

    // Scroll at top (offset 0) with custom line count
    auto el_top = tui::render_agents_visualizer_panel(files, 0, 0, nullptr, 20);
    CHECK(el_top != nullptr);

    // Scroll in middle (offset 10)
    auto el_mid = tui::render_agents_visualizer_panel(files, 0, 10, nullptr, 20);
    CHECK(el_mid != nullptr);

    // Out-of-bounds scroll offset clamped safely
    auto el_out = tui::render_agents_visualizer_panel(files, 99, 1000, nullptr, 20);
    CHECK(el_out != nullptr);
}

TEST_CASE("PromptComponents: render_startup_banner_panel supports context_sources_hitbox", "[tui][banner]") {
    ftxui::Box hitbox{0, -1, 0, -1};
    auto banner = tui::render_startup_banner_panel(
        "anthropic",
        "claude-3-7-sonnet",
        2,
        "AGENTS.md",
        "",
        "12:00:00",
        {},
        nullptr,
        &hitbox);
    CHECK(banner != nullptr);
}
