#include "core/context/SteeringLoader.hpp"
#include "tui/AgentsVisualizerPanel.hpp"
#include "tui/PromptComponents.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <chrono>
#include <format>
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

namespace {

[[nodiscard]] std::string strip_sgr(const std::string& raw) {
    std::string out;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\x1b' && i + 1 < raw.size() && raw[i + 1] == '[') {
            i += 2;
            while (i < raw.size() && !(raw[i] >= '@' && raw[i] <= '~')) ++i;
            continue;
        }
        out.push_back(raw[i]);
    }
    return out;
}

[[nodiscard]] std::vector<core::context::SteeringFile> numbered_files(std::size_t count, int lines) {
    std::vector<core::context::SteeringFile> files;
    for (std::size_t f = 0; f < count; ++f) {
        std::string content;
        for (int i = 1; i <= lines; ++i) content += std::format("Line {}\n", i);
        files.push_back({.path = std::format("/test/{}.md", f),
                         .label = std::format("{}.md", f),
                         .content = std::move(content)});
    }
    return files;
}

[[nodiscard]] ftxui::Event mouse_event(ftxui::Mouse::Button button, int x, int y) {
    ftxui::Mouse mouse;
    mouse.button = button;
    mouse.motion = ftxui::Mouse::Pressed;
    mouse.x = x;
    mouse.y = y;
    return ftxui::Event::Mouse("", mouse);
}

} // namespace

TEST_CASE("Startup banner draws the context sources as a popover trigger",
          "[tui][banner][agents_visualizer]") {
    const std::vector<tui::ThreadTab> one_thread{};
    const std::vector<tui::ThreadTab> two_threads{{.label = "main", .active = true}, {.label = "other"}};
    for (const auto* tabs : {&one_thread, &two_threads}) {
        for (const bool open : {false, true}) {
            ftxui::Box hitbox{0, -1, 0, -1};
            auto banner = tui::render_startup_banner_panel(
                "anthropic", "claude", 2, "AGENTS.md", "", "12:00:00", *tabs, nullptr,
                tui::PopoverTrigger{.hitbox = &hitbox, .open = open});
            auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(120),
                                                ftxui::Dimension::Fit(banner));
            ftxui::Render(screen, banner);
            const std::string label = open ? "AGENTS.md ▴" : "AGENTS.md ▾";
            CHECK_THAT(strip_sgr(screen.ToString()), Catch::Matchers::ContainsSubstring(label));
            // "🤖 AGENTS.md ▾": the emoji takes two cells, the padding none.
            CHECK(hitbox.x_max - hitbox.x_min + 1 == 14);
            // The label sits at the right of the banner, so its popover is
            // right-aligned with it.
            CHECK(hitbox.x_max > 100);
        }
    }
}

TEST_CASE("Agents popover scrolls within the file and switches files",
          "[tui][agents_visualizer]") {
    using Result = tui::AgentsVisualizerEventResult;
    const auto files = numbered_files(2, 30);
    const ftxui::Box popover{10, 60, 5, 30};
    const std::vector<ftxui::Box> tabs{{20, 25, 7, 7}, {27, 32, 7, 7}};
    tui::AgentsVisualizerPanelState state{.active = true};
    const auto send = [&](ftxui::Event event) {
        return tui::handle_agents_visualizer_panel_event(state, event, files, 10, popover, tabs);
    };

    CHECK(send(ftxui::Event::ArrowUp) == Result::handled);
    CHECK(state.scroll_offset == 0);
    CHECK(send(ftxui::Event::PageDown) == Result::handled);
    CHECK(state.scroll_offset == 10);
    CHECK(send(ftxui::Event::End) == Result::handled);
    CHECK(state.scroll_offset == 20);
    // The offset stops at the last page, so scrolling back answers at once.
    CHECK(send(ftxui::Event::ArrowDown) == Result::handled);
    CHECK(state.scroll_offset == 20);
    CHECK(send(ftxui::Event::ArrowUp) == Result::handled);
    CHECK(state.scroll_offset == 19);

    CHECK(send(ftxui::Event::ArrowRight) == Result::handled);
    CHECK(state.selected_file == 1);
    CHECK(state.scroll_offset == 0);
    CHECK(send(ftxui::Event::ArrowRight) == Result::handled);
    CHECK(state.selected_file == 0);
    CHECK(send(mouse_event(ftxui::Mouse::Left, 28, 7)) == Result::handled);
    CHECK(state.selected_file == 1);

    CHECK(send(ftxui::Event::Character('u')) == Result::toggle_selected_file);
    CHECK(send(ftxui::Event::Character('x')) == Result::handled);
    CHECK(send(ftxui::Event::Custom) == Result::ignored);
    CHECK(state.active);

    CHECK(send(ftxui::Event::Escape) == Result::handled);
    CHECK_FALSE(state.active);
    CHECK(send(ftxui::Event::Character('x')) == Result::ignored);
}

TEST_CASE("Agents popover owns the mouse only inside its box",
          "[tui][agents_visualizer]") {
    using Result = tui::AgentsVisualizerEventResult;
    const auto files = numbered_files(1, 30);
    const ftxui::Box popover{10, 60, 5, 30};
    tui::AgentsVisualizerPanelState state{.active = true};
    const auto send = [&](ftxui::Event event) {
        return tui::handle_agents_visualizer_panel_event(state, event, files, 10, popover, {});
    };

    CHECK(send(mouse_event(ftxui::Mouse::WheelDown, 20, 10)) == Result::handled);
    CHECK(state.scroll_offset == 3);
    // The wheel outside keeps scrolling the transcript under it.
    CHECK(send(mouse_event(ftxui::Mouse::WheelDown, 80, 10)) == Result::ignored);
    CHECK(state.scroll_offset == 3);
    CHECK(state.active);
    // A press outside closes the popover and still reaches its target.
    CHECK(send(mouse_event(ftxui::Mouse::Left, 80, 10)) == Result::ignored);
    CHECK_FALSE(state.active);
}

TEST_CASE("Agents popover fits the rows under its trigger",
          "[tui][agents_visualizer][popover]") {
    for (const std::size_t count : {std::size_t{1}, std::size_t{2}}) {
        const auto files = numbered_files(count, 200);
        const int available_rows = 30;
        const int page = tui::agents_visualizer_page_lines(count, available_rows);
        auto panel = tui::render_agents_visualizer_panel(files, 0, 50, nullptr, page);
        auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(100),
                                            ftxui::Dimension::Fit(panel));
        ftxui::Render(screen, panel);
        CHECK(screen.dimy() <= available_rows);
        const auto output = strip_sgr(screen.ToString());
        CHECK_THAT(output, Catch::Matchers::ContainsSubstring("more lines above"));
        CHECK_THAT(output, Catch::Matchers::ContainsSubstring("more lines below"));
        CHECK_THAT(output, Catch::Matchers::ContainsSubstring("click outside to close"));
    }
}
