#include "tui/AnchoredPopover.hpp"
#include "tui/McpServersPanel.hpp"
#include "tui/PromptComponents.hpp"
#include "core/mcp/McpDynamicTool.hpp"
#include "core/mcp/McpServerStatus.hpp"
#include "TestSessionContext.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <ftxui/screen/screen.hpp>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using core::mcp::McpServerState;
using core::mcp::McpServerStatus;

namespace {

std::string strip_sgr(const std::string& raw);

std::string render(const std::vector<McpServerStatus>& servers,
                   std::size_t selected = 0,
                   std::chrono::system_clock::time_point now = std::chrono::system_clock::now(),
                   int width = 120) {
    auto element = tui::render_mcp_servers_panel(servers, selected, now, 40);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fit(element));
    ftxui::Render(screen, element);
    return strip_sgr(screen.ToString());
}

std::string strip_sgr(const std::string& raw) {
    // Drop SGR sequences so assertions see the text a user reads.
    std::string plain;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\x1b' && i + 1 < raw.size() && raw[i + 1] == '[') {
            i += 2;
            while (i < raw.size() && raw[i] != 'm') ++i;
            continue;
        }
        plain.push_back(raw[i]);
    }
    return plain;
}

McpServerStatus connected_linear(std::chrono::system_clock::time_point now) {
    McpServerStatus server{
        .name = "linear",
        .transport = "http",
        .endpoint = "https://mcp.linear.app/mcp",
        .auth = "OAuth",
        .state = McpServerState::connected,
        .server_info = {.name = "Linear MCP", .version = "1.2.0", .protocol_version = "2025-11-25"},
        .tools = {"list_issues", "get_issue", "save_issue"},
        .resources = true,
        .connected_at = now - std::chrono::minutes(18),
        .connect_duration = std::chrono::milliseconds(820),
    };
    server.activity.calls = 4;
    server.activity.failures = 1;
    server.activity.last_tool = "list_issues";
    server.activity.last_call_at = now - std::chrono::minutes(2);
    server.activity.last_duration = std::chrono::milliseconds(340);
    return server;
}

class FakeSession final : public core::mcp::IMcpClientSession {
public:
    std::string reply = R"({"result":"ok"})";
    bool fail = false;

    std::vector<core::mcp::McpToolDef> initialize() override { return {}; }
    std::string call_tool(const std::string&, const std::string&) override {
        if (fail) throw std::runtime_error("socket closed");
        return reply;
    }
    std::vector<core::mcp::McpResourceDef> list_resources() override { return {}; }
    std::vector<core::mcp::McpResourceTemplateDef> list_resource_templates() override { return {}; }
    std::string read_resource(const std::string&) override { return {}; }
    std::vector<core::mcp::McpPromptDef> list_prompts() override { return {}; }
    std::string get_prompt(const std::string&, const std::string&) override { return {}; }
    bool supports_resources() const noexcept override { return false; }
    bool supports_prompts() const noexcept override { return false; }
    bool supports_sampling() const noexcept override { return false; }
    void shutdown() noexcept override {}
};

} // namespace

TEST_CASE("MCP servers panel explains a connected server", "[tui][mcp][mcp_panel]") {
    const auto now = std::chrono::system_clock::now();
    const auto output = render({connected_linear(now)}, 0, now);

    CHECK_THAT(output, ContainsSubstring("MCP servers · 1 connected"));
    CHECK_THAT(output, ContainsSubstring("linear"));
    CHECK_THAT(output, ContainsSubstring("https://mcp.linear.app/mcp"));
    CHECK_THAT(output, ContainsSubstring("OAuth"));
    CHECK_THAT(output, ContainsSubstring("Linear MCP v1.2.0 · protocol 2025-11-25"));
    CHECK_THAT(output, ContainsSubstring("connected"));
    CHECK_THAT(output, ContainsSubstring("18m ago"));
    CHECK_THAT(output, ContainsSubstring("handshake 820 ms"));
    CHECK_THAT(output, ContainsSubstring("✓ resources"));
    CHECK_THAT(output, ContainsSubstring("– prompts"));
    CHECK_THAT(output, ContainsSubstring("4 calls · 1 failed"));
    CHECK_THAT(output, ContainsSubstring("list_issues · 340 ms · 2m ago"));
    CHECK_THAT(output, ContainsSubstring("save_issue"));
}

TEST_CASE("MCP servers panel shows a running tool call", "[tui][mcp][mcp_panel]") {
    const auto now = std::chrono::system_clock::now();
    auto server = connected_linear(now);
    server.activity.in_flight = 1;
    server.activity.last_tool = "get_issue";
    const auto output = render({server}, 0, now);
    CHECK_THAT(output, ContainsSubstring("running get_issue"));
}

TEST_CASE("MCP servers panel tells the user how to fix a failed server",
          "[tui][mcp][mcp_panel]") {
    const auto now = std::chrono::system_clock::now();
    McpServerStatus failed{
        .name = "github",
        .transport = "http",
        .endpoint = "https://api.githubcopilot.com/mcp/",
        .auth = "OAuth",
        .state = McpServerState::failed,
        .error = "HTTP 401 Unauthorized",
        .needs_login = true,
        .connect_duration = std::chrono::milliseconds(1200),
    };
    const auto output = render({connected_linear(now), failed}, 1, now);

    CHECK_THAT(output, ContainsSubstring("1 connected · 1 failed"));
    CHECK_THAT(output, ContainsSubstring("needs /mcp login"));
    CHECK_THAT(output, ContainsSubstring("failed after 1.2 s"));
    CHECK_THAT(output, ContainsSubstring("HTTP 401 Unauthorized"));
    CHECK_THAT(output, ContainsSubstring("Run /mcp login github"));
    CHECK_THAT(output, ContainsSubstring("↑/↓"));
}

TEST_CASE("MCP servers panel without servers explains how to add one",
          "[tui][mcp][mcp_panel]") {
    const auto output = render({});
    CHECK_THAT(output, ContainsSubstring("No MCP servers configured"));
    CHECK_THAT(output, ContainsSubstring("/mcp add"));
}

TEST_CASE("MCP servers panel navigation stays in range and swallows typing",
          "[tui][mcp][mcp_panel]") {
    const ftxui::Box popover{10, 60, 5, 20};
    tui::McpServersPanelState state{.active = true};
    CHECK(tui::handle_mcp_servers_panel_event(state, ftxui::Event::ArrowDown, 2, popover));
    CHECK(state.selected == 1);
    CHECK(tui::handle_mcp_servers_panel_event(state, ftxui::Event::ArrowDown, 2, popover));
    CHECK(state.selected == 1);
    CHECK(tui::handle_mcp_servers_panel_event(state, ftxui::Event::ArrowUp, 2, popover));
    CHECK(state.selected == 0);
    CHECK(tui::handle_mcp_servers_panel_event(state, ftxui::Event::Character('x'), 2, popover));
    CHECK(state.active);
    CHECK(tui::handle_mcp_servers_panel_event(state, ftxui::Event::Escape, 2, popover));
    CHECK_FALSE(state.active);
    CHECK_FALSE(tui::handle_mcp_servers_panel_event(state, ftxui::Event::ArrowDown, 2, popover));
}

namespace {

ftxui::Event mouse_event(ftxui::Mouse::Button button, int x, int y) {
    ftxui::Mouse mouse;
    mouse.button = button;
    mouse.motion = ftxui::Mouse::Pressed;
    mouse.x = x;
    mouse.y = y;
    return ftxui::Event::Mouse("", mouse);
}

} // namespace

TEST_CASE("MCP servers popover owns the mouse only inside its box",
          "[tui][mcp][mcp_panel]") {
    const ftxui::Box popover{10, 60, 5, 20};
    tui::McpServersPanelState state{.active = true};

    // The wheel selects servers over the popover and scrolls the page elsewhere.
    CHECK(tui::handle_mcp_servers_panel_event(
        state, mouse_event(ftxui::Mouse::WheelDown, 20, 10), 2, popover));
    CHECK(state.selected == 1);
    CHECK_FALSE(tui::handle_mcp_servers_panel_event(
        state, mouse_event(ftxui::Mouse::WheelUp, 70, 10), 2, popover));
    CHECK(state.selected == 1);

    // A click inside is consumed; a click outside closes and passes through.
    CHECK(tui::handle_mcp_servers_panel_event(
        state, mouse_event(ftxui::Mouse::Left, 20, 10), 2, popover));
    CHECK(state.active);
    CHECK_FALSE(tui::handle_mcp_servers_panel_event(
        state, mouse_event(ftxui::Mouse::Left, 5, 30), 2, popover));
    CHECK_FALSE(state.active);
}

TEST_CASE("Popover placement drops down from the anchor and stays on screen",
          "[tui][popover][mcp_panel]") {
    const ftxui::Box screen{0, 119, 0, 39};

    SECTION("left edge lines up with the anchor when it fits") {
        const auto box = tui::place_popover_below({10, 25, 8, 8}, screen, 12, {.preferred_width = 60});
        CHECK(box.x_min == 10);
        CHECK(box.x_max == 69);
        CHECK(box.y_min == 9);
        CHECK(box.y_max == 20);
    }
    SECTION("narrows to stay under the anchor") {
        const auto box = tui::place_popover_below(
            {50, 65, 8, 8}, screen, 12, {.preferred_width = 96, .min_width = 60});
        CHECK(box.x_min == 50);
        CHECK(box.x_max == 118);
    }
    SECTION("slides left once it cannot narrow further") {
        const auto box = tui::place_popover_below(
            {90, 105, 8, 8}, screen, 12, {.preferred_width = 60, .min_width = 40});
        CHECK(box.x_max == 118);
        CHECK(box.x_min == 79);
    }
    SECTION("narrows to the screen and cuts the height to the room below") {
        const auto box = tui::place_popover_below({4, 18, 8, 8}, {0, 49, 0, 19}, 30);
        CHECK(box.x_min == 1);
        CHECK(box.x_max == 48);
        CHECK(box.y_min == 9);
        CHECK(box.y_max == 18);
    }
    SECTION("without an anchor it is centred at the top") {
        const auto box = tui::place_popover_below(tui::kEmptyBox, screen, 5, {.preferred_width = 60});
        CHECK(box.x_min == 30);
        CHECK(box.y_min == 1);
    }
    SECTION("end alignment lines the right edges up for a right-hand trigger") {
        const auto box = tui::place_popover_below(
            {90, 105, 8, 8}, screen, 12,
            {.preferred_width = 60, .align = tui::PopoverAlign::end});
        CHECK(box.x_max == 105);
        CHECK(box.x_min == 46);
        CHECK(box.y_min == 9);
    }
    SECTION("end alignment narrows toward the left edge before sliding") {
        const auto box = tui::place_popover_below(
            {10, 70, 8, 8}, screen, 12,
            {.preferred_width = 96, .min_width = 40, .align = tui::PopoverAlign::end});
        CHECK(box.x_min == 1);
        CHECK(box.x_max == 70);
    }
    SECTION("end alignment slides right once it cannot narrow further") {
        const auto box = tui::place_popover_below(
            {5, 20, 8, 8}, screen, 12,
            {.preferred_width = 60, .min_width = 40, .align = tui::PopoverAlign::end});
        CHECK(box.x_min == 1);
        CHECK(box.x_max == 40);
    }
}

TEST_CASE("Popover overlay reads the anchor laid out in the same frame",
          "[tui][popover][mcp_panel]") {
    ftxui::Box anchor = tui::kEmptyBox;
    ftxui::Box popover_box = tui::kEmptyBox;
    auto base = ftxui::vbox({
        ftxui::text("header line"),
        ftxui::hbox({ftxui::text("    "), ftxui::text("TRIGGER") | ftxui::reflect(anchor)}),
        ftxui::text("transcript that the popover covers"),
        ftxui::text("more transcript"),
        ftxui::text("prompt"),
    });
    auto element = tui::overlay_popover_below(
        std::move(base), ftxui::text("POPOVER"), &anchor,
        {.preferred_width = 7, .min_width = 7, .margin = 0}, &popover_box);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40), ftxui::Dimension::Fixed(5));
    ftxui::Render(screen, element);

    CHECK(popover_box.x_min == 4);
    CHECK(popover_box.y_min == 2);
    CHECK(popover_box.y_max == 2);
    const auto output = strip_sgr(screen.ToString());
    CHECK_THAT(output, ContainsSubstring("TRIGGER"));
    // Drawn under the trigger, cleared underneath, and the base around it kept.
    CHECK_THAT(output, ContainsSubstring("tranPOPOVERthat the popover covers"));
    CHECK_THAT(output, ContainsSubstring("prompt"));
}

TEST_CASE("MCP auth descriptions never reveal header values", "[mcp][mcp_panel]") {
    core::config::McpServerConfig keyed{
        .name = "linear-key",
        .transport = "http",
        .url = "https://mcp.linear.app/mcp",
        .headers = {{"Authorization", "Bearer lin_api_secret"}},
    };
    CHECK(core::mcp::describe_mcp_auth(keyed) == "Authorization header");

    core::config::McpServerConfig custom{
        .name = "custom",
        .transport = "http",
        .url = "http://localhost:9000/mcp",
        .headers = {{"X-Api-Key", "secret"}},
    };
    CHECK(core::mcp::describe_mcp_auth(custom) == "header: X-Api-Key");

    core::config::McpServerConfig oauth{
        .name = "linear", .transport = "http", .url = "https://mcp.linear.app/mcp"};
    CHECK(core::mcp::describe_mcp_auth(oauth) == "OAuth");
    CHECK(core::mcp::mcp_error_needs_login(oauth, "HTTP 401"));
    CHECK_FALSE(core::mcp::mcp_error_needs_login(oauth, "connection refused"));

    core::config::McpServerConfig local{
        .name = "fs", .transport = "stdio", .command = "npx", .args = {"-y", "server"},
        .env = {"TOKEN=x"}};
    CHECK(core::mcp::describe_mcp_endpoint(local) == "npx -y server");
    CHECK(core::mcp::describe_mcp_auth(local) == "local process · 1 env var");
}

TEST_CASE("MCP tools record their calls in the server activity", "[mcp][mcp_panel]") {
    auto session = std::make_shared<FakeSession>();
    auto activity = std::make_shared<core::mcp::McpServerActivity>();
    core::mcp::McpDynamicTool tool(session, {.name = "get_issue"}, "linear", activity);
    const auto context = test_support::make_session_context({});

    CHECK(tool.execute("{}", context) == R"({"result":"ok"})");
    auto snapshot = activity->snapshot();
    CHECK(snapshot.calls == 1);
    CHECK(snapshot.failures == 0);
    CHECK(snapshot.in_flight == 0);
    CHECK(snapshot.last_tool == "get_issue");
    CHECK_FALSE(snapshot.last_failed);

    session->reply = R"({"error":"Issue not found"})";
    (void)tool.execute("{}", context);
    session->fail = true;
    (void)tool.execute("{}", context);
    snapshot = activity->snapshot();
    CHECK(snapshot.calls == 3);
    CHECK(snapshot.failures == 2);
    CHECK(snapshot.last_failed);
    CHECK_THAT(snapshot.last_error, ContainsSubstring("socket closed"));
}

TEST_CASE("Startup banner draws the MCP server count as a popover trigger",
          "[tui][banner][mcp_panel]") {
    for (const bool with_context : {false, true}) {
        for (const bool open : {false, true}) {
            ftxui::Box mcp_box = tui::kEmptyBox;
            auto banner = tui::render_startup_banner_panel(
                "provider", "model", 1, with_context ? "AGENTS.md" : "", {}, "12:00:00",
                {}, nullptr, {}, false, 0,
                tui::PopoverTrigger{.hitbox = &mcp_box, .open = open});
            auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(120),
                                                ftxui::Dimension::Fit(banner));
            ftxui::Render(screen, banner);
            const std::string label = open ? "MCP servers: 1 ▴" : "MCP servers: 1 ▾";
            CHECK_THAT(strip_sgr(screen.ToString()), ContainsSubstring(label));
            // The chevron is part of the clickable area.
            CHECK(mcp_box.x_max - mcp_box.x_min + 1
                  == static_cast<int>(std::string_view("MCP servers: 1 ▾").size()) - 2);
        }
    }
}
