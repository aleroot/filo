#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/PromptComponents.hpp"
#include "tui/ThreadModalHost.hpp"

#include <ftxui/component/event.hpp>
#include <ftxui/screen/screen.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace tui;
using namespace std::chrono_literals;

namespace {

core::tools::QuestionRequest make_request(std::string session_id,
                                          std::string question = "How?") {
    core::tools::QuestionRequest request;
    request.session_id = std::move(session_id);
    request.questions.push_back(core::tools::QuestionItem{
        .question = std::move(question),
        .header = "Ask",
        .options = {
            {"Yes", "Accept the default."},
            {"No", "Reject it."},
        },
    });
    request.promise = std::make_shared<
        std::promise<std::optional<QuestionDialogAnswers>>>();
    return request;
}

PermissionPrompt make_permission(std::string session_id) {
    PermissionPrompt prompt;
    prompt.session_id = std::move(session_id);
    prompt.tool_name = "run_terminal_command";
    prompt.args_preview = R"({"command":"ls"})";
    prompt.allow_label = "shell:ls";
    prompt.remember_rule = "shell:ls";
    prompt.promise = std::make_shared<std::promise<bool>>();
    return prompt;
}

std::string strip_ansi(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    for (std::size_t i = 0; i < input.size();) {
        if (input[i] == '\x1b' && i + 1 < input.size() && input[i + 1] == '[') {
            i += 2;
            while (i < input.size()) {
                const char character = input[i++];
                if (character >= '@' && character <= '~') {
                    break;
                }
            }
            continue;
        }
        output.push_back(input[i++]);
    }
    return output;
}

std::string render_text(ftxui::Element element) {
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(120),
        ftxui::Dimension::Fit(element));
    ftxui::Render(screen, element);
    return strip_ansi(screen.ToString());
}

template <typename Pred>
bool wait_until(Pred pred, std::chrono::milliseconds timeout = 2s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

} // namespace

TEST_CASE("hidden-thread question does not paint or consume keys",
          "[tui][thread-modal][question]") {
    ThreadModalHost host;
    auto request = make_request("thread-b", "Ship the change?");
    auto future = request.promise->get_future();
    host.post_question(std::move(request));

    CHECK_FALSE(host.question_visible("thread-a"));
    CHECK(host.question_visible("thread-b"));
    CHECK(host.waiting("thread-b"));
    CHECK_FALSE(host.waiting("thread-a"));
    CHECK(host.waiting_session_ids().contains("thread-b"));

    auto stolen = host.handle_question_event(
        "thread-a", ftxui::Event::Return, false);
    CHECK_FALSE(stolen.handled);
    CHECK_FALSE(stolen.has_resolution());
    CHECK(future.wait_for(0ms) == std::future_status::timeout);
    CHECK(render_text(host.render_question("thread-a")).find("Ship the change?")
          == std::string::npos);

    REQUIRE_THAT(render_text(host.render_question("thread-b")),
                 Catch::Matchers::ContainsSubstring("Ship the change?"));

    auto answered = host.handle_question_event(
        "thread-b", ftxui::Event::Return, false);
    REQUIRE(answered.handled);
    REQUIRE(answered.has_resolution());
    CHECK(answered.origin_session_id == "thread-b");
    answered.resolve();

    const auto answers = future.get();
    REQUIRE(answers.has_value());
    REQUIRE_FALSE(answers->empty());
    CHECK(answers->front().second == "Yes");
    CHECK_FALSE(host.question_visible("thread-b"));
    CHECK_FALSE(host.waiting("thread-b"));
}

TEST_CASE("two sessions can hold questions at the same time",
          "[tui][thread-modal][question][concurrency]") {
    ThreadModalHost host;
    auto first = make_request("alpha", "First question?");
    auto first_future = first.promise->get_future();
    auto second = make_request("beta", "Second question?");
    auto second_future = second.promise->get_future();
    host.post_question(std::move(first));
    host.post_question(std::move(second));

    CHECK(host.question_visible("alpha"));
    CHECK(host.question_visible("beta"));

    auto first_result = host.handle_question_event(
        "alpha", ftxui::Event::Return, false);
    REQUIRE(first_result.has_resolution());
    first_result.resolve();
    CHECK(first_future.get().has_value());
    CHECK(host.question_visible("beta"));
    CHECK_FALSE(host.question_visible("alpha"));

    auto second_result = host.handle_question_event(
        "beta", ftxui::Event::Character('2'), false);
    REQUIRE(second_result.has_resolution());
    second_result.resolve();
    const auto answers = second_future.get();
    REQUIRE(answers.has_value());
    CHECK(answers->front().second == "No");
}

TEST_CASE("same-session questions queue until the live dialog resolves",
          "[tui][thread-modal][question][queue]") {
    ThreadModalHost host;
    auto first = make_request("same", "Hold the slot?");
    auto first_future = first.promise->get_future();
    host.post_question(std::move(first));

    auto second = make_request("same", "Queued question?");
    auto second_future = second.promise->get_future();
    std::thread waiter([&]() { host.post_question(std::move(second)); });

    REQUIRE(wait_until([&] {
        return host.blocked_question_waiters("same") == 1;
    }));
    CHECK(host.question_visible("same"));
    CHECK(render_text(host.render_question("same")).find("Queued")
          == std::string::npos);

    auto first_result = host.handle_question_event(
        "same", ftxui::Event::Return, false);
    REQUIRE(first_result.has_resolution());
    first_result.resolve();
    CHECK(first_future.get().has_value());

    waiter.join();
    CHECK(host.question_visible("same"));
    REQUIRE_THAT(render_text(host.render_question("same")),
                 Catch::Matchers::ContainsSubstring("Queued question?"));

    auto queued = host.handle_question_event("same", ftxui::Event::Escape, false);
    REQUIRE(queued.has_resolution());
    queued.resolve();
    CHECK_FALSE(second_future.get().has_value());
}

TEST_CASE("hidden-thread permission does not paint or consume keys",
          "[tui][thread-modal][permission]") {
    ThreadModalHost host;
    auto slot = host.acquire_permission_slot("thread-b");
    REQUIRE(slot);

    auto prompt = make_permission("thread-b");
    auto future = prompt.promise->get_future();
    host.post_permission(std::move(prompt));

    CHECK_FALSE(host.permission_visible("thread-a"));
    CHECK(host.permission_visible("thread-b"));
    CHECK(host.waiting("thread-b"));
    CHECK_FALSE(host.permission_view("thread-a").has_value());
    const auto view = host.permission_view("thread-b");
    REQUIRE(view.has_value());
    CHECK(view->tool_name == "run_terminal_command");
    CHECK(view->selected == 0);

    auto stolen = host.handle_permission_event(
        "thread-a", ftxui::Event::Character('1'), false);
    CHECK_FALSE(stolen.handled);
    CHECK(future.wait_for(0ms) == std::future_status::timeout);
    CHECK(host.permission_visible("thread-b"));

    auto down = host.handle_permission_event(
        "thread-b", ftxui::Event::ArrowDown, false);
    REQUIRE(down.handled);
    CHECK_FALSE(down.has_resolution());
    CHECK(host.permission_view("thread-b")->selected == 1);

    auto approved = host.handle_permission_event(
        "thread-b", ftxui::Event::Character('1'), false);
    REQUIRE(approved.has_resolution());
    CHECK(approved.approved == true);
    CHECK_FALSE(approved.always_allow);
    CHECK_FALSE(host.permission_visible("thread-b"));
    approved.resolve();
    CHECK(future.get());
}

TEST_CASE("permission Return follows the current selection",
          "[tui][thread-modal][permission]") {
    ThreadModalHost host;
    auto slot = host.acquire_permission_slot("sess");
    auto prompt = make_permission("sess");
    auto future = prompt.promise->get_future();
    host.post_permission(std::move(prompt));

    REQUIRE(host.handle_permission_event(
                "sess", ftxui::Event::ArrowDown, false).handled);
    REQUIRE(host.handle_permission_event(
                "sess", ftxui::Event::ArrowDown, false).handled);
    REQUIRE(host.handle_permission_event(
                "sess", ftxui::Event::ArrowDown, false).handled);
    CHECK(host.permission_view("sess")->selected == 3);

    auto rejected = host.handle_permission_event(
        "sess", ftxui::Event::Return, false);
    REQUIRE(rejected.has_resolution());
    CHECK(rejected.approved == false);
    rejected.resolve();
    CHECK_FALSE(future.get());
}

TEST_CASE("permission always-allow and yolo shortcuts populate the result",
          "[tui][thread-modal][permission]") {
    ThreadModalHost host;
    {
        auto slot = host.acquire_permission_slot("a");
        auto prompt = make_permission("a");
        auto future = prompt.promise->get_future();
        host.post_permission(std::move(prompt));
        auto result = host.handle_permission_event(
            "a", ftxui::Event::Character('2'), false);
        REQUIRE(result.has_resolution());
        CHECK(result.always_allow);
        CHECK(result.remember_rule == "shell:ls");
        result.resolve();
        CHECK(future.get());
    }
    {
        auto slot = host.acquire_permission_slot("b");
        auto prompt = make_permission("b");
        auto future = prompt.promise->get_future();
        host.post_permission(std::move(prompt));
        auto result = host.handle_permission_event(
            "b", ftxui::Event::Character('3'), false);
        REQUIRE(result.has_resolution());
        CHECK(result.enable_yolo);
        CHECK(result.approved == true);
        result.resolve();
        CHECK(future.get());
    }
}

TEST_CASE("concurrent permission slots across sessions do not block each other",
          "[tui][thread-modal][permission][concurrency]") {
    ThreadModalHost host;
    auto first = host.acquire_permission_slot("alpha");
    auto second = host.acquire_permission_slot("beta");
    REQUIRE(first);
    REQUIRE(second);
    host.post_permission(make_permission("alpha"));
    host.post_permission(make_permission("beta"));
    CHECK(host.permission_visible("alpha"));
    CHECK(host.permission_visible("beta"));
}

TEST_CASE("same-session permission slots queue until the guard is released",
          "[tui][thread-modal][permission][queue]") {
    ThreadModalHost host;
    auto first = host.acquire_permission_slot("same");
    REQUIRE(first);

    std::atomic_bool second_acquired{false};
    std::thread waiter([&]() {
        auto second = host.acquire_permission_slot("same");
        second_acquired.store(static_cast<bool>(second), std::memory_order_release);
    });

    REQUIRE(wait_until([&] {
        return host.blocked_permission_waiters("same") == 1;
    }));
    CHECK_FALSE(second_acquired.load(std::memory_order_acquire));

    first = {};
    REQUIRE(wait_until([&] {
        return second_acquired.load(std::memory_order_acquire);
    }));
    waiter.join();
}

TEST_CASE("Ctrl+C on a visible permission stops the origin session",
          "[tui][thread-modal][permission]") {
    ThreadModalHost host;
    auto slot = host.acquire_permission_slot("origin");
    auto prompt = make_permission("origin");
    auto future = prompt.promise->get_future();
    host.post_permission(std::move(prompt));

    auto cancelled = host.handle_permission_event(
        "origin", ftxui::Event::Escape, true);
    REQUIRE(cancelled.has_resolution());
    CHECK(cancelled.stop_agent);
    CHECK(cancelled.origin_session_id == "origin");
    CHECK(cancelled.approved == false);
    cancelled.resolve();
    CHECK_FALSE(future.get());
}

TEST_CASE("interrupt_question cancels only that session",
          "[tui][thread-modal][question]") {
    ThreadModalHost host;
    auto keep = make_request("keep", "Stay open?");
    auto keep_future = keep.promise->get_future();
    auto drop = make_request("drop", "Go away?");
    auto drop_future = drop.promise->get_future();
    host.post_question(std::move(keep));
    host.post_question(std::move(drop));

    auto cancelled = host.interrupt_question("drop");
    REQUIRE(cancelled.has_resolution());
    cancelled.resolve();
    CHECK_FALSE(drop_future.get().has_value());
    CHECK(host.question_visible("keep"));
    CHECK_FALSE(host.question_visible("drop"));
    CHECK(keep_future.wait_for(0ms) == std::future_status::timeout);
}

TEST_CASE("erase_session interrupts overlays and drops the slot",
          "[tui][thread-modal][lifecycle]") {
    ThreadModalHost host;
    auto request = make_request("gone", "Leaving?");
    auto question_future = request.promise->get_future();
    host.post_question(std::move(request));

    auto slot = host.acquire_permission_slot("also-gone");
    auto prompt = make_permission("also-gone");
    auto permission_future = prompt.promise->get_future();
    host.post_permission(std::move(prompt));

    auto dismissed = host.erase_session("gone");
    REQUIRE(dismissed.has_resolution());
    dismissed.resolve();
    CHECK_FALSE(question_future.get().has_value());
    CHECK_FALSE(host.question_visible("gone"));
    CHECK_FALSE(host.waiting("gone"));

    auto perm_dismissed = host.erase_session("also-gone");
    CHECK_FALSE(perm_dismissed.has_resolution());
    CHECK_FALSE(permission_future.get());
    CHECK_FALSE(host.permission_visible("also-gone"));
}

TEST_CASE("shutdown cancels waiters and late posts",
          "[tui][thread-modal][lifecycle]") {
    ThreadModalHost host;
    auto request = make_request("hidden");
    auto question_future = request.promise->get_future();
    host.post_question(std::move(request));

    auto slot = host.acquire_permission_slot("other");
    REQUIRE(slot);
    auto prompt = make_permission("other");
    auto permission_future = prompt.promise->get_future();
    host.post_permission(std::move(prompt));

    std::atomic_bool queued_denied{false};
    std::thread queued([&]() {
        auto waiting_slot = host.acquire_permission_slot("other");
        queued_denied.store(!waiting_slot, std::memory_order_release);
    });

    REQUIRE(wait_until([&] {
        return host.blocked_permission_waiters("other") == 1;
    }));

    host.request_shutdown();
    CHECK(host.shutting_down());
    CHECK_FALSE(question_future.get().has_value());
    CHECK_FALSE(permission_future.get());
    queued.join();
    CHECK(queued_denied.load(std::memory_order_acquire));

    auto late = make_request("late");
    auto late_future = late.promise->get_future();
    host.post_question(std::move(late));
    CHECK_FALSE(late_future.get().has_value());
    CHECK_FALSE(host.acquire_permission_slot("late"));
}

TEST_CASE("startup banner marks a waiting hidden thread with a question mark",
          "[tui][banner][threads][waiting]") {
    std::vector<ftxui::Box> tab_hitboxes;
    auto panel = render_startup_banner_panel(
        "provider", "model", 0, {}, {}, "12:34:56",
        {
            {.label = "main", .active = true, .running = true},
            {.label = "work", .waiting = true},
        },
        &tab_hitboxes);
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(100), ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);
    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("? work"));
    CHECK(output.find("? main") == std::string::npos);
}
