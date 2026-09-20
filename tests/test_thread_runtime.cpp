#include <catch2/catch_test_macros.hpp>

#include "tui/ThreadRuntime.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <string>

namespace {

tui::ThreadRuntime::Ptr make_runtime(std::string id) {
    return std::make_shared<tui::ThreadRuntime>(
        tui::ThreadRuntimeMetadata{
            .session_id = std::move(id),
            .created_at = "2026-08-08T12:00:00Z",
        },
        nullptr,
        std::make_shared<tui::ThreadRuntime::Messages>(),
        nullptr);
}

} // namespace

TEST_CASE("ThreadRuntime keeps turn admission and steering isolated",
          "[tui][thread_runtime][concurrency]") {
    auto first = make_runtime("aaaa1111");
    auto second = make_runtime("bbbb2222");

    tui::PendingAgentTurn first_turn{.text = "first"};
    REQUIRE(first->begin_or_queue(first_turn));
    CHECK(first->turn_active());
    CHECK_FALSE(second->turn_active());

    tui::PendingAgentTurn steering{.text = "steer"};
    CHECK_FALSE(first->begin_or_queue(steering));
    CHECK(first->queued_turn_count() == 1);
    CHECK(second->queued_turn_count() == 0);

    auto queued = first->finish_turn(false);
    REQUIRE(queued.has_value());
    CHECK(queued->text == "steer");
    CHECK(first->turn_active());

    CHECK_FALSE(first->finish_turn(true).has_value());
    CHECK_FALSE(first->turn_active());
    CHECK(first->completion_status() == tui::TurnCompletionStatus::Succeeded);
}

TEST_CASE("ThreadRuntime keeps thread and session names independent",
          "[tui][thread_runtime][identity]") {
    auto runtime = std::make_shared<tui::ThreadRuntime>(
        tui::ThreadRuntimeMetadata{
            .session_id = "aaaa1111",
            .thread_name = "main",
            .session_name = "saved-conversation",
            .created_at = "2026-08-08T12:00:00Z",
        },
        nullptr,
        std::make_shared<tui::ThreadRuntime::Messages>(),
        nullptr);

    auto metadata = runtime->metadata();
    REQUIRE(metadata.thread_name == "main");
    REQUIRE(metadata.session_name == "saved-conversation");

    metadata.thread_name = "filo";
    runtime->update_metadata(metadata);
    CHECK(runtime->metadata().thread_name == "filo");
    CHECK(runtime->metadata().session_name == "saved-conversation");
}

TEST_CASE("ThreadRuntime keeps model selection and history independent",
          "[tui][thread_runtime][model]") {
    auto main = make_runtime("main0001");
    auto secondary = make_runtime("work0001");

    auto main_metadata = main->metadata();
    main_metadata.model_selection = {
        .mode = tui::ModelSelectionMode::Manual,
        .manual_provider_name = "openai",
        .manual_model_name = "gpt-main",
    };
    main->update_metadata(main_metadata);

    auto secondary_metadata = secondary->metadata();
    secondary_metadata.model_selection = {
        .mode = tui::ModelSelectionMode::Router,
        .manual_provider_name = "anthropic",
        .manual_model_name = "claude-work",
        .router_policy = "work-policy",
    };
    secondary_metadata.previous_model_selection = tui::ModelSelectionSnapshot{
        .mode = tui::ModelSelectionMode::Manual,
        .manual_provider_name = "anthropic",
        .manual_model_name = "claude-old",
    };
    secondary->update_metadata(secondary_metadata);

    CHECK(main->metadata().model_selection.manual_model_name == "gpt-main");
    CHECK(main->metadata().model_selection.mode == tui::ModelSelectionMode::Manual);
    CHECK_FALSE(main->metadata().previous_model_selection.has_value());
    CHECK(secondary->metadata().model_selection.manual_model_name == "claude-work");
    CHECK(secondary->metadata().model_selection.router_policy == "work-policy");
    REQUIRE(secondary->metadata().previous_model_selection.has_value());
    CHECK(secondary->metadata().previous_model_selection->manual_model_name == "claude-old");
}

TEST_CASE("ThreadRuntime keeps approval state isolated",
          "[tui][thread_runtime][permissions]") {
    auto first = make_runtime("trust001");
    auto second = make_runtime("trust002");

    first->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.yolo_enabled = true;
        metadata.permission_rules.insert("shell:git");
    });

    CHECK(first->metadata().yolo_enabled);
    CHECK(first->metadata().permission_rules.contains("shell:git"));
    CHECK_FALSE(second->metadata().yolo_enabled);
    CHECK(second->metadata().permission_rules.empty());
}

TEST_CASE("ThreadRuntime request_stop does not finish sibling turns",
          "[tui][thread_runtime][stop]") {
    auto first = make_runtime("aaaa1111");
    auto second = make_runtime("bbbb2222");

    tui::PendingAgentTurn first_turn{.text = "first"};
    tui::PendingAgentTurn second_turn{.text = "second"};
    REQUIRE(first->begin_or_queue(first_turn));
    REQUIRE(second->begin_or_queue(second_turn));

    first->request_stop();
    CHECK(second->turn_active());
    CHECK(second->queued_turn_count() == 0);

    CHECK_FALSE(second->finish_turn(true).has_value());
    CHECK_FALSE(second->turn_active());
    CHECK(first->turn_active());
}

TEST_CASE("ThreadRuntimeRegistry selection never stops another runtime",
          "[tui][thread_runtime][concurrency]") {
    tui::ThreadRuntimeRegistry registry;
    auto first = make_runtime("aaaa1111");
    auto second = make_runtime("bbbb2222");
    REQUIRE(registry.insert(first));
    REQUIRE(registry.insert(second));
    REQUIRE(registry.select(first->session_id()) == first);

    tui::PendingAgentTurn turn{.text = "background"};
    REQUIRE(first->begin_or_queue(turn));
    REQUIRE(registry.select(second->session_id()) == second);

    CHECK(registry.current() == second);
    CHECK(first->turn_active());
    CHECK_FALSE(second->turn_active());
    CHECK(registry.running_session_ids().contains("aaaa1111"));

    first->messages()->push_back(tui::UiMessage{.text = "still owned by first"});
    CHECK(first->messages()->size() == 1);
    CHECK(second->messages()->empty());

    CHECK_FALSE(first->finish_turn(true).has_value());
}

TEST_CASE("ThreadRuntimeRegistry only erases idle non-current runtimes",
          "[tui][thread_runtime][lifecycle]") {
    tui::ThreadRuntimeRegistry registry;
    auto current = make_runtime("main0001");
    auto idle = make_runtime("idle0001");
    auto running = make_runtime("run00001");
    REQUIRE(registry.insert(current));
    REQUIRE(registry.insert(idle));
    REQUIRE(registry.insert(running));
    REQUIRE(registry.select(current->session_id()) == current);

    tui::PendingAgentTurn turn{.text = "work"};
    REQUIRE(running->begin_or_queue(turn));

    CHECK_FALSE(registry.erase(current->session_id()));
    CHECK_FALSE(registry.erase(running->session_id()));
    CHECK(registry.erase(idle->session_id()));
    CHECK_FALSE(registry.find(idle->session_id()));
    CHECK(registry.find(current->session_id()) == current);

    CHECK_FALSE(running->finish_turn(true).has_value());
    running->begin_worker();
    CHECK_FALSE(registry.erase(running->session_id()));
    running->finish_worker();
    CHECK(registry.erase(running->session_id()));
}

TEST_CASE("ThreadRuntime idle barrier includes detached worker lifetime",
          "[tui][thread_runtime][lifecycle]") {
    auto runtime = make_runtime("worker01");
    runtime->begin_worker();
    auto waiter = std::async(std::launch::async, [runtime]() {
        runtime->wait_until_idle();
    });

    CHECK(runtime->workers_in_flight() == 1);
    CHECK(waiter.wait_for(std::chrono::milliseconds{20}) == std::future_status::timeout);
    runtime->finish_worker();
    CHECK(waiter.wait_for(std::chrono::seconds{1}) == std::future_status::ready);
    CHECK(runtime->workers_in_flight() == 0);
}

TEST_CASE("ThreadRuntime save generations are scoped per thread",
          "[tui][thread_runtime][persistence]") {
    auto first = make_runtime("aaaa1111");
    auto second = make_runtime("bbbb2222");

    const auto first_old = first->request_save();
    const auto second_only = second->request_save();
    const auto first_new = first->request_save();

    CHECK_FALSE(first->is_latest_save(first_old));
    CHECK(first->is_latest_save(first_new));
    CHECK(second->is_latest_save(second_only));
}

TEST_CASE("ThreadRuntimeRegistry is a birth-order queue",
          "[tui][thread_runtime][ordering]") {
    tui::ThreadRuntimeRegistry registry;
    auto first = make_runtime("aaaa1111");
    auto second = make_runtime("bbbb2222");
    auto third = make_runtime("cccc3333");
    REQUIRE(registry.insert(first));
    REQUIRE(registry.insert(second));
    REQUIRE(registry.insert(third));
    REQUIRE(registry.size() == 3);

    const auto ordered = registry.ordered_snapshot();
    REQUIRE(ordered.size() == 3);
    CHECK(ordered[0] == first);
    CHECK(ordered[1] == second);
    CHECK(ordered[2] == third);
    CHECK(registry.primary() == first);
    CHECK(registry.successor(first->session_id()) == second);
    CHECK(registry.successor(second->session_id()) == third);
    CHECK(registry.successor(third->session_id()) == second);

    // Closing the queue head while another tab is visible promotes the next
    // oldest living thread, not the visible one.
    REQUIRE(registry.select(third->session_id()) == third);
    REQUIRE(registry.erase(first->session_id()));
    CHECK(registry.primary() == second);
    CHECK(registry.current() == third);
    CHECK(registry.size() == 2);
    CHECK(registry.ordered_snapshot()[0] == second);
    CHECK(registry.ordered_snapshot()[1] == third);
    CHECK(registry.successor(third->session_id()) == second);

    REQUIRE(registry.erase(second->session_id()));
    CHECK(registry.primary() == third);
    CHECK(registry.size() == 1);
    CHECK_FALSE(registry.erase(third->session_id()));
}

TEST_CASE("ThreadRuntimeRegistry rekey keeps queue position",
          "[tui][thread_runtime][ordering]") {
    tui::ThreadRuntimeRegistry registry;
    auto first = make_runtime("old00001");
    auto second = make_runtime("bbbb2222");
    REQUIRE(registry.insert(first));
    REQUIRE(registry.insert(second));
    REQUIRE(registry.rekey("old00001", "new00001"));
    CHECK(first->session_id() == "old00001");
    first->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.session_id = "new00001";
    });
    CHECK(registry.primary() == first);
    CHECK(registry.primary()->session_id() == "new00001");
    CHECK(registry.ordered_snapshot()[1] == second);
}

TEST_CASE("workspace changes rename only the owning thread",
          "[tui][thread_runtime][workspace]") {
    tui::ThreadRuntimeRegistry registry;
    auto first = make_runtime("aaaa1111");
    first->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "mlx-llm";
        metadata.auto_thread_name = true;
        metadata.session_name = "saved-conversation";
    });
    auto second = make_runtime("bbbb2222");
    second->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "mlx-llm 2";
        metadata.auto_thread_name = true;
    });
    REQUIRE(registry.insert(first));
    REQUIRE(registry.insert(second));

    SECTION("changing the second tab leaves the first alone") {
        REQUIRE(registry.select(second->session_id()) == second);
        registry.retitle_auto_named_thread(second->session_id(), "filo");
        CHECK(first->metadata().thread_name == "mlx-llm");
        CHECK(second->metadata().thread_name == "filo");

        // Repeating the change must not reserve the tab's own name.
        registry.retitle_auto_named_thread(second->session_id(), "filo");
        CHECK(first->metadata().thread_name == "mlx-llm");
        CHECK(second->metadata().thread_name == "filo");

        // Returning to the original project restores a unique suffix.
        registry.retitle_auto_named_thread(second->session_id(), "mlx-llm");
        CHECK(first->metadata().thread_name == "mlx-llm");
        CHECK(second->metadata().thread_name == "mlx-llm 2");
    }

    SECTION("changing the first tab does not renumber the second") {
        registry.retitle_auto_named_thread(first->session_id(), "filo");
        CHECK(first->metadata().thread_name == "filo");
        CHECK(second->metadata().thread_name == "mlx-llm 2");
    }

    SECTION("reapplying the workspace keeps the existing names") {
        registry.retitle_auto_named_thread(second->session_id(), "mlx-llm");
        CHECK(first->metadata().thread_name == "mlx-llm");
        CHECK(second->metadata().thread_name == "mlx-llm 2");
    }

    SECTION("an unknown thread does not rename any tabs") {
        registry.retitle_auto_named_thread("missing", "filo");
        CHECK(first->metadata().thread_name == "mlx-llm");
        CHECK(second->metadata().thread_name == "mlx-llm 2");
    }

    CHECK(first->metadata().auto_thread_name);
    CHECK(second->metadata().auto_thread_name);
    CHECK(first->metadata().session_name == "saved-conversation");
}

TEST_CASE("workspace tab names reserve all other titles and preserve custom names",
          "[tui][thread_runtime][workspace]") {
    tui::ThreadRuntimeRegistry registry;
    auto existing = make_runtime("aaaa1111");
    existing->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "filo";
        metadata.auto_thread_name = true;
    });
    auto custom = make_runtime("bbbb2222");
    custom->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "filo 2";
    });
    auto moving = make_runtime("cccc3333");
    moving->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "mlx-llm";
        metadata.auto_thread_name = true;
    });
    REQUIRE(registry.insert(existing));
    REQUIRE(registry.insert(custom));
    REQUIRE(registry.insert(moving));

    registry.retitle_auto_named_thread(moving->session_id(), "filo");
    CHECK(existing->metadata().thread_name == "filo");
    CHECK(custom->metadata().thread_name == "filo 2");
    CHECK(moving->metadata().thread_name == "filo 3");

    registry.retitle_auto_named_thread(custom->session_id(), "new-project");
    CHECK(existing->metadata().thread_name == "filo");
    CHECK(custom->metadata().thread_name == "filo 2");
    CHECK(moving->metadata().thread_name == "filo 3");
    CHECK_FALSE(custom->metadata().auto_thread_name);
}

TEST_CASE("ThreadRuntime keeps review activity isolated per thread",
          "[tui][thread_runtime][review]") {
    auto first = make_runtime("aaaa1111");
    auto second = make_runtime("bbbb2222");

    first->mutate_review_activity([](tui::ReviewActivity& state) {
        state.active = true;
        state.hint = "staged changes";
        state.view.total_groups = 17;
    });

    CHECK(first->review_activity().active);
    CHECK(first->review_activity().hint == "staged changes");
    CHECK(first->review_activity().view.total_groups == 17);
    CHECK_FALSE(second->review_activity().active);
    CHECK(second->review_activity().view.total_groups == 0);

    REQUIRE(first->begin_turn());
    CHECK(first->turn_active());
    CHECK_FALSE(second->turn_active());

    tui::PendingAgentTurn other{.text = "work on the other thread"};
    REQUIRE(second->begin_or_queue(other));
    CHECK(second->turn_active());
    CHECK(first->turn_active());
    CHECK(first->review_activity().active);
}

TEST_CASE("ThreadRuntime parks prompt drafts independently",
          "[tui][thread_runtime][prompt]") {
    auto first = make_runtime("aaaa1111");
    auto second = make_runtime("bbbb2222");

    first->set_prompt_draft({.text = "from A", .cursor = 4});
    CHECK(first->prompt_draft().text == "from A");
    CHECK(first->prompt_draft().cursor == 4);
    CHECK(second->prompt_draft().text.empty());
    CHECK(second->prompt_draft().cursor == 0);

    first->set_prompt_draft({.text = "hi", .cursor = 99});
    CHECK(first->prompt_draft().text == "hi");
    CHECK(first->prompt_draft().cursor == 2);
}

TEST_CASE("exchange_prompt_draft isolates the visible composer across tabs",
          "[tui][thread_runtime][prompt]") {
    auto first = make_runtime("aaaa1111");
    auto second = make_runtime("bbbb2222");
    second->set_prompt_draft({.text = "from B", .cursor = 4});

    std::string visible = "from A";
    int cursor = 2;
    tui::exchange_prompt_draft(*first, *second, visible, cursor);
    CHECK(visible == "from B");
    CHECK(cursor == 4);
    CHECK(first->prompt_draft().text == "from A");
    CHECK(first->prompt_draft().cursor == 2);
    CHECK(second->prompt_draft().text == "from B");

    tui::exchange_prompt_draft(*second, *first, visible, cursor);
    CHECK(visible == "from A");
    CHECK(cursor == 2);
    CHECK(second->prompt_draft().text == "from B");
    CHECK(second->prompt_draft().cursor == 4);

    tui::exchange_prompt_draft(*first, *first, visible, cursor);
    CHECK(visible == "from A");
    CHECK(cursor == 2);
}
