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

TEST_CASE("ThreadRuntimeRegistry ordered_snapshot puts the main thread first",
          "[tui][thread_runtime][ordering]") {
    tui::ThreadRuntimeRegistry registry;
    auto middle = make_runtime("bbbb2222");
    auto newest = make_runtime("cccc3333");
    auto main = make_runtime("aaaa1111");
    middle->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.created_at = "2026-08-08T12:00:00Z";
    });
    newest->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.created_at = "2026-08-08T13:00:00Z";
    });
    main->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.created_at = "2026-08-08T14:00:00Z";
    });
    REQUIRE(registry.insert(middle));
    REQUIRE(registry.insert(newest));
    REQUIRE(registry.insert(main));

    const auto ordered = registry.ordered_snapshot(main->session_id());
    REQUIRE(ordered.size() == 3);
    CHECK(ordered[0] == main);
    CHECK(ordered[1] == middle);
    CHECK(ordered[2] == newest);
}

TEST_CASE("ThreadRuntimeRegistry retitle_auto_named follows the workspace",
          "[tui][thread_runtime][workspace]") {
    tui::ThreadRuntimeRegistry registry;

    auto main = make_runtime("aaaa1111");
    main->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "main";
        metadata.created_at = "2026-08-08T12:00:00Z";
    });
    auto first_auto = make_runtime("bbbb2222");
    first_auto->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "oldproj";
        metadata.auto_thread_name = true;
        metadata.created_at = "2026-08-08T13:00:00Z";
    });
    auto second_auto = make_runtime("cccc3333");
    second_auto->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "oldproj 2";
        metadata.auto_thread_name = true;
        metadata.created_at = "2026-08-08T14:00:00Z";
    });
    // A user-renamed thread already holds the new base name: auto titles must
    // neither take it nor overwrite it.
    auto user_named = make_runtime("dddd4444");
    user_named->mutate_metadata([](tui::ThreadRuntimeMetadata& metadata) {
        metadata.thread_name = "newproj";
        metadata.created_at = "2026-08-08T15:00:00Z";
    });
    REQUIRE(registry.insert(main));
    REQUIRE(registry.insert(first_auto));
    REQUIRE(registry.insert(second_auto));
    REQUIRE(registry.insert(user_named));

    registry.retitle_auto_named("newproj", main->session_id());

    CHECK(main->metadata().thread_name == "main");
    CHECK(user_named->metadata().thread_name == "newproj");
    CHECK(first_auto->metadata().thread_name == "newproj 2");
    CHECK(second_auto->metadata().thread_name == "newproj 3");
    CHECK(first_auto->metadata().auto_thread_name);
    CHECK_FALSE(user_named->metadata().auto_thread_name);

    // Retitling is idempotent when the workspace has not actually changed.
    registry.retitle_auto_named("newproj", main->session_id());
    CHECK(first_auto->metadata().thread_name == "newproj 2");
    CHECK(second_auto->metadata().thread_name == "newproj 3");
}
