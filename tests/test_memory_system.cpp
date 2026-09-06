#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/ContextBuilder.hpp"
#include "core/memory/MemorySystem.hpp"
#include "core/memory/ToolRecoveryMemory.hpp"
#include "TestSessionContext.hpp"

#include <algorithm>
#include <filesystem>
#include <ranges>
#include <memory>
#include <string>

using Catch::Matchers::ContainsSubstring;

namespace {

struct TempDir {
    std::filesystem::path path;
    explicit TempDir(std::string name)
        : path(std::filesystem::temp_directory_path() / std::move(name)) {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

class SpyRecoveryMemory final : public core::memory::ToolRecoveryMemory {
public:
    [[nodiscard]] std::optional<std::string> recall(
        const core::memory::RecoveryKey&) override {
        ++recalls;
        return std::nullopt;
    }

    [[nodiscard]] std::vector<std::string> recall_runtime_hints(
        const std::string&, const std::string&) override {
        return {};
    }

    void record(const core::memory::RecoveryLesson&) override { ++records; }

    int recalls = 0;
    int records = 0;
};

} // namespace

TEST_CASE("MemorySystem is an owned handle, not a process singleton",
          "[memory][system]") {
    auto first = std::make_shared<core::memory::MemorySystem>();
    auto second = std::make_shared<core::memory::MemorySystem>();
    REQUIRE(first);
    REQUIRE(second);
    CHECK(first.get() != second.get());
}

TEST_CASE("default MemorySystem uses inert tool-recovery", "[memory][system]") {
    core::memory::MemorySystem system;
    const core::memory::RecoveryKey key{
        .tool = "search",
        .schema_fingerprint = "fp",
        .issue_code = "unknown_argument",
        .parameter = "query",
    };
    system.tool_recovery().record({.key = key, .hint = "use pattern"});
    system.tool_recovery().record({.key = key, .hint = "use pattern"});
    CHECK_FALSE(system.tool_recovery().recall(key).has_value());
    CHECK(system.tool_recovery().recall_runtime_hints("search", "fp").empty());
}

TEST_CASE("make_memory_system honours the tool-recovery config flag",
          "[memory][system]") {
    const auto disabled = core::memory::make_memory_system(
        core::memory::MemoryConfig{.tool_recovery = false});
    REQUIRE(disabled);
    CHECK(dynamic_cast<core::memory::NullToolRecoveryMemory*>(
              &disabled->tool_recovery())
          != nullptr);

    const auto enabled = core::memory::make_memory_system(
        core::memory::MemoryConfig{.tool_recovery = true});
    REQUIRE(enabled);
    CHECK(dynamic_cast<core::memory::FileToolRecoveryMemory*>(
              &enabled->tool_recovery())
          != nullptr);
}

TEST_CASE("make_memory_system wraps an injected recovery port",
          "[memory][system]") {
    auto spy = std::make_shared<SpyRecoveryMemory>();
    auto system = core::memory::make_memory_system(
        std::static_pointer_cast<core::memory::ToolRecoveryMemory>(spy));
    REQUIRE(system);

    const core::memory::RecoveryKey key{
        .tool = "search",
        .issue_code = "unknown_argument",
        .parameter = "query",
    };
    (void)system->tool_recovery().recall(key);
    system->tool_recovery().record({.key = key, .hint = "use pattern"});
    CHECK(spy->recalls == 1);
    CHECK(spy->records == 1);
}

TEST_CASE("MemorySystem semantic prompt is projected from the owned store",
          "[memory][system]") {
    TempDir dir{"filo_memory_system_prompt"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    core::memory::MemorySettings settings;
    settings.enabled = true;
    settings.auto_capture = true;
    REQUIRE(store.save_settings(settings));
    const auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path});
    REQUIRE(store.for_context(context).remember("Prefer concise engineering summaries.").ok);

    core::memory::MemorySystem system{
        store, std::make_shared<core::memory::NullToolRecoveryMemory>()};
    const auto with_capture = system.semantic_prompt_block(context);
    CHECK_THAT(with_capture, ContainsSubstring("[Memory]"));
    CHECK_THAT(with_capture,
               ContainsSubstring("Prefer concise engineering summaries."));
    CHECK_THAT(with_capture, ContainsSubstring("[Memory Capture]"));

    const auto without_capture = system.semantic_prompt_block(context, 24, false);
    CHECK_THAT(without_capture, ContainsSubstring("[Memory]"));
    CHECK(without_capture.find("[Memory Capture]") == std::string::npos);

    auto other_context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "other-project"});
    const auto other_prompt = system.semantic_prompt_block(other_context);
    CHECK(other_prompt.find("Prefer concise engineering summaries.") == std::string::npos);
    other_context.memory_policy.use_memories = false;
    CHECK(system.semantic_prompt_block(other_context).empty());
}

TEST_CASE("ContextBuilder uses an injected memory prompt and never opens a store",
          "[memory][system][context]") {
    auto context = test_support::make_workspace_session_context();
    const auto without = core::context::ContextBuilder(context)
        .include_project_context(false)
        .include_project_facts(false)
        .build_layers();
    CHECK(std::none_of(without.begin(), without.end(), [](const auto& layer) {
        return layer.kind == core::context::ContextLayerKind::Memory;
    }));

    const auto with = core::context::ContextBuilder(context)
        .include_project_context(false)
        .include_project_facts(false)
        .with_memory_prompt("\n\n[Memory]\n- Prefer rg.\n")
        .build_layers();
    const auto memory = std::ranges::find_if(with, [](const auto& layer) {
        return layer.kind == core::context::ContextLayerKind::Memory;
    });
    REQUIRE(memory != with.end());
    CHECK_THAT(memory->content, ContainsSubstring("Prefer rg."));
}
