#include <catch2/catch_test_macros.hpp>

#include "tui/editor/ExternalEditorController.hpp"
#include "tui/editor/PromptBuffer.hpp"
#include "tui/editor/PromptEditorCatalog.hpp"
#include "tui/editor/SystemPromptEditor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

namespace {

using namespace tui::editor;

// A backend with no side effects, so the controller can be exercised without
// spawning an editor.
class FakeEditor final : public PromptEditor {
public:
    FakeEditor(LaunchMode launch, EditStatus status, std::string replacement = {})
        : launch_(launch), status_(status), replacement_(std::move(replacement)) {}

    [[nodiscard]] EditorDescriptor descriptor() const noexcept override {
        return EditorDescriptor{
            .id = "fake",
            .label = "Fake",
            .description = "Test backend.",
            .launch = launch_,
        };
    }

    [[nodiscard]] EditResult edit(const EditContext& context) override {
        context.report_progress(EditProgress::Editing);
        switch (status_) {
            case EditStatus::Saved: {
                std::ofstream out(context.file, std::ios::binary | std::ios::trunc);
                out << replacement_;
                return EditResult::saved();
            }
            case EditStatus::Cancelled:
                return EditResult::cancelled();
            case EditStatus::Failed:
                break;
        }
        return EditResult::failed("fake failure");
    }

private:
    LaunchMode launch_;
    EditStatus status_;
    std::string replacement_;
};

// A detached backend that idles until it is cancelled.
class BlockingEditor final : public PromptEditor {
public:
    [[nodiscard]] EditorDescriptor descriptor() const noexcept override {
        return EditorDescriptor{
            .id = "blocking",
            .label = "Blocking",
            .description = "Test backend.",
            .launch = LaunchMode::Detached,
        };
    }

    [[nodiscard]] EditResult edit(const EditContext& context) override {
        started.store(true);
        while (!context.cancellation.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return EditResult::cancelled();
    }

    std::atomic_bool started{false};
};

[[nodiscard]] PromptEditorCatalog catalog_with(PromptEditorCatalog::Factory factory,
                                               EditorDescriptor descriptor) {
    PromptEditorCatalog catalog;
    catalog.register_editor(descriptor, std::move(factory));
    return catalog;
}

[[nodiscard]] std::optional<EditorOutcome> wait_for_outcome(ExternalEditorController& controller) {
    for (int attempt = 0; attempt < 2000; ++attempt) {
        if (auto outcome = controller.take_outcome()) {
            return outcome;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::nullopt;
}

} // namespace

TEST_CASE("PromptBuffer round-trips the draft and cleans up after itself", "[prompt_editor]") {
    std::filesystem::path file;
    {
        auto buffer = PromptBuffer::create("hello\nworld");
        REQUIRE(buffer.has_value());
        file = buffer->file();

        REQUIRE(std::filesystem::exists(file));
        REQUIRE(buffer->session_id().size() >= 16);

        const auto text = buffer->read();
        REQUIRE(text.has_value());
        REQUIRE(*text == "hello\nworld");
    }
    REQUIRE_FALSE(std::filesystem::exists(file));
    REQUIRE_FALSE(std::filesystem::exists(file.parent_path()));
}

TEST_CASE("PromptBuffer sessions never collide", "[prompt_editor]") {
    auto first = PromptBuffer::create("a");
    auto second = PromptBuffer::create("b");
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(first->file() != second->file());
    REQUIRE(first->session_id() != second->session_id());
}

TEST_CASE("The builtin catalog offers Lampo on macOS only", "[prompt_editor]") {
    const auto& catalog = builtin_prompt_editors();
    REQUIRE(catalog.find(default_prompt_editor_id()) != nullptr);

#if defined(__APPLE__)
    const auto* lampo = catalog.find("lampo");
    REQUIRE(lampo != nullptr);
    REQUIRE(lampo->launch == LaunchMode::Detached);
#else
    REQUIRE(catalog.find("lampo") == nullptr);
#endif

    // Every descriptor must be creatable and self-consistent.
    for (const auto& descriptor : catalog.descriptors()) {
        auto editor = catalog.create(descriptor.id);
        REQUIRE(editor != nullptr);
        REQUIRE(editor->descriptor().id == descriptor.id);
    }
}

TEST_CASE("Catalog lookup ignores case", "[prompt_editor]") {
    const auto& catalog = builtin_prompt_editors();
    REQUIRE(catalog.find("SYSTEM") != nullptr);
    REQUIRE(catalog.create("System") != nullptr);
    REQUIRE(catalog.create("does-not-exist") == nullptr);
}

TEST_CASE("Configured values resolve to a usable backend", "[prompt_editor]") {
    const auto& catalog = builtin_prompt_editors();

    SECTION("empty falls back to the default backend") {
        auto selection = select_prompt_editor(catalog, "");
        REQUIRE(selection.editor != nullptr);
        REQUIRE(selection.editor->descriptor().id == default_prompt_editor_id());
        REQUIRE(selection.warning.empty());
    }

    SECTION("a registered id is honoured") {
        auto selection = select_prompt_editor(catalog, "system");
        REQUIRE(selection.editor->descriptor().id == "system");
        REQUIRE(selection.warning.empty());
    }

    SECTION("an unknown but executable value is treated as a command") {
        auto selection = select_prompt_editor(catalog, "sh");
        REQUIRE(selection.editor != nullptr);
        REQUIRE(selection.warning.empty());
    }

    SECTION("an unusable value warns and falls back") {
        auto selection = select_prompt_editor(catalog, "definitely-not-an-editor-9034");
        REQUIRE(selection.editor != nullptr);
        REQUIRE(selection.editor->descriptor().id == default_prompt_editor_id());
        REQUIRE_FALSE(selection.warning.empty());
    }
}

TEST_CASE("The system editor command honours VISUAL then EDITOR", "[prompt_editor]") {
    REQUIRE(resolve_system_editor_command("nano") == "nano");

    ::unsetenv("VISUAL");
    ::unsetenv("EDITOR");
    REQUIRE(resolve_system_editor_command("") == "vi");

    ::setenv("EDITOR", "nano", 1);
    REQUIRE(resolve_system_editor_command("") == "nano");

    ::setenv("VISUAL", "hx", 1);
    REQUIRE(resolve_system_editor_command("") == "hx");

    ::unsetenv("VISUAL");
    ::unsetenv("EDITOR");
}

TEST_CASE("Editor invocations quote the buffer and add blocking flags", "[prompt_editor]") {
    SECTION("GUI editors are forced to block") {
        REQUIRE(build_editor_invocation("code", "/tmp/p.md") == "code --wait '/tmp/p.md'");
        REQUIRE(build_editor_invocation("subl", "/tmp/p.md") == "subl -w '/tmp/p.md'");
        REQUIRE(build_editor_invocation("code --wait", "/tmp/p.md") == "code --wait '/tmp/p.md'");
        REQUIRE(
            build_editor_invocation(
                R"("/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code")",
                "/tmp/p.md")
            == R"("/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" --wait '/tmp/p.md')");
    }

    SECTION("the vi family runs without a user init file") {
        REQUIRE(build_editor_invocation("vim", "/tmp/p.md") == "vim -i NONE '/tmp/p.md'");
        REQUIRE(build_editor_invocation("/usr/bin/nvim", "/tmp/p.md")
                == "/usr/bin/nvim -i NONE '/tmp/p.md'");
    }

    SECTION("plain terminal editors are left alone") {
        REQUIRE(build_editor_invocation("nano", "/tmp/p.md") == "nano '/tmp/p.md'");
    }

    SECTION("editor names are matched, not substrings") {
        // "vidra" contains "vi" but is not a vi-family editor.
        REQUIRE(build_editor_invocation("vidra", "/tmp/p.md") == "vidra '/tmp/p.md'");
    }

    SECTION("paths with quotes cannot break out of the argument") {
        const std::string invocation = build_editor_invocation("nano", "/tmp/it's here.md");
        REQUIRE(invocation == R"(nano '/tmp/it'\''s here.md')");
    }
}

TEST_CASE("Executable detection resolves through PATH", "[prompt_editor]") {
    REQUIRE(editor_command_is_executable("sh"));
    REQUIRE(editor_command_is_executable("sh -c"));
    REQUIRE(editor_command_is_executable("/bin/sh"));
    REQUIRE(editor_command_is_executable("'/bin/sh' -c"));
    REQUIRE_FALSE(editor_command_is_executable(""));
    REQUIRE_FALSE(editor_command_is_executable("'/bin/sh"));
    REQUIRE_FALSE(editor_command_is_executable("definitely-not-an-editor-9034"));

    auto buffer = PromptBuffer::create("");
    REQUIRE(buffer.has_value());
    const std::filesystem::path editor = buffer->file().parent_path() / "editor with spaces";
    {
        std::ofstream executable(editor);
        executable << "#!/bin/sh\n";
    }
    std::filesystem::permissions(
        editor,
        std::filesystem::perms::owner_read
            | std::filesystem::perms::owner_write
            | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace);

    REQUIRE(editor_command_is_executable("'" + editor.string() + "' --wait"));
}

TEST_CASE("A terminal backend completes inline", "[prompt_editor]") {
    const auto catalog = catalog_with(
        [] { return std::make_unique<FakeEditor>(LaunchMode::Terminal, EditStatus::Saved, "edited"); },
        EditorDescriptor{
            .id = "fake", .label = "Fake", .description = {}, .launch = LaunchMode::Terminal});

    ExternalEditorController controller({}, catalog);
    const auto outcome = controller.open("fake", "draft");

    REQUIRE(outcome.has_value());
    REQUIRE(outcome->text.has_value());
    REQUIRE(*outcome->text == "edited");
    REQUIRE_FALSE(outcome->notice.has_value());
    REQUIRE_FALSE(controller.busy());
}

TEST_CASE("A cancelled edit leaves the draft untouched and stays quiet", "[prompt_editor]") {
    const auto catalog = catalog_with(
        [] { return std::make_unique<FakeEditor>(LaunchMode::Terminal, EditStatus::Cancelled); },
        EditorDescriptor{
            .id = "fake", .label = "Fake", .description = {}, .launch = LaunchMode::Terminal});

    ExternalEditorController controller({}, catalog);
    const auto outcome = controller.open("fake", "draft");

    REQUIRE(outcome.has_value());
    REQUIRE_FALSE(outcome->text.has_value());
    REQUIRE_FALSE(outcome->notice.has_value());
}

TEST_CASE("A failed edit reports a notice naming the backend", "[prompt_editor]") {
    const auto catalog = catalog_with(
        [] { return std::make_unique<FakeEditor>(LaunchMode::Terminal, EditStatus::Failed); },
        EditorDescriptor{
            .id = "fake", .label = "Fake", .description = {}, .launch = LaunchMode::Terminal});

    ExternalEditorController controller({}, catalog);
    const auto outcome = controller.open("fake", "draft");

    REQUIRE(outcome.has_value());
    REQUIRE_FALSE(outcome->text.has_value());
    REQUIRE(outcome->notice.has_value());
    REQUIRE_FALSE(outcome->notice->success);
    REQUIRE(outcome->notice->message.starts_with("Fake editor:"));
}

TEST_CASE("A detached backend reports asynchronously", "[prompt_editor]") {
    const auto catalog = catalog_with(
        [] { return std::make_unique<FakeEditor>(LaunchMode::Detached, EditStatus::Saved, "async"); },
        EditorDescriptor{
            .id = "fake", .label = "Fake", .description = {}, .launch = LaunchMode::Detached});

    std::atomic_int wakeups{0};
    ExternalEditorController controller(
        {.wake_ui = [&wakeups] { wakeups.fetch_add(1); }, .with_terminal = {}},
        catalog);

    REQUIRE_FALSE(controller.open("fake", "draft").has_value());

    const auto outcome = wait_for_outcome(controller);
    REQUIRE(outcome.has_value());
    REQUIRE(outcome->text.has_value());
    REQUIRE(*outcome->text == "async");
    REQUIRE(wakeups.load() > 0);
    REQUIRE_FALSE(controller.busy());
    REQUIRE_FALSE(controller.take_outcome().has_value());
}

TEST_CASE("A detached session can be cancelled and restarted", "[prompt_editor]") {
    const auto catalog = catalog_with(
        [] { return std::make_unique<BlockingEditor>(); },
        EditorDescriptor{
            .id = "blocking",
            .label = "Blocking",
            .description = {},
            .launch = LaunchMode::Detached});

    ExternalEditorController controller({}, catalog);
    REQUIRE_FALSE(controller.open("blocking", "draft").has_value());

    for (int attempt = 0; attempt < 2000 && !controller.busy(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(controller.busy());
    REQUIRE_FALSE(controller.status_label().empty());

    // A second request must not clobber the running session.
    REQUIRE_FALSE(controller.open("blocking", "other").has_value());

    REQUIRE(controller.cancel());
    const auto outcome = wait_for_outcome(controller);
    REQUIRE(outcome.has_value());
    REQUIRE_FALSE(outcome->text.has_value());
    REQUIRE_FALSE(controller.busy());
    REQUIRE(controller.status_label().empty());
    REQUIRE_FALSE(controller.cancel());
}

TEST_CASE("An unknown backend still edits, and says why", "[prompt_editor]") {
    ExternalEditorController controller({});
    const auto outcome = controller.open("definitely-not-an-editor-9034", "draft");

    // Falls back to the system editor, which fails without a terminal handoff.
    REQUIRE(outcome.has_value());
    REQUIRE(outcome->notice.has_value());
    REQUIRE_FALSE(outcome->notice->success);
}
