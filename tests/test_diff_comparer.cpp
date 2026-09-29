#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/viewer/DiffViewerCatalog.hpp"
#include "tui/viewer/ExternalDiffViewerController.hpp"
#include "tui/viewer/TerminalDiffViewer.hpp"

#if defined(__APPLE__)
#include "tui/lampo/LampoCliSession.hpp"
#endif

#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <unistd.h>

namespace {

using namespace tui::viewer;
using Catch::Matchers::ContainsSubstring;

constexpr std::string_view kPatch = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n";

/// What one comparison handed to a backend, so a test can assert on it after the
/// controller has let go of the backend.
struct Recorder {
    std::string patch;
    std::string session_id;
    bool saw_terminal = false;
    std::atomic_bool started{false};
    std::atomic_bool released{false};
};

// A backend with no side effects, so the controller can be exercised without
// paging anything or reaching for another application.
class RecordingViewer final : public DiffViewer {
public:
    RecordingViewer(
        std::shared_ptr<Recorder> recorder,
        ViewerLaunch launch,
        ViewStatus status)
        : recorder_(std::move(recorder)), launch_(launch), status_(status) {}

    [[nodiscard]] ViewerDescriptor descriptor() const noexcept override {
        return ViewerDescriptor{
            .id = "fake",
            .label = "Fake",
            .description = "Test backend.",
            .launch = launch_,
        };
    }

    [[nodiscard]] ViewResult view(const ViewContext& context) override {
        recorder_->started.store(true);
        recorder_->patch = std::string(context.patch);
        recorder_->session_id = context.session_id;
        recorder_->saw_terminal = static_cast<bool>(context.with_terminal);

        if (launch_ == ViewerLaunch::Detached) {
            // Stands in for another application taking its time, and for the
            // cancellation the host offers while it does.
            while (!context.cancellation.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            recorder_->released.store(true);
            return ViewResult::cancelled();
        }

        switch (status_) {
            case ViewStatus::Shown:
                return ViewResult::shown();
            case ViewStatus::Cancelled:
                return ViewResult::cancelled();
            case ViewStatus::Failed:
                break;
        }
        return ViewResult::failed("fake failure");
    }

private:
    std::shared_ptr<Recorder> recorder_;
    ViewerLaunch launch_;
    ViewStatus status_;
};

[[nodiscard]] DiffViewerCatalog catalog_recording_into(
    std::shared_ptr<Recorder> recorder,
    ViewerLaunch launch,
    ViewStatus status = ViewStatus::Shown) {
    DiffViewerCatalog catalog;
    catalog.register_viewer(
        ViewerDescriptor{
            .id = "fake",
            .label = "Fake",
            .description = {},
            .launch = launch,
        },
        [recorder = std::move(recorder), launch, status] {
            return std::make_unique<RecordingViewer>(recorder, launch, status);
        });
    return catalog;
}

[[nodiscard]] std::optional<ViewerOutcome> wait_for_outcome(
    ExternalDiffViewerController& controller) {
    for (int attempt = 0; attempt < 2000; ++attempt) {
        if (auto outcome = controller.take_outcome()) {
            return outcome;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::nullopt;
}

/// Lampo's session shape: exactly sixteen hexadecimal characters.
[[nodiscard]] bool is_lampo_session_id(std::string_view id) {
    if (id.size() != 16) {
        return false;
    }
    for (const char character : id) {
        const bool hex = (character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f');
        if (!hex) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("The builtin catalog offers Lampo on macOS only", "[diff_comparer]") {
    const auto& catalog = builtin_diff_comparers();
    REQUIRE(catalog.create("terminal") != nullptr);

#if defined(__APPLE__)
    auto lampo = catalog.create("lampo");
    REQUIRE(lampo != nullptr);
    REQUIRE(lampo->descriptor().launch == ViewerLaunch::Detached);
#else
    REQUIRE(catalog.create("lampo") == nullptr);
#endif

    // Every descriptor must be creatable and self-consistent.
    for (const auto& descriptor : catalog.descriptors()) {
        auto viewer = catalog.create(descriptor.id);
        REQUIRE(viewer != nullptr);
        REQUIRE(viewer->descriptor().id == descriptor.id);
    }

    // The built-in choice is a settings entry, not a backend: the transcript
    // renders diffs itself and nothing is launched for it.
    REQUIRE(catalog.create(default_diff_comparer_id()) == nullptr);
    REQUIRE(builtin_diff_comparer_descriptor().id == default_diff_comparer_id());
    REQUIRE(builtin_diff_comparer_descriptor().launch == ViewerLaunch::Inline);
}

TEST_CASE("Catalog lookup ignores case", "[diff_comparer]") {
    const auto& catalog = builtin_diff_comparers();
    REQUIRE(catalog.create("TERMINAL") != nullptr);
    REQUIRE(catalog.create("Terminal") != nullptr);
    REQUIRE(catalog.create("does-not-exist") == nullptr);
}

TEST_CASE("Configured values resolve to a usable comparer", "[diff_comparer]") {
    const auto& catalog = builtin_diff_comparers();

    SECTION("empty and builtin both mean the transcript") {
        for (const std::string_view configured : {"", "builtin", "BUILTIN"}) {
            auto selection = select_diff_viewer(catalog, configured);
            REQUIRE(selection.inline_view());
            REQUIRE(selection.warning.empty());
        }
    }

    SECTION("a registered id is honoured") {
        auto selection = select_diff_viewer(catalog, "terminal");
        REQUIRE_FALSE(selection.inline_view());
        REQUIRE(selection.viewer->descriptor().id == "terminal");
        REQUIRE(selection.warning.empty());
    }

    SECTION("an unknown but executable value is treated as a pager command") {
        auto selection = select_diff_viewer(catalog, "sh");
        REQUIRE_FALSE(selection.inline_view());
        REQUIRE(selection.warning.empty());
        REQUIRE(selection.viewer->descriptor().id == "terminal");
    }

    SECTION("an unusable value falls back to the transcript with a warning") {
        auto selection = select_diff_viewer(catalog, "definitely-not-a-comparer-9034");
        REQUIRE(selection.inline_view());
        REQUIRE_THAT(selection.warning, ContainsSubstring("definitely-not-a-comparer-9034"));
    }
}

TEST_CASE("The pager resolves and quotes its invocation", "[diff_comparer]") {
    SECTION("an explicit command wins") {
        REQUIRE(resolve_pager_command("delta --side-by-side") == "delta --side-by-side");
    }

    SECTION("an empty command falls back to the environment") {
        REQUIRE_FALSE(resolve_pager_command("").empty());
    }

    SECTION("the patch reaches the command on standard input, quoted for the shell") {
        REQUIRE(build_pager_invocation("less", "/tmp/a.patch") == "(less) < '/tmp/a.patch'");
        REQUIRE(
            build_pager_invocation("less", "/tmp/it's here.patch")
            == R"((less) < '/tmp/it'\''s here.patch')");
    }
}

TEST_CASE("A pager reads the patch on standard input, pipelines included", "[diff_comparer]") {
    const std::function<void(const std::function<void()>&)> with_terminal =
        [](const std::function<void()>& body) { body(); };
    const ViewContext context{
        .patch = kPatch,
        .session_id = "0123456789abcdef",
        .with_terminal = with_terminal,
    };
    const auto received = std::filesystem::temp_directory_path()
        / std::format("filo-pager-stdin-{}.patch", ::getpid());
    const auto read_received = [&received] {
        std::ifstream input(received, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input), {});
    };

    SECTION("a single command") {
        TerminalDiffViewer viewer(std::format("cat > '{}'", received.string()));
        REQUIRE(viewer.view(context).status() == ViewStatus::Shown);
        CHECK(read_received() == kPatch);
    }
    SECTION("the first stage of a pipeline") {
        TerminalDiffViewer viewer(std::format("cat | cat > '{}'", received.string()));
        REQUIRE(viewer.view(context).status() == ViewStatus::Shown);
        CHECK(read_received() == kPatch);
    }
    std::filesystem::remove(received);
}

TEST_CASE("A terminal backend fails cleanly without a terminal", "[diff_comparer]") {
    TerminalDiffViewer viewer("false");
    const ViewContext context{.patch = kPatch, .session_id = "0123456789abcdef"};

    const auto result = viewer.view(context);

    REQUIRE(result.status() == ViewStatus::Failed);
    REQUIRE_THAT(result.error(), ContainsSubstring("terminal"));
}

TEST_CASE("A pager backend runs its child on the restored terminal", "[diff_comparer]") {
    bool terminal_was_restored = false;
    const std::function<void(const std::function<void()>&)> with_terminal =
        [&terminal_was_restored](const std::function<void()>& body) {
            terminal_was_restored = true;
            body();
        };
    const ViewContext context{
        .patch = kPatch,
        .session_id = "0123456789abcdef",
        .with_terminal = with_terminal,
    };

    SECTION("a clean exit means the diff was shown") {
        TerminalDiffViewer viewer("true");
        REQUIRE(viewer.view(context).status() == ViewStatus::Shown);
        REQUIRE(terminal_was_restored);
    }

    SECTION("a failing pager is reported with its status") {
        TerminalDiffViewer viewer("false");
        const auto result = viewer.view(context);
        REQUIRE(result.status() == ViewStatus::Failed);
        REQUIRE_THAT(result.error(), ContainsSubstring("status 1"));
    }
}

TEST_CASE("The built-in comparer answers inline", "[diff_comparer]") {
    auto catalog = catalog_recording_into(std::make_shared<Recorder>(), ViewerLaunch::Terminal);
    ExternalDiffViewerController controller({}, catalog);

    const auto outcome = controller.open("builtin", kPatch);

    REQUIRE(outcome.has_value());
    REQUIRE(outcome->inline_view);
    REQUIRE_FALSE(outcome->notice.has_value());
    REQUIRE_FALSE(controller.busy());
}

TEST_CASE("An unknown comparer warns and stays inline", "[diff_comparer]") {
    auto catalog = catalog_recording_into(std::make_shared<Recorder>(), ViewerLaunch::Terminal);
    ExternalDiffViewerController controller({}, catalog);

    const auto outcome = controller.open("definitely-not-a-comparer-9034", kPatch);

    REQUIRE(outcome.has_value());
    REQUIRE(outcome->inline_view);
    REQUIRE(outcome->notice.has_value());
    REQUIRE_FALSE(outcome->notice->success);
    REQUIRE_THAT(outcome->notice->message, ContainsSubstring("definitely-not-a-comparer-9034"));
}

TEST_CASE("A terminal backend completes inline and receives the patch", "[diff_comparer]") {
    auto recorder = std::make_shared<Recorder>();
    auto catalog = catalog_recording_into(recorder, ViewerLaunch::Terminal);
    ExternalDiffViewerController controller(
        ExternalDiffViewerController::Host{
            .with_terminal = [](const std::function<void()>& body) { body(); },
        },
        catalog);

    const auto outcome = controller.open("fake", kPatch);

    REQUIRE(outcome.has_value());
    REQUIRE_FALSE(outcome->inline_view);
    // The host attaches what the patch leaves out only to a shown diff.
    REQUIRE(outcome->shown);
    // Success stays silent: the diff on screen is the confirmation.
    REQUIRE_FALSE(outcome->notice.has_value());
    REQUIRE_FALSE(controller.busy());
    REQUIRE(controller.status_label().empty());

    REQUIRE(recorder->patch == kPatch);
    // Lampo requires exactly sixteen hexadecimal characters of a session, so
    // every backend gets an identifier in that shape.
    REQUIRE(is_lampo_session_id(recorder->session_id));
    // A terminal backend runs its child on the restored terminal.
    REQUIRE(recorder->saw_terminal);
}

TEST_CASE("A failing backend reports which comparer failed", "[diff_comparer]") {
    auto catalog = catalog_recording_into(
        std::make_shared<Recorder>(),
        ViewerLaunch::Terminal,
        ViewStatus::Failed);
    ExternalDiffViewerController controller({}, catalog);

    const auto outcome = controller.open("fake", kPatch);

    REQUIRE(outcome.has_value());
    REQUIRE_FALSE(outcome->shown);
    REQUIRE(outcome->notice.has_value());
    REQUIRE_FALSE(outcome->notice->success);
    REQUIRE_THAT(outcome->notice->message, ContainsSubstring("Fake"));
    REQUIRE_THAT(outcome->notice->message, ContainsSubstring("fake failure"));
}

TEST_CASE("A detached backend publishes its outcome later", "[diff_comparer]") {
    auto recorder = std::make_shared<Recorder>();
    auto catalog = catalog_recording_into(recorder, ViewerLaunch::Detached);
    ExternalDiffViewerController controller({}, catalog);

    REQUIRE_FALSE(controller.open("fake", kPatch).has_value());
    REQUIRE(controller.busy());
    REQUIRE_THAT(controller.status_label(), ContainsSubstring("Fake"));

    // A second request while one is running is swallowed, not queued.
    REQUIRE_FALSE(controller.open("fake", kPatch).has_value());

    REQUIRE(controller.cancel());
    const auto outcome = wait_for_outcome(controller);
    REQUIRE(outcome.has_value());
    REQUIRE_FALSE(outcome->inline_view);
    REQUIRE_FALSE(outcome->shown);
    // Cancelling is a deliberate, silent user action.
    REQUIRE_FALSE(outcome->notice.has_value());
    REQUIRE(recorder->started.load());
    REQUIRE(recorder->patch == kPatch);
    REQUIRE(is_lampo_session_id(recorder->session_id));
    // The worker saw the cancellation rather than being abandoned.
    REQUIRE(recorder->released.load());
}

#if defined(__APPLE__)

TEST_CASE("The Lampo protocols keep the shape Lampo publishes", "[diff_comparer][lampo]") {
    // Lampo/Reviewer/Diff/ComparerCli.swift and Lampo/Prompter/PrompterCli.swift
    // declare these strings. A drift is silent: Lampo adopts an open only when
    // the directory, the file name and the announced session all match, so
    // anything else is treated as an ordinary document.
    CHECK(tui::lampo::kComparerProtocol.directory_prefix == "lampo-comparer-view-v1-");
    CHECK(tui::lampo::kComparerProtocol.pasteboard_prefix
          == "alessio.pollero.Lampo.comparer-view.");
    CHECK(tui::lampo::kComparerProtocol.pasteboard_type
          == "alessio.pollero.Lampo.comparer-view.v1");
    CHECK(tui::lampo::kComparerProtocol.file_name == "changes.patch");

    CHECK(tui::lampo::kPromptEditProtocol.directory_prefix == "lampo-prompter-edit-v1-");
    CHECK(tui::lampo::kPromptEditProtocol.pasteboard_prefix
          == "alessio.pollero.Lampo.prompter-edit.");
    CHECK(tui::lampo::kPromptEditProtocol.pasteboard_type
          == "alessio.pollero.Lampo.prompter-edit.v1");
    CHECK(tui::lampo::kPromptEditProtocol.file_name == "prompt.md");

    CHECK(tui::lampo::bundle_identifier() == "alessio.pollero.Lampo");
    // Lampo shows the client name as the title of a session it did not start.
    CHECK_FALSE(tui::lampo::client_name().empty());
}

#endif // defined(__APPLE__)
