#include "LampoDiffViewer.hpp"

#include "tui/lampo/LampoCliSession.hpp"

#include <chrono>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace tui::viewer {
namespace {

constexpr auto kPollInterval = std::chrono::milliseconds(100);
// Lampo reconstructs both revisions of every file before it answers, so a large
// turn gets the same grace a cold launch does.
constexpr auto kAcknowledgementTimeout = std::chrono::seconds(30);

class LampoDiffViewer final : public DiffViewer {
public:
    [[nodiscard]] ViewerDescriptor descriptor() const noexcept override {
        return lampo_diff_viewer_descriptor();
    }

    [[nodiscard]] ViewResult view(const ViewContext& context) override {
        auto session = lampo::CliSession::create(
            lampo::kComparerProtocol,
            context.session_id,
            context.patch);
        if (!session) {
            return ViewResult::failed(std::move(session.error()));
        }

        if (auto launched = session->launch(context.cancellation); !launched) {
            session->cancel();
            if (context.cancellation.stop_requested()) {
                return ViewResult::cancelled();
            }
            return ViewResult::failed(std::move(launched.error()));
        }

        return await_completion(*session, context);
    }

private:
    /// Follows the comparer half of the protocol. A display-only Lampo answers
    /// `viewing` and the window then lives independently of Filo. One that took
    /// Filo's request for the comments back answers `annotating` instead, and
    /// keeps the session open until its window closes: `saved`, whose comments
    /// it wrote over the patch, or `cancelled` when it held none. The user asked to see the diff,
    /// so a Lampo that quits before showing it is a failure; one that quits
    /// during a review merely ends it.
    [[nodiscard]] static ViewResult await_completion(
        lampo::CliSession& session,
        const ViewContext& context) {
        bool acknowledged = false;
        const auto deadline = std::chrono::steady_clock::now() + kAcknowledgementTimeout;
        auto next_answer = [&]() -> std::optional<ViewResult> {
            const auto message = session.poll();
            if (!message) { return std::nullopt; }
            if (message->state == lampo::state::annotating) {
                acknowledged = true;
                if (context.report_viewing) { context.report_viewing(); }
            } else if (message->state == lampo::state::viewing) {
                // A Lampo that does not know the request predates returning
                // comments: the diff is on screen and the comments stay in its
                // chat, which is what this comparison always did.
                return ViewResult::shown();
            } else if (message->state == lampo::state::saved) {
                return read_comments(session);
            } else if (message->state == lampo::state::cancelled) {
                return ViewResult::cancelled();
            } else if (message->state == lampo::state::failed) {
                return ViewResult::failed(
                    message->error.empty() ? "Lampo could not complete the review." : message->error);
            }
            return std::nullopt;
        };

        while (!context.cancellation.stop_requested()) {
            if (auto answer = next_answer()) { return std::move(*answer); }
            if (!session.app_running()) {
                // Lampo may have answered between the poll above and its exit.
                if (auto answer = next_answer()) { return std::move(*answer); }
                session.cancel();
                return acknowledged ? ViewResult::cancelled()
                    : ViewResult::failed("Lampo quit before showing the diff.");
            }
            if (!acknowledged && std::chrono::steady_clock::now() >= deadline) {
                session.cancel();
                return ViewResult::failed(
                    "Lampo did not acknowledge the comparison. "
                    "Update Lampo, or choose another comparer in /settings.");
            }
            lampo::CliSession::idle(context.cancellation, kPollInterval);
        }

        session.cancel();
        return ViewResult::cancelled();
    }

    /// The comments Lampo wrote over the patch it was handed.
    [[nodiscard]] static ViewResult read_comments(const lampo::CliSession& session) {
        std::ifstream input(session.file(), std::ios::binary);
        if (!input) {
            return ViewResult::failed("Could not read Lampo's review comments.");
        }
        return ViewResult::reviewed(std::string(std::istreambuf_iterator<char>(input), {}));
    }
};

} // namespace

ViewerDescriptor lampo_diff_viewer_descriptor() noexcept {
    return ViewerDescriptor{
        .id = "lampo",
        .label = "Lampo",
        .description = "Open the diff in the Lampo macOS app's comparer.",
        .launch = ViewerLaunch::Detached,
    };
}

std::unique_ptr<DiffViewer> make_lampo_diff_viewer() {
    return std::make_unique<LampoDiffViewer>();
}

} // namespace tui::viewer
