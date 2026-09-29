#include "LampoDiffViewer.hpp"

#include "tui/lampo/LampoCliSession.hpp"

#include <chrono>
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

        return await_viewing(*session, context);
    }

private:
    /// The comparer protocol is one-way: `viewing` means Lampo has read the
    /// whole patch and put it on screen, and its window then lives
    /// independently of Filo. Only a refusal, or a Lampo that disappears while
    /// still reading, can follow. The user asked to see the diff, so a Lampo
    /// that quits before showing it is a failure, not a cancellation.
    [[nodiscard]] static ViewResult await_viewing(
        lampo::CliSession& session,
        const ViewContext& context) {
        const auto deadline = std::chrono::steady_clock::now() + kAcknowledgementTimeout;

        while (!context.cancellation.stop_requested()) {
            if (auto answer = next_answer(session)) {
                return std::move(*answer);
            }
            if (!session.app_running()) {
                // Lampo may have answered between the poll above and its exit.
                if (auto answer = next_answer(session)) {
                    return std::move(*answer);
                }
                return ViewResult::failed("Lampo quit before showing the diff.");
            }
            if (std::chrono::steady_clock::now() >= deadline) {
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

    /// Lampo's answer, when it published one since the last poll.
    [[nodiscard]] static std::optional<ViewResult> next_answer(lampo::CliSession& session) {
        const auto message = session.poll();
        if (!message.has_value()) {
            return std::nullopt;
        }
        if (message->state == lampo::state::viewing) {
            return ViewResult::shown();
        }
        if (message->state == lampo::state::failed) {
            return ViewResult::failed(
                message->error.empty() ? "Lampo could not reconstruct the diff." : message->error);
        }
        return std::nullopt;
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
