#include "LampoPromptEditor.hpp"

#include "tui/lampo/LampoCliSession.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace tui::editor {
namespace {

constexpr auto kPollInterval = std::chrono::milliseconds(100);
// If Lampo never acknowledges, the user is left staring at an overlay; fail
// loudly instead and point them back at /settings.
constexpr auto kAcknowledgementTimeout = std::chrono::seconds(15);

/// Copies the saved draft back over the host's scratch buffer, which is the
/// only channel the editor-neutral host reads a result through.
[[nodiscard]] std::string restore_prompt(
    const std::filesystem::path& edited,
    const std::filesystem::path& scratch) {
    std::error_code ec;
    if (!std::filesystem::copy_file(
            edited, scratch, std::filesystem::copy_options::overwrite_existing, ec)) {
        return std::format("Could not restore the prompt from Lampo: {}", ec.message());
    }
    return {};
}

class LampoPromptEditor final : public PromptEditor {
public:
    [[nodiscard]] EditorDescriptor descriptor() const noexcept override {
        return lampo_prompt_editor_descriptor();
    }

    [[nodiscard]] EditResult edit(const EditContext& context) override {
        auto session = lampo::CliSession::create_from_file(
            lampo::kPromptEditProtocol,
            context.session_id,
            context.file);
        if (!session) {
            return EditResult::failed(std::move(session.error()));
        }

        context.report_progress(EditProgress::Launching);
        if (auto launched = session->launch(context.cancellation); !launched) {
            session->cancel();
            if (context.cancellation.stop_requested()) {
                return EditResult::cancelled();
            }
            return EditResult::failed(std::move(launched.error()));
        }

        EditResult result = await_completion(*session, context);
        if (result.status() == EditStatus::Saved) {
            if (auto error = restore_prompt(session->file(), context.file); !error.empty()) {
                return EditResult::failed(std::move(error));
            }
        }
        return result;
    }

private:
    /// Follows the prompter half of the protocol: `editing` once Lampo has the
    /// draft, then one of `saved`, `cancelled` or `failed` when the user is done.
    [[nodiscard]] static EditResult await_completion(
        lampo::CliSession& session,
        const EditContext& context) {
        bool acknowledged = false;
        const auto deadline = std::chrono::steady_clock::now() + kAcknowledgementTimeout;

        while (!context.cancellation.stop_requested()) {
            if (const auto message = session.poll(); message.has_value()) {
                if (message->state == lampo::state::editing) {
                    acknowledged = true;
                    context.report_progress(EditProgress::Editing);
                } else if (message->state == lampo::state::saved) {
                    return EditResult::saved();
                } else if (message->state == lampo::state::cancelled) {
                    return EditResult::cancelled();
                } else if (message->state == lampo::state::failed) {
                    return EditResult::failed(
                        message->error.empty()
                            ? "Lampo could not complete the edit."
                            : message->error);
                }
            }

            // Quitting the exact Lampo instance at any point cancels the edit.
            if (!session.app_running()) {
                return EditResult::cancelled();
            }
            if (!acknowledged && std::chrono::steady_clock::now() >= deadline) {
                session.cancel();
                return EditResult::failed(
                    "Lampo did not acknowledge the editing session. "
                    "Update Lampo, or choose System in /settings.");
            }
            lampo::CliSession::idle(context.cancellation, kPollInterval);
        }

        session.cancel();
        return EditResult::cancelled();
    }
};

} // namespace

EditorDescriptor lampo_prompt_editor_descriptor() noexcept {
    return EditorDescriptor{
        .id = "lampo",
        .label = "Lampo",
        .description = "Edit the draft in the Lampo macOS app.",
        .launch = LaunchMode::Detached,
    };
}

std::unique_ptr<PromptEditor> make_lampo_prompt_editor() {
    return std::make_unique<LampoPromptEditor>();
}

} // namespace tui::editor
