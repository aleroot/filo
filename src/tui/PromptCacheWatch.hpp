#pragma once

// Watches the prompt caches of the TUI's threads and puts the two cold-cache
// decisions to the user:
//
//   * before a prompt is sent, when the heavy cache it would re-read has
//     expired — "Compact, then send / Send anyway / Cancel";
//   * shortly before an idle heavy cache expires — "Keep warm / Compact now /
//     Let it expire". An unanswered notice is withdrawn at expiry and replaced
//     by a one-line note in the transcript. With `prompt_cache_keep_warm`
//     set to auto the cache is renewed instead, without asking, until the
//     renewals would cost more than the cold write they avoid.
//
// The decisions themselves are core::agent::PromptCacheGuard's and the wording
// is prompt_cache::*'s; this class only owns timing and threads. Everything
// that touches the TUI goes through Dependencies, so MainApp stays the single
// owner of its overlays and transcript.

#include "DeadlineScheduler.hpp"
#include "PromptCacheDialogs.hpp"
#include "QuestionDialogController.hpp"
#include "ThreadRuntime.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace tui {

class PromptCacheWatch final {
public:
    struct Dependencies {
        /// Shows @p question as @p session_id's dialog and returns at once;
        /// the answer arrives through @p promise. May block while another
        /// dialog of the same session is open. Called from worker threads.
        std::function<void(std::string session_id,
                           core::tools::QuestionItem question,
                           QuestionDialogPromise promise)> post_question;
        /// Closes @p session_id's dialog if it is the one answering
        /// @p promise; true when it did. Called from worker threads.
        std::function<bool(std::string_view session_id,
                           const QuestionDialogPromise& promise)> withdraw_question;
        /// Appends a status line to the runtime's transcript.
        std::function<void(const ThreadRuntime::Ptr& runtime, std::string text)> notify;
        /// The thresholds in force now; read at every decision so a change in
        /// /settings applies to sessions already being watched.
        std::function<core::agent::PromptCacheGuardSettings()> settings;
        /// True while the user is writing in @p session_id's visible composer.
        /// An unsolicited dialog would take over the keyboard mid-sentence
        /// (digits select options), so the notice waits for a pause.
        std::function<bool(std::string_view session_id)> composer_in_use;
    };

    explicit PromptCacheWatch(Dependencies dependencies);
    ~PromptCacheWatch();

    PromptCacheWatch(const PromptCacheWatch&) = delete;
    PromptCacheWatch& operator=(const PromptCacheWatch&) = delete;

    /// (Re)arms the expiry notice for an idle runtime: after a turn settles,
    /// after a resume, after the cache was kept warm. Cheap and idempotent.
    void track(const ThreadRuntime::Ptr& runtime);

    /// Disarms a runtime that is closing.
    void forget(std::string_view session_id);

    /// Asks before @p runtime sends a prompt over an expired heavy cache;
    /// answers Send at once when there is nothing to ask. A prompt also
    /// restarts the session's budget of automatic renewals. Blocks until the
    /// user answers, so call it from the turn's worker thread, never the UI.
    [[nodiscard]] prompt_cache::SendChoice confirm_send(const ThreadRuntime::Ptr& runtime);

    /// Stops timers and detaches from Dependencies. Call after the modal host
    /// has released its waiters and before anything Dependencies captures is
    /// destroyed. Idempotent.
    void shutdown();

private:
    struct Shared;

    std::shared_ptr<Shared> shared_;
};

} // namespace tui
