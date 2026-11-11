#pragma once

#include "../continuation/ThreadLedger.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

/// Anthropic message threads (`message-threads-2026-08-12`) over a Claude
/// Messages body. Pure functions over JSON bytes plus the session state a
/// provider's protocol clones share. Every function reports "cannot" instead
/// of guessing; the caller then sends the request exactly as without threads.
namespace core::llm::protocols::claude_threads {

inline constexpr std::string_view kBeta = "message-threads-2026-08-12";

/// Digests of a complete, unsigned Claude Messages body. nullopt when the body
/// has a shape a thread cannot carry (not an object, no `messages` array, a
/// `system` that is not a block array, a `thread` field already present).
/// @p fixed_extra names request properties outside the body that the server
/// fingerprints too (the beta set), so a change to them also starts a thread.
[[nodiscard]] std::optional<continuation::WireTranscript> wire_transcript(
    std::string_view unsigned_body, std::string_view fixed_extra);

/// Digest of a reply as the next request replays it, from the
/// `{"content":[...]}` assistant-content record of a completed response. 0
/// when the record cannot be read.
[[nodiscard]] std::uint64_t reply_digest(std::string_view assistant_content);

/// What a continue sends of the fields the thread already holds.
/// Ordered: a session only ever moves down this list.
enum class InheritedFields : std::uint8_t {
    /// Only the billing block of `system`, no `tools` (Claude Code's shape).
    billing_only,
    /// Neither `system` nor `tools` (the documented shape).
    omitted,
    /// The complete `system` and `tools`, after the server refused to infer them.
    full,
};

/// The thread request for @p plan, built from the complete unsigned body and
/// still unsigned. nullopt for an unthreaded plan or a body it cannot cut.
[[nodiscard]] std::optional<std::string> threaded_body(
    std::string_view unsigned_body, const continuation::Plan& plan, InheritedFields fields);

enum class Rejection : std::uint8_t {
    none,               ///< Not a 400/404: the ordinary error handling applies.
    not_found,          ///< 404 `thread_not_found`.
    fingerprint_mismatch,
    already_continued,
    continue_in_flight,
    unsupported_request,
    other,              ///< A 400/404 without a thread error code.
};

struct RejectionDetails {
    Rejection kind = Rejection::none;
    /// A fingerprint mismatch that names `system` or `tools`, or names nothing
    /// readable: the continue must send those fields itself.
    bool names_inherited_fields = false;
};

[[nodiscard]] RejectionDetails classify_rejection(int status_code, std::string_view body);
[[nodiscard]] std::string_view to_string(Rejection rejection) noexcept;

/// Thread state shared by every clone of one provider's protocol.
class ThreadSession {
public:
    explicit ThreadSession(continuation::ThreadPolicy policy = {}) : ledger_(std::move(policy)) {}

    [[nodiscard]] const continuation::ThreadLedger& ledger() const noexcept { return ledger_; }

    /// A model the server refused threads for stays unthreaded for the session.
    [[nodiscard]] bool suspended(std::string_view model) const;
    void suspend(std::string_view model);

    /// Continues leave `system`/`tools` to the thread until the server refuses it once.
    [[nodiscard]] InheritedFields inherited_fields() const noexcept {
        return static_cast<InheritedFields>(inherited_.load(std::memory_order_acquire));
    }
    /// Moves to @p fields when it is further down the list; never back.
    void escalate(InheritedFields fields) noexcept;

private:
    continuation::ThreadLedger ledger_;
    mutable std::mutex mutex_;
    std::unordered_set<std::string> suspended_models_;
    std::atomic<std::uint8_t> inherited_{static_cast<std::uint8_t>(InheritedFields::billing_only)};
};

/// How a threaded response ended, as far as the thread cares.
struct ReplyOutcome {
    std::string_view stop_reason;
    bool incomplete_tool_call = false;
    /// The `{"content":[...]}` assistant-content record of the reply, when
    /// the stream produced one.
    std::optional<std::string_view> assistant_content;
};

/// One request's part in a message thread: its plan, the server's answers to
/// it, and what its reply leaves for the next request. A protocol clone holds
/// one per request; the ThreadSession it points at is shared by every clone.
///
/// Every member is a total function: none throws, and anything unexpected
/// leaves the request unthreaded, so the caller sends its ordinary body.
class RequestThread {
public:
    RequestThread() = default;
    explicit RequestThread(std::shared_ptr<ThreadSession> session) noexcept : session_(std::move(session)) {}

    [[nodiscard]] const std::shared_ptr<ThreadSession>& session() const noexcept { return session_; }

    /// Whether a request of @p scope for @p model may be threaded at all; a
    /// caller checks it before paying for the body plan() needs.
    [[nodiscard]] bool available(std::string_view scope, std::string_view model) const noexcept;

    /// Plans the request from its complete unsigned body. @p fixed_extra
    /// names request properties outside the body the server fingerprints.
    void plan(std::string_view scope, std::string_view model, std::string_view unsigned_body,
              std::string_view fixed_extra) noexcept;

    [[nodiscard]] bool threaded() const noexcept { return continuation::is_threaded(attempt_.plan()); }

    /// The unsigned body to send for the current plan, cut from the complete
    /// @p unsigned_body; nullopt sends the complete body as it is.
    [[nodiscard]] std::optional<std::string> body(std::string_view unsigned_body) const noexcept;

    /// Walks the fallback ladder (continue -> create -> unthreaded) for a
    /// rejected response; true when the request should be built again.
    [[nodiscard]] bool on_rejection(int status_code, std::string_view response_body) noexcept;
    /// A 200 answered the request: an unthreaded resend that passed after a
    /// thread was refused suspends threads for the model.
    void on_success() noexcept;

    /// Stream observers, called for every event of a threaded response.
    void observe_event(std::string_view raw_event) noexcept;
    void observe_response_id(std::string_view id) noexcept;
    void settle(const ReplyOutcome& outcome) noexcept;

    /// A transport retry resends the same body; only per-response facts reset.
    void reset_attempt() noexcept { fallback_reply_ = false; }
    /// Ends the request; an unsettled thread is forgotten. The session stays.
    void reset() noexcept;

private:
    void drop(std::string_view why) noexcept;

    std::shared_ptr<ThreadSession> session_;
    /// Declared after session_: an unsettled attempt releases its scope first.
    continuation::Attempt attempt_;
    std::string scope_;
    std::string model_;
    bool fallback_reply_ = false;     ///< A fallback model answered: not stored.
    bool suspend_on_success_ = false; ///< An unthreaded resend proves threads were the problem.
};

} // namespace core::llm::protocols::claude_threads
