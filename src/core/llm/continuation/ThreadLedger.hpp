#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

/// Server-side conversation continuation ("message threads"), provider-neutral.
///
/// A provider that can keep a conversation on the server lets a client send
/// only the messages after the last stored response. The ledger decides, for
/// one request, whether that is safe. Every decision is derived from digests of
/// the bytes a request would carry, never from callers remembering to reset
/// state: an edit, a compaction, a model or effort change shows up as a
/// mismatch, and the answer is a fresh thread. Doubt always means a create.
namespace core::llm::continuation {

using Clock = std::chrono::steady_clock;

/// One message as it appears on the wire.
struct WireMessage {
    std::uint64_t digest = 0;       ///< Digest of the whole serialized message.
    std::uint64_t reply_digest = 0; ///< Digest of an assistant message's content; 0 otherwise.

    friend bool operator==(const WireMessage&, const WireMessage&) = default;
};

/// What a complete (unthreaded) request would send.
struct WireTranscript {
    std::vector<WireMessage> messages;
    std::uint64_t create_only_digest = 0; ///< Fields only a create may carry (system, tools).
    std::uint64_t fixed_digest = 0;       ///< Fields every request of a thread must repeat unchanged.
};

enum class CreateReason : std::uint8_t {
    first_request,
    expired,
    fixed_fields_changed,
    create_only_changed,
    transcript_drift,
    anchor_rewritten,
    nothing_new,
    replay_after_rejection,
};

[[nodiscard]] std::string_view to_string(CreateReason reason) noexcept;

/// Send the whole conversation and store it as a new thread.
struct Create {
    CreateReason reason = CreateReason::first_request;
};

/// Send only the messages from `first_new_message` on, after `previous_message_id`.
struct Continue {
    std::string previous_message_id;
    std::size_t first_new_message = 0;
};

enum class UnthreadedReason : std::uint8_t {
    not_planned,
    no_scope,
    scope_busy, ///< Another request of the same scope is in flight.
    rejected,   ///< The server refused the thread request.
    dropped,    ///< The client could not build the thread request.
};

[[nodiscard]] std::string_view to_string(UnthreadedReason reason) noexcept;

/// Send the request exactly as without threads.
struct Unthreaded {
    UnthreadedReason reason = UnthreadedReason::not_planned;
};

using Plan = std::variant<Unthreaded, Create, Continue>;

[[nodiscard]] constexpr bool is_threaded(const Plan& plan) noexcept {
    return !std::holds_alternative<Unthreaded>(plan);
}

struct ThreadPolicy {
    /// How long the server keeps a thread after its last stored response.
    std::chrono::seconds lifetime{300};
    /// A continue sent this close to the expiry is sent as a create instead.
    std::chrono::seconds safety_margin{10};
    /// Scopes remembered at once; the least recently stored is dropped first.
    std::size_t max_scopes = 64;
    std::function<Clock::time_point()> now = [] { return Clock::now(); };
};

class Attempt;

/// Shared, thread-safe record of the last stored response per scope. Copies
/// are handles to the same ledger, so protocol clones share it cheaply.
///
/// Provider-neutral by construction: a provider describes its request as a
/// WireTranscript and the server's retention as a ThreadPolicy, and turns the
/// Plan into its own wire fields. The ledger never sees a body, a header or a
/// model id.
class ThreadLedger {
public:
    explicit ThreadLedger(ThreadPolicy policy = {});

    /// Plans one request of @p scope. An empty scope, or one with a request
    /// already in flight, is unthreaded.
    [[nodiscard]] Attempt open(std::string_view scope, WireTranscript transcript) const;

    /// Drops what is known about @p scope; its next request is a create.
    void forget(std::string_view scope) const;

    /// Number of scopes with a stored thread (for tests and diagnostics).
    [[nodiscard]] std::size_t size() const;

private:
    friend class Attempt;
    struct State;
    std::shared_ptr<State> state_;
};

/// One planned request. Move-only; settle it once the response is known. An
/// attempt destroyed unsettled (cancelled, failed, thrown) forgets its scope,
/// so the next request never continues from a response nobody saw complete.
///
/// Invariant: a plan becomes threaded only while the attempt holds its scope.
/// Once the scope is released (abandoned, replanned unthreaded, destroyed) no
/// replan can thread it again, so no thread request is ever sent without the
/// ledger counting it in flight.
class Attempt {
public:
    Attempt() = default;
    Attempt(Attempt&&) noexcept;
    Attempt& operator=(Attempt&&) noexcept;
    Attempt(const Attempt&) = delete;
    Attempt& operator=(const Attempt&) = delete;
    ~Attempt();

    [[nodiscard]] const Plan& plan() const noexcept { return plan_; }
    [[nodiscard]] const WireTranscript& transcript() const noexcept { return transcript_; }

    /// Replaces the plan after the server rejected it. Downgrading to
    /// Unthreaded forgets the scope; a threaded plan is refused once the
    /// scope is released.
    void replan(Plan plan);

    /// The response id the server assigned (`message_start`).
    void observe_response_id(std::string_view id);

    /// The response completed and the server stored it: the next request may
    /// continue from it. @p reply_digest identifies the reply as the next
    /// request must replay it; 0, or no observed id, forgets the scope instead.
    void commit(std::uint64_t reply_digest);

    /// The response must not be continued (refusal, truncated tool call...).
    void abandon();

private:
    friend class ThreadLedger;
    /// Gives the scope back and forgets what was stored for it.
    void release() noexcept;

    std::shared_ptr<ThreadLedger::State> ledger_;
    std::string scope_;
    WireTranscript transcript_;
    Plan plan_ = Unthreaded{};
    std::string response_id_;
    bool owns_scope_ = false;
};

} // namespace core::llm::continuation
