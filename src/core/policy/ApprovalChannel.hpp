#pragma once

#include "PolicyTypes.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace core::policy {

/**
 * What a user decided about one request.
 *
 * The outcome is deliberately narrow. There is no "allow everything" answer:
 * a standing authorization is issued as a grant whose scope is exactly the
 * request that was shown.
 */
enum class ApprovalOutcome {
    approved_once,       // authorize this operation only
    approved_for_session, // authorize this exact destination for the session
    denied,
    cancelled,
    unavailable,          // no channel could ask anyone
};

// Who asked, so the question is shown in that chat and cancellation watches
// that chat's stop state. An empty session has nobody to ask.
struct ApprovalContext {
    std::string session_id;
    std::function<bool()> stopped;
};

// The request a user is asked to decide on.
struct ApprovalRequest {
    Subject subject;
    Action action;
    Resource resource;
    // A short, human-readable description of the operation.
    std::string description;
    // The chat that caused the request. The channel must not substitute
    // whichever chat happens to be visible when the question is posted.
    ApprovalContext origin;
};

/**
 * The interface through which a policy asks a person for a decision.
 *
 * Implementations live outside the policy core: the TUI prompts, headless
 * mode does not. The core never assumes a channel exists, and a missing
 * channel denies.
 */
class IApprovalChannel {
public:
    virtual ~IApprovalChannel() = default;
    virtual ApprovalOutcome request(const ApprovalRequest& request) = 0;
};

/**
 * A channel whose delegate is attached later.
 *
 * The application composes the policy before the user interface exists, and
 * the interface is what can ask a person. This channel bridges the two: it
 * denies until a delegate is attached, and accepts only one delegate, so a
 * decision can never be answered by two different channels.
 */
class DeferredApprovalChannel final : public IApprovalChannel {
public:
    // Returns false when a delegate is already attached.
    bool attach(std::shared_ptr<IApprovalChannel> delegate);
    [[nodiscard]] bool attached() const;

    ApprovalOutcome request(const ApprovalRequest& request) override;

private:
    mutable std::mutex mutex_;
    std::shared_ptr<IApprovalChannel> delegate_;
};

/**
 * A channel that never authorizes anything.
 *
 * Used by headless sessions and by background workers that may outlive the
 * interface that could have asked.
 */
class UnavailableApprovalChannel final : public IApprovalChannel {
public:
    ApprovalOutcome request(const ApprovalRequest&) override { return ApprovalOutcome::unavailable; }
};

/**
 * Names the chat whose work runs on this thread, for as long as it runs.
 *
 * A request that needs a person's decision is asked in that chat, and is
 * withdrawn when that chat stops. Work starts where the chat is known: the
 * tool manager holds a scope around each tool call and the model provider
 * around each request. A request made outside any scope names no chat and is
 * denied without asking, so a question can never appear in the wrong chat.
 */
class ApprovalOriginScope {
public:
    ApprovalOriginScope(std::string session_id, std::function<bool()> stopped);
    ~ApprovalOriginScope();
    ApprovalOriginScope(const ApprovalOriginScope&) = delete;
    ApprovalOriginScope& operator=(const ApprovalOriginScope&) = delete;

    [[nodiscard]] static const ApprovalContext* current() noexcept;

private:
    ApprovalContext context_;
    const ApprovalContext* previous_{nullptr};
};

} // namespace core::policy
