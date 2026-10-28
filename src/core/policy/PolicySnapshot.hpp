#pragma once

#include "ApprovalChannel.hpp"
#include "PolicyRules.hpp"
#include "PolicyTypes.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace core::policy {

/**
 * A standing authorization for one exact destination.
 *
 * A grant is always narrower than the request that produced it: it names one
 * chat, principal, operation, host and port. It never widens the
 * enforcement the user was shown.
 */
struct PolicyGrant {
    std::string session_id;
    Subject subject;
    Action action;
    Resource resource;
    std::chrono::steady_clock::time_point expires_at{};
    // Remaining authorizations. A value of zero or less means unlimited.
    int uses{1};
};

/**
 * Thread-safe store of standing authorizations.
 *
 * Grants are consumed atomically: a single-use grant authorizes exactly one
 * operation even when several threads request it at once.
 */
class GrantStore {
public:
    void issue(PolicyGrant grant);

    /**
     * Consumes one use of a live grant matching the request.
     *
     * Returns false when no grant matches, when every matching grant has
     * expired, or when the matching grants have no uses left. Expired grants
     * are removed.
     */
    [[nodiscard]] bool consume(const Subject& subject,
                                const Action& action,
                                const Resource& resource,
                                std::string_view session_id);

    void revoke_session(std::string_view session_id);
    void revoke_all();
    [[nodiscard]] std::size_t size() const;

private:
    [[nodiscard]] static bool matches(const PolicyGrant& grant,
                                      const Subject& subject,
                                      const Action& action,
                                      const Resource& resource,
                                      std::string_view session_id) noexcept;

    mutable std::mutex mutex_;
    std::vector<PolicyGrant> grants_;
};

/**
 * A destination the application already holds because the user configured it.
 *
 * It is not a rule. An explicit matching rule still decides, including a
 * deny or an ask. It replaces an unmatched ask default, but not a deny.
 */
struct ConfiguredDestination {
    PrincipalKind kind{PrincipalKind::web_tool};
    std::string id;
    OperationKind operation{OperationKind::read};
    DataFlow flow{DataFlow::none};
    Resource resource;
};

/**
 * The immutable bundle an enforcement point holds.
 *
 * A snapshot holds the rules, the store of standing authorizations and the
 * channel that can ask a person. It is immutable, so one snapshot can be
 * shared by every thread and every subagent without further coordination. The
 * mutable state it owns lives behind shared pointers, and each of those
 * handles its own synchronization.
 */
class PolicySnapshot {
public:
    PolicySnapshot(PolicySet policy,
                   std::shared_ptr<GrantStore> grants = {},
                   std::shared_ptr<IApprovalChannel> channel = {});

    // A snapshot without a channel is the same policy, so it keeps the
    // identity the tool manager uses to recognise it.
    PolicySnapshot(const PolicySnapshot& other,
                   std::shared_ptr<IApprovalChannel> channel);

    /**
     * Decides one request.
     *
     * The rules decide first. A request that needs approval is checked
     * against the standing authorizations before anyone is asked, so an
     * approval the user already gave is not asked for again.
     */
    [[nodiscard]] Decision authorize(const NetworkIntent& intent) const;

    /**
     * Same decision as @c authorize, except that @p configured replaces an
     * unmatched ask default.
     *
     * A null @p configured is exactly @c authorize. A configured destination
     * never overrides a matching rule: an explicit deny still denies, and an
     * explicit ask still asks. A question is asked in the chat named by the
     * enclosing @c ApprovalOriginScope; with none, it is denied unasked.
     */
    [[nodiscard]] Decision authorize(const NetworkIntent& intent,
                                     const ConfiguredDestination* configured) const;

    [[nodiscard]] const PolicySet& policy() const noexcept { return policy_; }

    /**
     * Identifies the policy, independently of who holds it.
     *
     * Two snapshots with the same identity are the same policy, so one can
     * be handed to a worker that must not ask a person. Two snapshots with
     * different identities are different policies, and combining them would
     * be ambiguous.
     */
    [[nodiscard]] std::uint64_t identity() const noexcept { return identity_; }
    [[nodiscard]] std::shared_ptr<GrantStore> grants() const noexcept { return grants_; }
    [[nodiscard]] std::shared_ptr<IApprovalChannel> channel() const noexcept { return channel_; }
    [[nodiscard]] bool valid() const noexcept { return policy_.valid(); }

    /**
     * Attaches the channel that can ask a person for a decision.
     *
     * The application composes the policy before the user interface exists,
     * so the snapshot is built with a deferred channel. This attaches the
     * real one. It fails when the snapshot has no deferred channel or when
     * one is already attached, so a decision can never be answered by two
     * different channels.
     */
    [[nodiscard]] bool attach_approval_channel(std::shared_ptr<IApprovalChannel> delegate) const;

    /**
     * A copy of this snapshot that cannot ask anyone.
     *
     * A background worker may outlive the interface that could have asked a
     * person, so it inherits the rules and the standing authorizations but
     * not the channel. A request that needs a decision is denied there.
     */
    [[nodiscard]] PolicySnapshot without_approval_channel() const;

    // A short description of the effective policy, for status output.
    [[nodiscard]] std::string describe() const;

private:
    PolicySet policy_;
    std::shared_ptr<GrantStore> grants_;
    std::shared_ptr<IApprovalChannel> channel_;
    std::uint64_t identity_{0};
};

/**
 * Describes one request in the words a user is asked to approve.
 */
[[nodiscard]] std::string describe_intent(const NetworkIntent& intent);

/**
 * The policy snapshot the process is running under.
 *
 * Some boundaries are created long before the tool layer that would
 * otherwise own the policy: a model provider and an MCP server are both
 * configured at startup. The registry gives those boundaries the same
 * snapshot the tools use, without making the provider layer depend on the
 * tool layer.
 *
 * The snapshot is installed once, before anything can use it, and is never
 * replaced. A second installation is refused rather than silently applied.
 * Without one, reading it costs a single atomic load.
 */
[[nodiscard]] std::shared_ptr<const PolicySnapshot> active_policy();

[[nodiscard]] bool install_active_policy(std::shared_ptr<const PolicySnapshot> snapshot);

/**
 * Tests only: makes @p snapshot the active policy until destroyed, then
 * restores the one before it. The application installs its policy once,
 * with install_active_policy, and never replaces it.
 */
class ScopedActivePolicyForTesting {
public:
    explicit ScopedActivePolicyForTesting(std::shared_ptr<const PolicySnapshot> snapshot);
    ~ScopedActivePolicyForTesting();

    ScopedActivePolicyForTesting(const ScopedActivePolicyForTesting&) = delete;
    ScopedActivePolicyForTesting& operator=(const ScopedActivePolicyForTesting&) = delete;

private:
    std::shared_ptr<const PolicySnapshot> previous_;
};

} // namespace core::policy
