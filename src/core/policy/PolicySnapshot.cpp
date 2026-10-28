#include "PolicySnapshot.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <format>
#include <mutex>

namespace core::policy {
namespace {

// Every snapshot gets its own identity when it is created.
std::uint64_t next_identity() {
    static std::atomic<std::uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

// How long a session-wide authorization lasts. A session can outlast it, in
// which case the request is asked again: an expired grant is a missed
// convenience, never a silent widening.
constexpr std::chrono::hours kSessionGrantLifetime{12};

[[nodiscard]] bool matches_configured(const ConfiguredDestination* configured,
                                      const NetworkIntent& intent) noexcept {
    return configured != nullptr
        && configured->kind == intent.subject.kind
        && configured->id == intent.subject.id
        && configured->operation == intent.action.operation
        && configured->flow == intent.action.flow
        && configured->resource == intent.resource;
}

// The slot that holds the policy of the process.
//
// Some boundaries are created long before the tool layer that would
// otherwise own the policy: a model provider and an MCP server are both
// configured at startup. The registry gives those boundaries the same
// snapshot the tools use, without making the provider layer depend on the
// tool layer.
std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::shared_ptr<const PolicySnapshot>& registry_slot() {
    static std::shared_ptr<const PolicySnapshot> snapshot;
    return snapshot;
}

// Lets the common case, no policy at all, skip the lock.
std::atomic<bool>& installed() {
    static std::atomic<bool> flag{false};
    return flag;
}

// Replaces the slot; the caller holds the lock.
void store_locked(std::shared_ptr<const PolicySnapshot> snapshot) {
    registry_slot() = std::move(snapshot);
    installed().store(registry_slot() != nullptr, std::memory_order_release);
}

} // namespace

PolicySnapshot::PolicySnapshot(PolicySet policy,
                               std::shared_ptr<GrantStore> grants,
                               std::shared_ptr<IApprovalChannel> channel)
    : policy_(std::move(policy)), grants_(std::move(grants)), channel_(std::move(channel)) {
    if (!grants_) grants_ = std::make_shared<GrantStore>();
    if (!channel_) channel_ = std::make_shared<UnavailableApprovalChannel>();
    identity_ = next_identity();
}

PolicySnapshot::PolicySnapshot(const PolicySnapshot& other,
                               std::shared_ptr<IApprovalChannel> channel)
    : policy_(other.policy_), grants_(other.grants_), channel_(std::move(channel)),
      identity_(other.identity_) {
    if (!channel_) channel_ = std::make_shared<UnavailableApprovalChannel>();
}

std::string describe_intent(const NetworkIntent& intent) {
    return std::format("{} ({}) to {}:{}",
                        intent.subject.id,
                        to_string(intent.subject.kind),
                        intent.resource.host,
                        intent.resource.port);
}

Decision PolicySnapshot::authorize(const NetworkIntent& intent) const {
    return authorize(intent, nullptr);
}

Decision PolicySnapshot::authorize(const NetworkIntent& intent,
                                   const ConfiguredDestination* configured) const {
    if (!policy_.valid())
        return {Effect::deny, "network policy is invalid: " + policy_.invalid_reason()};

    const auto decision = policy_.evaluate(intent.subject, intent.action, intent.resource);
    // Configured destinations are implicit approvals only for an ask default.
    // A deny-by-default policy must also deny configured destinations.
    if (!decision.matched_rule && decision.effect == Effect::ask
        && matches_configured(configured, intent))
        return {Effect::allow, "allowed because the user configured this destination"};
    if (decision.effect != Effect::ask) return decision;

    const ApprovalContext* origin = ApprovalOriginScope::current();
    const std::string_view session_id = origin ? origin->session_id : std::string_view{};
    if (!session_id.empty() && grants_
        && grants_->consume(intent.subject, intent.action, intent.resource, session_id))
        return {Effect::allow, "allowed by a standing network grant"};

    if (!channel_) return {Effect::deny, "network approval is unavailable"};

    ApprovalOutcome outcome{ApprovalOutcome::unavailable};
    try {
        outcome = channel_->request(ApprovalRequest{
            .subject = intent.subject,
            .action = intent.action,
            .resource = intent.resource,
            .description = describe_intent(intent),
            .origin = origin != nullptr ? *origin : ApprovalContext{}});
    } catch (...) {
        // The approval channel crosses a trust boundary. A failure there can
        // never authorize an operation, and it must not leak the channel's
        // exception text.
        return {Effect::deny, "network approval failed"};
    }

    switch (outcome) {
        case ApprovalOutcome::approved_once:
            return {Effect::allow, "approved for this request"};
        case ApprovalOutcome::approved_for_session:
            if (grants_ && !session_id.empty()) {
                grants_->issue(PolicyGrant{.session_id = std::string(session_id),
                                           .subject = intent.subject,
                                           .action = intent.action,
                                           .resource = intent.resource,
                                           .expires_at = std::chrono::steady_clock::now()
                                                            + kSessionGrantLifetime,
                                           .uses = 0});
            }
            return {Effect::allow, "approved for this session"};
        case ApprovalOutcome::cancelled:
            return {Effect::deny, "network request cancelled"};
        case ApprovalOutcome::denied:
            return {Effect::deny, "network request denied"};
        case ApprovalOutcome::unavailable:
            break;
    }
    return {Effect::deny, "network approval is unavailable"};
}

std::string PolicySnapshot::describe() const {
    if (!policy_.valid())
        return "invalid: " + policy_.invalid_reason();
    return std::format("{} rule(s), default {}",
                        policy_.size(),
                        to_string(policy_.default_effect()));
}

PolicySnapshot PolicySnapshot::without_approval_channel() const {
    return PolicySnapshot(*this, std::make_shared<UnavailableApprovalChannel>());
}

bool PolicySnapshot::attach_approval_channel(std::shared_ptr<IApprovalChannel> delegate) const {
    if (!delegate || !channel_) return false;
    const auto deferred = std::dynamic_pointer_cast<DeferredApprovalChannel>(channel_);
    if (!deferred) return false;
    return deferred->attach(std::move(delegate));
}

void GrantStore::issue(PolicyGrant grant) {
    std::lock_guard lock(mutex_);
    grants_.push_back(std::move(grant));
}

bool GrantStore::matches(const PolicyGrant& grant,
                         const Subject& subject,
                         const Action& action,
                         const Resource& resource,
                         std::string_view session_id) noexcept {
    // A grant names one principal, one operation and one destination. It is
    // never a wildcard: the user approved a specific destination, so only
    // that destination may reuse the approval.
    if (session_id.empty() || grant.session_id != session_id) return false;
    if (grant.subject.kind != subject.kind) return false;
    if (grant.subject.id != subject.id) return false;
    if (grant.action.operation != action.operation) return false;
    if (grant.action.flow != action.flow) return false;
    if (grant.resource.host != resource.host) return false;
    if (grant.resource.port != resource.port) return false;
    return true;
}

bool GrantStore::consume(const Subject& subject,
                         const Action& action,
                         const Resource& resource,
                         std::string_view session_id) {
    std::lock_guard lock(mutex_);
    const auto now = std::chrono::steady_clock::now();
    std::erase_if(grants_, [&](const PolicyGrant& grant) { return grant.expires_at <= now; });

    for (std::size_t index = 0; index < grants_.size(); ++index) {
        auto& grant = grants_[index];
        if (!matches(grant, subject, action, resource, session_id)) continue;
        if (grant.uses > 0) {
            --grant.uses;
            if (grant.uses == 0) {
                grants_.erase(grants_.begin() + static_cast<std::ptrdiff_t>(index));
                return true;
            }
        }
        // A grant with no remaining uses limit authorizes every request.
        return true;
    }
    return false;
}

void GrantStore::revoke_session(std::string_view session_id) {
    if (session_id.empty()) return;
    std::lock_guard lock(mutex_);
    std::erase_if(grants_, [&](const PolicyGrant& grant) { return grant.session_id == session_id; });
}

void GrantStore::revoke_all() {
    std::lock_guard lock(mutex_);
    grants_.clear();
}

std::size_t GrantStore::size() const {
    std::lock_guard lock(mutex_);
    return grants_.size();
}

std::shared_ptr<const PolicySnapshot> active_policy() {
    if (!installed().load(std::memory_order_acquire)) return {};
    std::lock_guard lock(registry_mutex());
    return registry_slot();
}

bool install_active_policy(std::shared_ptr<const PolicySnapshot> snapshot) {
    if (!snapshot) return false;
    std::lock_guard lock(registry_mutex());
    if (registry_slot()) return false;
    store_locked(std::move(snapshot));
    return true;
}

ScopedActivePolicyForTesting::ScopedActivePolicyForTesting(
    std::shared_ptr<const PolicySnapshot> snapshot) {
    std::lock_guard lock(registry_mutex());
    previous_ = registry_slot();
    store_locked(std::move(snapshot));
}

ScopedActivePolicyForTesting::~ScopedActivePolicyForTesting() {
    std::lock_guard lock(registry_mutex());
    store_locked(previous_);
}

} // namespace core::policy
