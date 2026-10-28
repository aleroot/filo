#include "NetworkPolicyFamily.hpp"

#include "PolicySource.hpp"

#include <format>
#include <memory>

namespace core::policy {
namespace {

// The reserved word that selects the policy that asks. It is the only word
// the argument may be, and the only place the option's grammar is written.
constexpr std::string_view kAskArgument{"ask"};

} // namespace

std::expected<NetworkPolicyArgument, std::string>
NetworkPolicyArgument::parse(std::string_view text) {
    if (text.empty()) {
        return std::unexpected(
            "the network policy needs an argument: ask, or the path of a policy file");
    }
    if (text == kAskArgument) {
        return NetworkPolicyArgument{.source = NetworkPolicySource::ask};
    }
    return NetworkPolicyArgument{.source = NetworkPolicySource::file,
                                 .file = std::filesystem::path(text)};
}

std::expected<std::shared_ptr<const IPolicy>, std::string>
NetworkPolicyFamily::create(std::string_view argument) const {
    const auto parsed = NetworkPolicyArgument::parse(argument);
    if (!parsed) return std::unexpected(parsed.error());

    // The "ask" policy has no rules: every destination the user did not
    // configure reaches the person who is asked about it.
    const auto rules = parsed->source == NetworkPolicySource::ask
        ? std::expected<PolicySet, std::string>{PolicySet({}, Effect::ask)}
        : load_policy_file(parsed->file);
    if (!rules) return std::unexpected(rules.error());

    // The channel stays unattached until the user interface exists. In a
    // headless session it never attaches, so every request that needs a
    // decision is denied.
    auto channel = std::make_shared<DeferredApprovalChannel>();
    auto snapshot = std::make_shared<PolicySnapshot>(
        std::move(*rules), std::make_shared<GrantStore>(), std::move(channel));
    return std::make_shared<const NetworkPolicy>(std::move(snapshot));
}

NetworkPolicy::NetworkPolicy(std::shared_ptr<const PolicySnapshot> snapshot)
    : snapshot_(std::move(snapshot)) {}

std::expected<void, std::string> NetworkPolicy::install() const {
    // The policy lives in the process-wide registry: the model provider, the
    // MCP sessions and the tools all read it from there, so they cannot end
    // up under different rules.
    if (!install_active_policy(snapshot_))
        return std::unexpected("a network policy is already installed");
    return {};
}

void NetworkPolicy::attach_approval_channel(std::shared_ptr<IApprovalChannel> channel) const {
    // A channel that is already attached means a decision could be answered
    // twice. The policy keeps the first one.
    (void)snapshot_->attach_approval_channel(std::move(channel));
}

std::string NetworkPolicy::describe() const {
    return std::format("network: {}", snapshot_->describe());
}

} // namespace core::policy
