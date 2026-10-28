#pragma once

#include "PolicyFamily.hpp"
#include "PolicySnapshot.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace core::policy {

/**
 * What a network policy is built from.
 */
enum class NetworkPolicySource {
    // Ask before every destination the user did not configure.
    ask,
    // Load the rules from a policy file.
    file,
};

/**
 * The argument of the network policy family.
 *
 * The argument is one of two things: the reserved word "ask", which asks
 * before every destination the user did not configure, or the path of a
 * policy file. Parsing it into a value keeps the grammar of the option in
 * one named place, so no other part of the program recognises the word or
 * guesses whether a string names a file.
 */
struct NetworkPolicyArgument {
    NetworkPolicySource source{NetworkPolicySource::ask};
    std::filesystem::path file;

    /**
     * Parses the argument of the network family.
     *
     * @return The argument, or why it cannot be used.
     */
    [[nodiscard]] static std::expected<NetworkPolicyArgument, std::string>
    parse(std::string_view text);
};

/**
 * The network policy: where this process may send data.
 *
 * The argument is `ask`, which asks before a destination the user did not
 * configure, or the path of a policy file. The policy covers the model
 * provider's requests, the remote MCP servers and their sign-in renewals,
 * and the web tools that fetch a document.
 */
class NetworkPolicyFamily final : public IPolicyFamily {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "network"; }

    [[nodiscard]] std::string_view usage() const noexcept override {
        return "network=ask|<file>";
    }

    // The child processes Filo launches must be confined for their traffic
    // to be denied, so the network policy only holds with the OS sandbox on.
    [[nodiscard]] bool requires_sandbox() const noexcept override { return true; }

    [[nodiscard]] std::expected<std::shared_ptr<const IPolicy>, std::string>
    create(std::string_view argument) const override;
};

/**
 * A configured network policy.
 *
 * It holds the rules, the store of standing authorizations and the channel
 * that asks a person. The channel is attached after the user interface
 * exists; until then a request that needs a decision is denied.
 */
class NetworkPolicy final : public IPolicy {
public:
    explicit NetworkPolicy(std::shared_ptr<const PolicySnapshot> snapshot);

    [[nodiscard]] std::expected<void, std::string> install() const override;

    void attach_approval_channel(std::shared_ptr<IApprovalChannel> channel) const override;

    [[nodiscard]] std::string describe() const override;

    [[nodiscard]] const PolicySnapshot& snapshot() const noexcept { return *snapshot_; }

private:
    std::shared_ptr<const PolicySnapshot> snapshot_;
};

} // namespace core::policy
