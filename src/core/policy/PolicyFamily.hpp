#pragma once

#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace core::policy {

class IApprovalChannel;

/**
 * A policy the application enforces.
 *
 * The application knows three things about a policy: it can be built from the
 * argument the user wrote, it is installed before anything can run, and it
 * can ask a person through a channel. A new kind of policy is a new
 * implementation of this interface and one entry in the registry; nothing
 * else in the application changes.
 */
class IPolicy {
public:
    virtual ~IPolicy() = default;

    /**
     * Installs the policy.
     *
     * Called once, before any tool or request can run. A policy that cannot
     * be installed is a startup failure: continuing would claim a guarantee
     * that does not exist. Installing changes the process, not the policy, so
     * the policy itself stays immutable.
     *
     * @return Why the policy cannot be installed, if it cannot.
     */
    [[nodiscard]] virtual std::expected<void, std::string> install() const = 0;

    /**
     * Lets the user interface answer the questions the policy asks.
     *
     * A policy that never asks ignores this. The channel is attached late,
     * because the interface does not exist when the policy is built.
     */
    virtual void attach_approval_channel(std::shared_ptr<IApprovalChannel> channel) const = 0;

    // A short description of the effective policy, for status output.
    [[nodiscard]] virtual std::string describe() const = 0;
};

/**
 * A kind of policy, and the argument that configures it.
 */
class IPolicyFamily {
public:
    virtual ~IPolicyFamily() = default;

    // The name the user selects on the command line.
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    // How the argument is written, for --help.
    [[nodiscard]] virtual std::string_view usage() const noexcept = 0;

    // Whether the family can only be enforced with the OS sandbox enabled.
    [[nodiscard]] virtual bool requires_sandbox() const noexcept { return false; }

    /**
     * Builds the family from its argument.
     *
     * @return The policy, or why the argument is not usable.
     */
    [[nodiscard]] virtual std::expected<std::shared_ptr<const IPolicy>, std::string>
    create(std::string_view argument) const = 0;
};

// The families this build knows.
[[nodiscard]] std::span<const std::shared_ptr<const IPolicyFamily>> policy_families();

// The family with that name, or nullopt when there is none.
[[nodiscard]] std::optional<std::shared_ptr<const IPolicyFamily>> find_policy_family(
    std::string_view name);

// One --policy option, split into the family and the argument it configures.
struct PolicyOption {
    std::string family;
    std::string argument;
};

/**
 * Splits one --policy option.
 *
 * The form is `<family>=<argument>`: the family names the kind of policy and
 * the argument configures it. A family whose argument may be left out is
 * written without the "=".
 */
[[nodiscard]] std::expected<PolicyOption, std::string> parse_policy_option(
    std::string_view text);

} // namespace core::policy
