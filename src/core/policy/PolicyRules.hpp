#pragma once

#include "PolicyTypes.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::policy {

/**
 * The rules of a policy: what a request is matched against, and the decision
 * a matching rule produces.
 *
 * The vocabulary of a policy (Effect, Subject, Action, Resource) lives in
 * PolicyTypes.hpp. Everything that decides with that vocabulary lives here.
 */

/**
 * Canonicalizes a host for comparison.
 *
 * Trims surrounding whitespace, strips IPv6 brackets, lower-cases ASCII
 * letters and removes a single trailing dot. Returns an empty string when the
 * result is not a plausible host name or IP literal, so a malformed value
 * never matches a rule.
 */
[[nodiscard]] std::string normalize_host(std::string_view host);

/**
 * A destination host matcher.
 *
 * Matching is purely textual: it never resolves a name. A name that resolves
 * to an unexpected address is caught by the transport adapter, which binds the
 * policy decision to the address actually connected to.
 *
 * Four shapes are supported:
 *
 *   any     "*"                  matches every host
 *   exact   "example.com"        matches exactly that host
 *   suffix  "*.example.com"      matches example.com and any subdomain
 *   cidr    "10.0.0.0/8"         matches IP literals inside the range
 *
 * A suffix pattern also matches the domain itself, so "*.example.com" and
 * "example.com" describe the same reachable set. That is deliberate: an
 * allowlist that excluded the apex host would surprise operators.
 */
class HostPattern {
public:
    enum class Kind { any, exact, suffix, cidr };

    HostPattern() = default;

    [[nodiscard]] static HostPattern any();
    [[nodiscard]] static HostPattern exact(std::string host);
    // Matches the domain itself and every subdomain of it.
    [[nodiscard]] static HostPattern suffix(std::string domain);
    [[nodiscard]] static HostPattern cidr(std::string text);

    // Parses "*", "example.com", "*.example.com" or "10.0.0.0/8".
    [[nodiscard]] static std::expected<HostPattern, std::string> parse(std::string_view text);

    [[nodiscard]] bool matches(std::string_view host) const noexcept;
    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    // Higher is narrower. Used to decide which matching rule wins.
    [[nodiscard]] int specificity() const noexcept;
    // The canonical text of the pattern, for diagnostics.
    [[nodiscard]] std::string text() const;

private:
    [[nodiscard]] bool in_cidr(const std::string& host) const noexcept;

    Kind kind_{Kind::any};
    std::string text_;
    // CIDR state. Both families keep their address bytes in network order, so
    // a comparison never depends on the host's byte order.
    std::array<std::uint8_t, 4> network4_{};
    std::array<std::uint8_t, 16> network6_{};
    int prefix_{0};
    bool ipv6_{false};
};

/**
 * A destination port matcher.
 *
 * Three shapes are supported: any port ("*"), one port ("443") and an
 * inclusive range ("8000-9000").
 */
class PortRange {
public:
    PortRange() = default;

    [[nodiscard]] static PortRange any();
    [[nodiscard]] static PortRange exact(std::uint16_t port);
    [[nodiscard]] static PortRange range(std::uint16_t low, std::uint16_t high);

    // Parses "*", "443" or "8000-9000".
    [[nodiscard]] static std::expected<PortRange, std::string> parse(std::string_view text);

    [[nodiscard]] bool contains(std::uint16_t port) const noexcept;
    // Higher is narrower. Used to decide which matching rule wins.
    [[nodiscard]] int specificity() const noexcept;
    [[nodiscard]] std::string text() const;

private:
    std::uint16_t low_{0};
    std::uint16_t high_{0};
    bool any_{true};
    bool exact_{false};
};

/**
 * One rule of a network policy.
 *
 * Every dimension is optional, and an omitted dimension matches everything.
 * A rule that constrains nothing but the effect is therefore a global
 * default for that effect.
 *
 * A rule is matched against a (subject, action, resource) triple. The
 * resource is matched textually: a name is never resolved here.
 */
struct PolicyRule {
    // Optional label used in decisions and diagnostics. Labels must be
    // unique inside one policy so a decision can name the rule that made it.
    std::string id;
    Effect effect{Effect::deny};

    // Which principal kind the rule applies to. Unset matches every kind.
    std::optional<PrincipalKind> principal;
    // Which principal within that kind. "*" matches every identifier.
    std::string principal_id{"*"};

    // Which operation the rule applies to. Unset matches every operation.
    std::optional<OperationKind> operation;
    // Which data flow the rule applies to. Unset matches every flow.
    std::optional<DataFlow> flow;

    // Which destination the rule applies to.
    HostPattern host;
    PortRange port;

    [[nodiscard]] bool matches(const Subject& subject,
                               const Action& action,
                               const Resource& resource) const noexcept;

    /**
     * How narrowly the rule constrains the request.
     *
     * Compared lexicographically: a narrower destination wins over a broader
     * one, a narrower port over a broader one, and so on. This is what lets an
     * allowlist coexist with a broad deny rule instead of cancelling it.
     */
    [[nodiscard]] std::array<int, 5> specificity() const noexcept;
};

/**
 * An ordered set of rules plus the effect used when nothing matches.
 *
 * Evaluation never fails open. A malformed policy denies every request and
 * says why, because a policy that cannot be understood must not be treated as
 * absent.
 *
 * When several rules match, the narrowest one decides. Ties are broken
 * conservatively: deny beats ask beats allow.
 */
class PolicySet {
public:
    // The rule set is always explicit: a default-constructed policy would be
    // ambiguous with the copy constructor and would hide the fact that an
    // empty rule set is a deliberate choice.
    explicit PolicySet(std::vector<PolicyRule> rules, Effect default_effect = Effect::deny);

    [[nodiscard]] Decision evaluate(const Subject& subject,
                                    const Action& action,
                                    const Resource& resource) const;

    // False when the rule set cannot be evaluated as written.
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] const std::string& invalid_reason() const noexcept { return invalid_reason_; }
    [[nodiscard]] Effect default_effect() const noexcept { return default_effect_; }
    [[nodiscard]] const std::vector<PolicyRule>& rules() const noexcept { return rules_; }
    [[nodiscard]] std::size_t size() const noexcept { return rules_.size(); }

private:
    std::vector<PolicyRule> rules_;
    Effect default_effect_{Effect::deny};
    bool valid_{true};
    std::string invalid_reason_;
};

} // namespace core::policy
