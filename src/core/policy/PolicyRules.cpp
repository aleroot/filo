#include "PolicyRules.hpp"

#include <arpa/inet.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstring>
#include <unordered_set>
#include <utility>

namespace core::policy {
namespace {

[[nodiscard]] std::string_view trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
        text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
        text.remove_suffix(1);
    return text;
}

// Conservative ordering used to break ties between equally narrow rules.
[[nodiscard]] constexpr int effect_rank(Effect effect) noexcept {
    switch (effect) {
        case Effect::deny: return 2;
        case Effect::ask: return 1;
        case Effect::allow: return 0;
    }
    std::unreachable();
}

} // namespace

std::string normalize_host(std::string_view host) {
    const auto trimmed = trim(host);
    if (trimmed.size() >= 2 && trimmed.front() == '[' && trimmed.back() == ']')
        host = trimmed.substr(1, trimmed.size() - 2);
    else
        host = trimmed;
    if (host.empty() || host.size() > 253) return {};

    std::string out;
    out.reserve(host.size());
    bool has_colon = false;
    for (const char raw : host) {
        const char ch = raw >= 'A' && raw <= 'Z'
            ? static_cast<char>(raw - 'A' + 'a')
            : raw;
        const bool allowed = (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')
            || ch == '-' || ch == '.' || ch == '_' || ch == ':';
        if (!allowed) return {};
        if (ch == ':') has_colon = true;
        out.push_back(ch);
    }
    if (!out.empty() && out.back() == '.') out.pop_back();
    if (out.empty() || out.front() == '.' || out.front() == '-') return {};
    if (out.contains("..")) return {};
    // A colon is only meaningful inside an IPv6 literal.
    if (has_colon) {
        for (const char ch : out) {
            const bool hex = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')
                || ch == ':' || ch == '.';
            if (!hex) return {};
        }
    }
    return out;
}

HostPattern HostPattern::any() {
    HostPattern pattern;
    pattern.kind_ = Kind::any;
    pattern.text_ = "*";
    return pattern;
}

HostPattern HostPattern::exact(std::string host) {
    HostPattern pattern;
    pattern.kind_ = Kind::exact;
    pattern.text_ = std::move(host);
    return pattern;
}

HostPattern HostPattern::suffix(std::string domain) {
    HostPattern pattern;
    pattern.kind_ = Kind::suffix;
    pattern.text_ = std::move(domain);
    return pattern;
}

HostPattern HostPattern::cidr(std::string text) {
    HostPattern pattern;
    pattern.kind_ = Kind::cidr;
    pattern.text_ = std::move(text);
    return pattern;
}

std::expected<HostPattern, std::string> HostPattern::parse(std::string_view text) {
    const auto trimmed = trim(text);
    if (trimmed.empty()) return std::unexpected("host pattern is empty");
    if (trimmed == "*") return HostPattern::any();

    if (trimmed.contains('/')) {
        const auto slash = trimmed.find('/');
        const auto address_text = trimmed.substr(0, slash);
        const auto prefix_text = trimmed.substr(slash + 1);
        int prefix = 0;
        const auto [end, error] = std::from_chars(
            prefix_text.data(), prefix_text.data() + prefix_text.size(), prefix);
        if (prefix_text.empty() || error != std::errc{} || end != prefix_text.data() + prefix_text.size())
            return std::unexpected("host pattern has an invalid prefix length");
        if (prefix > 128)
            return std::unexpected("host pattern has an out-of-range prefix length");

        const std::string address(address_text);
        in_addr v4{};
        if (::inet_pton(AF_INET, address.c_str(), &v4) == 1) {
            if (prefix > 32)
                return std::unexpected("host pattern has an out-of-range IPv4 prefix length");
            HostPattern pattern = HostPattern::cidr(std::string(trimmed));
            pattern.prefix_ = prefix;
            std::copy(reinterpret_cast<const std::uint8_t*>(&v4.s_addr),
                      reinterpret_cast<const std::uint8_t*>(&v4.s_addr) + 4,
                      pattern.network4_.begin());
            // Clear the host bits so the range matches on the prefix alone.
            for (int bit = prefix; bit < 32; ++bit) {
                pattern.network4_[static_cast<std::size_t>(bit / 8)] &=
                    static_cast<std::uint8_t>(~(1u << (7 - (bit % 8))));
            }
            return pattern;
        }
        in6_addr v6{};
        if (::inet_pton(AF_INET6, address.c_str(), &v6) == 1) {
            HostPattern pattern = HostPattern::cidr(std::string(trimmed));
            pattern.prefix_ = prefix;
            pattern.ipv6_ = true;
            std::copy(std::begin(v6.s6_addr), std::end(v6.s6_addr), pattern.network6_.begin());
            return pattern;
        }
        return std::unexpected("host pattern is not a valid address range");
    }

    if (trimmed.starts_with("*.")) {
        const auto domain = normalize_host(trimmed.substr(2));
        if (domain.empty()) return std::unexpected("host pattern has an invalid domain");
        return HostPattern::suffix(domain);
    }

    const auto host = normalize_host(trimmed);
    if (host.empty()) return std::unexpected("host pattern has an invalid host");
    return HostPattern::exact(host);
}

bool HostPattern::matches(std::string_view host) const noexcept {
    if (kind_ == Kind::any) return true;
    const auto normalized = normalize_host(host);
    if (normalized.empty()) return false;
    switch (kind_) {
        case Kind::any:
            return true;
        case Kind::exact:
            return normalized == text_;
        case Kind::suffix: {
            // The domain itself, or a name under it.
            if (normalized == text_) return true;
            const std::string suffix = "." + text_;
            return normalized.ends_with(suffix);
        }
        case Kind::cidr:
            return in_cidr(normalized);
    }
    std::unreachable();
}

int HostPattern::specificity() const noexcept {
    switch (kind_) {
        case Kind::any: return 0;
        case Kind::suffix: return 1;
        case Kind::cidr: return 2;
        case Kind::exact: return 3;
    }
    std::unreachable();
}

std::string HostPattern::text() const {
    if (kind_ == Kind::suffix) return "*." + text_;
    return text_;
}

bool HostPattern::in_cidr(const std::string& host) const noexcept {
    if (!ipv6_) {
        in_addr address{};
        if (::inet_pton(AF_INET, host.c_str(), &address) != 1) return false;
        const auto* bytes = reinterpret_cast<const unsigned char*>(&address.s_addr);
        for (int bit = 0; bit < prefix_; ++bit) {
            const auto mask = static_cast<std::uint8_t>(0x80u >> (bit % 8));
            const auto index = static_cast<std::size_t>(bit / 8);
            if ((network4_[index] & mask) != (bytes[index] & mask)) return false;
        }
        return true;
    }
    in6_addr address{};
    if (::inet_pton(AF_INET6, host.c_str(), &address) != 1) return false;
    // An IPv4-mapped address is not a member of an IPv6 range: keeping the two
    // families apart avoids a rule written for one silently covering the other.
    if (IN6_IS_ADDR_V4MAPPED(&address)) return false;
    const int full_bytes = prefix_ / 8;
    const int remaining_bits = prefix_ % 8;
    if (full_bytes > 0
        && std::memcmp(network6_.data(), address.s6_addr, static_cast<std::size_t>(full_bytes)) != 0)
        return false;
    if (remaining_bits == 0) return true;
    const auto mask = static_cast<std::uint8_t>(0xFFu << (8 - remaining_bits));
    return (network6_[static_cast<std::size_t>(full_bytes)] & mask)
        == (address.s6_addr[full_bytes] & mask);
}

PortRange PortRange::any() {
    return {};
}

PortRange PortRange::exact(std::uint16_t port) {
    PortRange range;
    range.low_ = port;
    range.high_ = port;
    range.any_ = false;
    range.exact_ = true;
    return range;
}

PortRange PortRange::range(std::uint16_t low, std::uint16_t high) {
    PortRange range;
    range.low_ = low;
    range.high_ = high;
    range.any_ = false;
    range.exact_ = false;
    return range;
}

std::expected<PortRange, std::string> PortRange::parse(std::string_view text) {
    const auto trimmed = trim(text);
    if (trimmed.empty() || trimmed == "*") return PortRange::any();

    const auto parse_port = [](std::string_view value) -> std::optional<std::uint16_t> {
        unsigned parsed = 0;
        const auto [end, error] = std::from_chars(
            value.data(), value.data() + value.size(), parsed);
        if (value.empty() || error != std::errc{} || end != value.data() + value.size()
            || parsed == 0 || parsed > 65535)
            return std::nullopt;
        return static_cast<std::uint16_t>(parsed);
    };

    const auto dash = trimmed.find('-');
    if (dash == std::string_view::npos) {
        const auto port = parse_port(trimmed);
        if (!port) return std::unexpected("port is not a valid port number");
        return PortRange::exact(*port);
    }
    const auto low = parse_port(trimmed.substr(0, dash));
    const auto high = parse_port(trimmed.substr(dash + 1));
    if (!low || !high) return std::unexpected("port range is not a valid port range");
    if (*low > *high) return std::unexpected("port range is inverted");
    return PortRange::range(*low, *high);
}

bool PortRange::contains(std::uint16_t port) const noexcept {
    if (any_) return true;
    return port >= low_ && port <= high_;
}

int PortRange::specificity() const noexcept {
    if (any_) return 0;
    return exact_ ? 2 : 1;
}

std::string PortRange::text() const {
    if (any_) return "*";
    if (exact_) return std::to_string(low_);
    return std::to_string(low_) + "-" + std::to_string(high_);
}

bool PolicyRule::matches(const Subject& subject,
                          const Action& action,
                          const Resource& resource) const noexcept {
    if (principal && *principal != subject.kind) return false;
    if (principal_id != "*" && principal_id != subject.id) return false;
    if (operation && *operation != action.operation) return false;
    if (flow && *flow != action.flow) return false;
    if (!host.matches(resource.host)) return false;
    if (!port.contains(resource.port)) return false;
    return true;
}

std::array<int, 5> PolicyRule::specificity() const noexcept {
    int principal_score = 0;
    if (principal) ++principal_score;
    if (principal_id != "*") ++principal_score;
    return {host.specificity(),
            port.specificity(),
            principal_score,
            operation ? 1 : 0,
            flow ? 1 : 0};
}

PolicySet::PolicySet(std::vector<PolicyRule> rules, Effect default_effect)
    : rules_(std::move(rules)), default_effect_(default_effect) {
    std::unordered_set<std::string> seen;
    for (const auto& rule : rules_) {
        if (rule.id.empty()) continue;
        if (!seen.insert(rule.id).second) {
            valid_ = false;
            invalid_reason_ = "duplicate rule id \"" + rule.id + "\"";
            return;
        }
    }
}

Decision PolicySet::evaluate(const Subject& subject,
                              const Action& action,
                              const Resource& resource) const {
    if (!valid_)
        return {Effect::deny, "network policy is invalid: " + invalid_reason_};

    const PolicyRule* best = nullptr;
    for (const auto& rule : rules_) {
        if (!rule.matches(subject, action, resource)) continue;
        if (best == nullptr) {
            best = &rule;
            continue;
        }
        const auto candidate = rule.specificity();
        const auto current = best->specificity();
        if (candidate > current) {
            best = &rule;
            continue;
        }
        if (candidate == current && effect_rank(rule.effect) > effect_rank(best->effect))
            best = &rule;
    }

    if (best == nullptr) {
        return {default_effect_,
                default_effect_ == Effect::deny ? "no policy rule matched"
                                                : "no policy rule matched, policy default applies"};
    }
    Decision decision{best->effect,
                      best->id.empty() ? "matched an unnamed policy rule"
                                       : "matched policy rule \"" + best->id + "\"",
                      best->id};
    decision.matched_rule = true;
    return decision;
}

} // namespace core::policy
