#include "PolicyFamily.hpp"

#include "NetworkPolicyFamily.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace core::policy {

std::span<const std::shared_ptr<const IPolicyFamily>> policy_families() {
    // A new kind of policy is one more entry here. Nothing else in the
    // application changes.
    static const std::vector<std::shared_ptr<const IPolicyFamily>> families = {
        std::make_shared<const NetworkPolicyFamily>()};
    return families;
}

std::optional<std::shared_ptr<const IPolicyFamily>> find_policy_family(
    std::string_view name) {
    const auto families = policy_families();
    const auto found = std::ranges::find_if(families, [&](const auto& family) {
        return family->name() == name;
    });
    if (found == families.end()) return std::nullopt;
    return *found;
}

std::expected<PolicyOption, std::string> parse_policy_option(std::string_view text) {
    const auto separator = text.find('=');
    PolicyOption option{
        .family = std::string(text.substr(0, separator)),
        .argument = separator == std::string_view::npos
            ? std::string()
            : std::string(text.substr(separator + 1))};
    if (option.family.empty())
        return std::unexpected("a policy option needs a policy name, as in network=ask");
    if (option.family.find_first_of(" \t") != std::string::npos)
        return std::unexpected("a policy name cannot contain spaces: " + option.family);
    return option;
}

} // namespace core::policy
