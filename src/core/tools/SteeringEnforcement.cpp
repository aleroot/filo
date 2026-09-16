#include "SteeringEnforcement.hpp"

#include "../context/SessionContext.hpp"
#include "../landrun/LandrunPathProtection.hpp"
#include "../utils/JsonUtils.hpp"

#include <format>
#include <utility>

namespace core::tools {

namespace {

/// The single shape every tool-facing refusal takes.
[[nodiscard]] std::string error_json(std::string_view message) {
    return std::format(R"({{"error":"{}"}})",
                       core::utils::escape_json_string(message));
}

} // namespace

SteeringEnforcement SteeringEnforcement::for_guard(
    core::context::SteeringGuard guard, Tier tier)
{
    SteeringEnforcement enforcement;
    // One invariant, established at construction and relied on everywhere else:
    // a tier other than `none` means there is something blocked, and vice versa.
    if (guard.empty()) {
        return enforcement;
    }
    enforcement.guard_ = std::move(guard);
    enforcement.tier_ = tier;
    return enforcement;
}

SteeringEnforcement SteeringEnforcement::for_context(
    const core::context::SessionContext& context)
{
    auto guard = core::context::SteeringGuard::for_context(context);
    if (guard.empty()) {
        return {};
    }
    // Whether the kernel can be the authority is a property of this host, not of
    // the session, and answering it costs a spawned child — so it is verified
    // once per process inside landrun_protects_paths().
    return for_guard(std::move(guard),
                     core::landrun::landrun_protects_paths() ? Tier::kernel
                                                             : Tier::heuristic);
}

core::landrun::LandrunPolicy SteeringEnforcement::child_policy(
    core::landrun::LandrunPolicy policy) const
{
    if (tier_ != Tier::kernel) {
        return policy;
    }
    for (const auto& path : guard_.blocked_paths()) {
        core::landrun::add_protected_read_path(policy, path);
    }
    return policy;
}

std::optional<std::string> SteeringEnforcement::command_error(
    std::string_view command,
    const std::filesystem::path& base_dir) const
{
    if (tier_ != Tier::heuristic) {
        return std::nullopt;
    }
    const auto reason = guard_.blocked_text_reason(command, base_dir);
    return reason ? std::make_optional(error_json(*reason)) : std::nullopt;
}

std::optional<std::string> SteeringEnforcement::in_process_code_error() const {
    if (!blocks_anything()) {
        return std::nullopt;
    }
    return error_json(
        "Access denied: this session's steering policy withholds project instruction files, and an "
        "interpreter running inside Filo is not covered by the OS sandbox that makes those files "
        "unreadable to child processes. Run the code with run_terminal_command instead, where the "
        "same restriction is enforced; ask the user to re-enable steering with /steering if the "
        "in-process interpreter is required.");
}

} // namespace core::tools
