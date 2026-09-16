#pragma once

#include "../context/SteeringGuard.hpp"
#include "../landrun/LandrunPolicy.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace core::context {
struct SessionContext;
}

namespace core::tools {

/**
 * How a session's withheld steering is enforced against code the agent runs.
 *
 * @ref core::context::SteeringGuard answers "which steering files did this
 * policy explicitly withhold?". This type answers the second, different
 * question: "what stops the agent from fetching them anyway?" — and the answer
 * depends on the host, so it is decided once here instead of being rediscovered
 * at each call site.
 *
 * Two surfaces need it, and they cannot share a mechanism:
 *
 *  - *Child processes* (the persistent shell, the code-block runner, anything
 *    spawned by them) can be confined by the OS. When the host can subtract
 *    individual paths from a granted view, the blocked files are handed to
 *    Landrun as protected paths and the kernel is the authority. No amount of
 *    quoting, brace expansion, base64 or `$( )` reaches them, because nothing
 *    in Filo inspects the command text at all.
 *  - *In-process code* (the embedded Python interpreter) runs inside Filo,
 *    which is deliberately unconfined. There is no kernel boundary to hide
 *    behind, so the interpreter is refused while steering is withheld — the
 *    same call the sandbox makes for in-process untrusted code, and the only
 *    answer that is not a guess about what a snippet of source will open.
 *
 * Text inspection survives as a fallback, and only as one: on a host whose
 * backend cannot express a subtraction (Landlock is allow-list only, so
 * `--sandbox` on Linux), or when no sandbox applies at all.
 */
class SteeringEnforcement {
public:
    enum class Tier {
        /// Nothing is withheld, so there is nothing to enforce.
        none,
        /// The OS denies the paths itself; command text is never inspected.
        kernel,
        /// No backend here can subtract paths, so references are matched textually.
        heuristic,
    };

    SteeringEnforcement() = default;

    [[nodiscard]] static SteeringEnforcement for_context(
        const core::context::SessionContext& context);

    /// Explicit construction for tests and for callers that already hold a guard.
    [[nodiscard]] static SteeringEnforcement for_guard(
        core::context::SteeringGuard guard, Tier tier);

    [[nodiscard]] Tier tier() const noexcept { return tier_; }
    [[nodiscard]] bool blocks_anything() const noexcept { return tier_ != Tier::none; }

    /**
     * @p policy as the session's child processes must run under it: the
     * withheld steering paths subtracted when the host can enforce that, and
     * returned untouched otherwise.
     *
     * Leaving them out is deliberate rather than optimistic. A backend that
     * cannot represent the subtraction fails closed on it, so adding the paths
     * on such a host would not weaken anything — it would break every command.
     */
    [[nodiscard]] core::landrun::LandrunPolicy child_policy(
        core::landrun::LandrunPolicy policy) const;

    /**
     * Tool-facing JSON error for a free-form command, or nullopt when the
     * command may run. Under Tier::kernel this is always nullopt: the command
     * runs and the kernel makes the file unreadable, which is both stronger and
     * free of the false positives textual matching produces (`grep -l x *.md`).
     */
    [[nodiscard]] std::optional<std::string> command_error(
        std::string_view command,
        const std::filesystem::path& base_dir) const;

    /**
     * Tool-facing JSON error for code about to run inside this process, or
     * nullopt when it may run.
     *
     * Withheld steering always produces an error here, whatever the tier: the
     * kernel confinement that makes Tier::kernel trustworthy applies to child
     * processes, and this code does not run in one. The alternative would be
     * pattern-matching source text and hoping, which is exactly the guessing
     * this type exists to remove.
     */
    [[nodiscard]] std::optional<std::string> in_process_code_error() const;

private:
    core::context::SteeringGuard guard_;
    Tier tier_{Tier::none};
};

} // namespace core::tools
