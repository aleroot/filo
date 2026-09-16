#pragma once

namespace core::landrun {

/**
 * Whether this host can deny individual paths to a child process tree while
 * leaving everything else as it is — a *protection-only* LandrunPolicy.
 *
 * Two things have to be true: the platform backend must support subtractive
 * rules (Seatbelt does; Landlock is allow-list only), and applying such a
 * policy must actually succeed here — which it does not when the current
 * executable is not a landrun helper dispatcher, or when the host already
 * confines Filo and refuses to stack another profile. Both are verified once
 * per process by running a disposable child under a sample policy, exactly
 * the way main() verifies confinement readiness at startup.
 *
 * Callers use the answer to choose an enforcement strategy up front; they
 * never discover the capability by watching a real command fail.
 */
[[nodiscard]] bool landrun_protects_paths();

} // namespace core::landrun
