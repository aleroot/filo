#pragma once

// ---------------------------------------------------------------------------
// InterruptSignal — a SIGINT that the work stops on, rather than a process
// that dies.
//
// The handler does nothing but set a flag: stopping the work needs locks and
// threads, which a signal handler must not touch. The owner reads the flag
// and stops the work on its own thread, so the endpoint is told that the
// response may stop before the request goes away.
// ---------------------------------------------------------------------------

#include <string_view>

namespace core::utils {

/// Watches for SIGINT (and SIGTERM) while it is alive.
///
/// The handler that was installed before is restored when the watcher goes
/// away, so a program that owns the signal keeps it.
class InterruptSignal {
public:
    /// Starts watching. A second watcher shares the same flag.
    InterruptSignal();
    ~InterruptSignal();

    InterruptSignal(const InterruptSignal&) = delete;
    InterruptSignal& operator=(const InterruptSignal&) = delete;

    /// Whether the process was asked to stop.
    [[nodiscard]] static bool interrupted() noexcept;

    /// Forgets an earlier interrupt, so a later run starts clean.
    static void clear() noexcept;

    /// The signal that asked the process to stop, empty when none did.
    [[nodiscard]] static std::string_view signal_name() noexcept;
};

} // namespace core::utils
