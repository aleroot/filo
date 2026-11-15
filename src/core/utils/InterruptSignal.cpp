#include "InterruptSignal.hpp"

#include <atomic>
#include <csignal>

namespace core::utils {
namespace {

// The handler may only touch a lock-free flag.
std::atomic_bool g_interrupted{false};
std::atomic<int> g_signal{0};

extern "C" void handle_interrupt(int signal_number) {
    g_signal.store(signal_number, std::memory_order_relaxed);
    g_interrupted.store(true, std::memory_order_release);
}

/// The handlers that were installed before, and how many watchers want them.
struct InstalledHandlers {
    struct sigaction interrupt {};
    struct sigaction terminate {};
    bool installed = false;
};

InstalledHandlers g_installed;
std::atomic<int> g_watchers{0};

void install_handler(int signal_number, struct sigaction& previous) {
    struct sigaction action {};
    action.sa_handler = handle_interrupt;
    sigemptyset(&action.sa_mask);
    // No SA_RESTART: a blocked read returns EINTR, so the owner notices.
    action.sa_flags = 0;
    if (::sigaction(signal_number, &action, &previous) == 0) {
        g_installed.installed = true;
    }
}

} // namespace

InterruptSignal::InterruptSignal() {
    if (g_watchers.fetch_add(1, std::memory_order_acq_rel) == 0) {
        install_handler(SIGINT, g_installed.interrupt);
        install_handler(SIGTERM, g_installed.terminate);
    }
}

InterruptSignal::~InterruptSignal() {
    if (g_watchers.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    if (::sigaction(SIGINT, &g_installed.interrupt, nullptr) != 0) return;
    ::sigaction(SIGTERM, &g_installed.terminate, nullptr);
}

bool InterruptSignal::interrupted() noexcept {
    return g_interrupted.load(std::memory_order_acquire);
}

void InterruptSignal::clear() noexcept {
    g_interrupted.store(false, std::memory_order_release);
    g_signal.store(0, std::memory_order_relaxed);
}

std::string_view InterruptSignal::signal_name() noexcept {
    switch (g_signal.load(std::memory_order_relaxed)) {
        case SIGINT:
            return "SIGINT";
        case SIGTERM:
            return "SIGTERM";
        default:
            return {};
    }
}

} // namespace core::utils
