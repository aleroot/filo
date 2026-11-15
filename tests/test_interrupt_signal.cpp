#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/utils/InterruptSignal.hpp"

#include <csignal>

TEST_CASE("An interrupt is recorded for the owner to act on", "[utils][interrupt]") {
    const core::utils::InterruptSignal watcher;
    core::utils::InterruptSignal::clear();
    CHECK_FALSE(core::utils::InterruptSignal::interrupted());
    CHECK(core::utils::InterruptSignal::signal_name().empty());

    // The handler does nothing but set the flag: stopping the work is the
    // owner's job, on its own thread.
    ::raise(SIGINT);
    CHECK(core::utils::InterruptSignal::interrupted());
    CHECK(core::utils::InterruptSignal::signal_name() == "SIGINT");

    core::utils::InterruptSignal::clear();
    CHECK_FALSE(core::utils::InterruptSignal::interrupted());
    CHECK(core::utils::InterruptSignal::signal_name().empty());
}

TEST_CASE("A watcher leaves the signal it found in place", "[utils][interrupt]") {
    // A program that owns the signal keeps it: the watcher remembers what was
    // installed and puts it back.
    struct sigaction previous {};
    struct sigaction ignored {};
    ignored.sa_handler = SIG_IGN;
    sigemptyset(&ignored.sa_mask);
    REQUIRE(::sigaction(SIGINT, &ignored, &previous) == 0);

    {
        const core::utils::InterruptSignal watcher;
        core::utils::InterruptSignal::clear();
        ::raise(SIGINT);
        CHECK(core::utils::InterruptSignal::interrupted());
    }

    // The handler that was there before is back: an interrupt is ignored.
    core::utils::InterruptSignal::clear();
    ::raise(SIGINT);
    CHECK_FALSE(core::utils::InterruptSignal::interrupted());

    struct sigaction restored {};
    REQUIRE(::sigaction(SIGINT, &previous, &restored) == 0);
}
