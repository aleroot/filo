#include "SleepInhibitor.hpp"
#include "linux/LinuxSleepInhibitor.hpp"
#include "macos/MacOSSleepInhibitor.hpp"

namespace core::power {
namespace {

class UnsupportedSleepInhibitor final : public SleepInhibitor {
public:
    [[nodiscard]] std::shared_ptr<SleepInhibitionLease>
    inhibit(std::string_view) override {
        return {};
    }
};

} // namespace

std::shared_ptr<SleepInhibitor> make_sleep_inhibitor() {
#if defined(__APPLE__)
    return std::make_shared<MacOSSleepInhibitor>();
#elif defined(__linux__)
    return std::make_shared<LinuxSleepInhibitor>();
#else
    return std::make_shared<UnsupportedSleepInhibitor>();
#endif
}

} // namespace core::power
