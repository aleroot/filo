#include "LandrunDriverFactory.hpp"
#include "linux/LinuxLandlockDriver.hpp"
#include "macos/MacOSSeatbeltDriver.hpp"

#include <string>

namespace core::landrun {
namespace {

class UnsupportedLandrunDriver final : public ILandrunDriver {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "unsupported";
    }

    [[nodiscard]] LandrunProbe probe() const override {
        return {.available = false,
                .backend = std::string(name()),
                .detail = "landrun is not implemented on this platform"};
    }

    [[nodiscard]] LandrunResult apply(const LandrunPolicy& policy) const override {
        if (!policy.enabled()) return {.success = true};
        return {.success = false,
                .detail = "landrun is not implemented on this platform"};
    }
};

} // namespace

std::unique_ptr<ILandrunDriver> make_landrun_driver() {
#if defined(__APPLE__)
    return std::make_unique<MacOSSeatbeltDriver>();
#elif defined(__linux__)
    return std::make_unique<LinuxLandlockDriver>();
#else
    return std::make_unique<UnsupportedLandrunDriver>();
#endif
}

} // namespace core::landrun
