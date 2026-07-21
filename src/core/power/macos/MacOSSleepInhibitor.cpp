#include "MacOSSleepInhibitor.hpp"

#include "core/logging/Logger.hpp"

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#endif

#include <memory>
#include <string>
#include <type_traits>

namespace core::power {

#if defined(__APPLE__)
namespace {

struct CFReleaseDeleter {
    void operator()(CFTypeRef value) const noexcept {
        if (value != nullptr) {
            CFRelease(value);
        }
    }
};

using CFStringHandle = std::unique_ptr<
    std::remove_pointer_t<CFStringRef>,
    CFReleaseDeleter>;

class MacOSSleepInhibitionLease final : public SleepInhibitionLease {
public:
    explicit MacOSSleepInhibitionLease(IOPMAssertionID assertion_id) noexcept
        : assertion_id_(assertion_id) {}

    ~MacOSSleepInhibitionLease() override {
        if (assertion_id_ != kIOPMNullAssertionID) {
            IOPMAssertionRelease(assertion_id_);
        }
    }

    [[nodiscard]] bool is_active() const noexcept override {
        return assertion_id_ != kIOPMNullAssertionID;
    }

private:
    IOPMAssertionID assertion_id_ = kIOPMNullAssertionID;
};

} // namespace
#endif

std::shared_ptr<SleepInhibitionLease>
MacOSSleepInhibitor::inhibit(std::string_view reason) {
#if !defined(__APPLE__)
    (void)reason;
    return {};
#else
    const std::string reason_text = reason.empty()
        ? "Filo is working"
        : std::string(reason);
    const CFStringHandle reason_string{CFStringCreateWithBytes(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(reason_text.data()),
        static_cast<CFIndex>(reason_text.size()),
        kCFStringEncodingUTF8,
        false)};
    if (!reason_string) {
        core::logging::warn("Could not create the macOS sleep-inhibition reason");
        return {};
    }

    IOPMAssertionID assertion_id = kIOPMNullAssertionID;
    const IOReturn result = IOPMAssertionCreateWithDescription(
        kIOPMAssertPreventUserIdleSystemSleep,
        CFSTR("Filo agent turn"),
        reason_string.get(),
        nullptr,
        nullptr,
        0,
        nullptr,
        &assertion_id);

    if (result != kIOReturnSuccess) {
        core::logging::warn(
            "Could not prevent macOS idle system sleep (IOKit error {})",
            result);
        return {};
    }
    return std::make_shared<MacOSSleepInhibitionLease>(assertion_id);
#endif
}

} // namespace core::power
