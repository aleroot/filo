#pragma once

#include "../SleepInhibitor.hpp"

namespace core::power {

class LinuxSleepInhibitor final : public SleepInhibitor {
public:
    [[nodiscard]] std::shared_ptr<SleepInhibitionLease>
    inhibit(std::string_view reason) override;
};

} // namespace core::power
