#include "LandrunPathProtection.hpp"

#include "LandrunDriverFactory.hpp"
#include "LandrunPolicy.hpp"
#include "LandrunReadiness.hpp"

#include <filesystem>
#include <system_error>

namespace core::landrun {

namespace {

[[nodiscard]] bool verify_path_protection() {
    const auto driver = make_landrun_driver();
    if (!driver->supports_protected_paths()) return false;

    // The sample subtracts a path nothing in the probe needs, so a success
    // proves the mechanism and a failure can only mean the mechanism.
    std::error_code ec;
    auto scratch = std::filesystem::temp_directory_path(ec);
    if (ec) scratch = "/tmp";
    LandrunPolicy sample;
    add_protected_read_path(sample, scratch / "filo-landrun-protection-probe");
    return sample.protects_paths() && verify_landrun_readiness(sample).success;
}

} // namespace

bool landrun_protects_paths() {
    static const bool verified = verify_path_protection();
    return verified;
}

} // namespace core::landrun
