#include "LandrunHelper.hpp"
#include "LandrunDescriptorSanitizer.hpp"
#include "LandrunDriverFactory.hpp"

#include <cerrno>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <unistd.h>

namespace core::landrun {

bool is_landrun_helper_invocation(int argc, char* const argv[]) noexcept {
    return argc >= 2 && argv && argv[1]
        && std::string_view(argv[1]) == "__landrun-exec";
}

int run_landrun_helper(int argc, char* const argv[]) {
    try {
        if (const auto result = sanitize_inherited_descriptors(); !result.success) {
            std::cerr << "[landrun] " << result.detail << '\n';
            return kLandrunHelperFailure;
        }
    } catch (const std::exception& error) {
        std::cerr << "[landrun] inherited descriptor sanitization failed: "
                  << error.what() << '\n';
        return kLandrunHelperFailure;
    }

    LandrunPolicy policy;
    bool mode_provided = false;
    int index = 0;
    for (; index < argc; ++index) {
        const std::string_view argument = argv[index] ? argv[index] : "";
        if (argument == "--") {
            ++index;
            break;
        }
        if (argument == "--mode") {
            if (++index >= argc || !argv[index]) {
                std::cerr << "[landrun] --mode requires a sandbox mode\n";
                return kLandrunHelperFailure;
            }
            const auto parsed = parse_landrun_mode(argv[index]);
            if (!parsed.has_value() || !landrun_enabled(*parsed)) {
                std::cerr << "[landrun] --mode requires an enabled sandbox mode\n";
                return kLandrunHelperFailure;
            }
            policy.mode = *parsed;
            mode_provided = true;
            continue;
        }
        if (argument == "--rw") {
            if (++index >= argc || !argv[index] || argv[index][0] != '/') {
                std::cerr << "[landrun] --rw requires an absolute path\n";
                return kLandrunHelperFailure;
            }
            add_writable_root(policy, std::filesystem::path(argv[index]));
            continue;
        }
        if (argument == "--ro") {
            if (++index >= argc || !argv[index] || argv[index][0] != '/') {
                std::cerr << "[landrun] --ro requires an absolute path\n";
                return kLandrunHelperFailure;
            }
            add_readable_root(policy, std::filesystem::path(argv[index]));
            continue;
        }
        if (argument == "--deny-read") {
            if (++index >= argc || !argv[index] || argv[index][0] != '/') {
                std::cerr << "[landrun] --deny-read requires an absolute path\n";
                return kLandrunHelperFailure;
            }
            add_protected_read_path(policy, std::filesystem::path(argv[index]));
            continue;
        }
        if (argument == "--deny-write") {
            if (++index >= argc || !argv[index] || argv[index][0] != '/') {
                std::cerr << "[landrun] --deny-write requires an absolute path\n";
                return kLandrunHelperFailure;
            }
            add_protected_write_path(policy, std::filesystem::path(argv[index]));
            continue;
        }
        if (argument == "--network") {
            policy.allow_network = true;
            continue;
        }
        if (argument == "--no-network") {
            policy.allow_network = false;
            continue;
        }
        std::cerr << "[landrun] unknown helper option: " << argument << '\n';
        return kLandrunHelperFailure;
    }
    if (index >= argc || !argv[index] || argv[index][0] != '/') {
        std::cerr << "[landrun] missing absolute target executable\n";
        return kLandrunHelperFailure;
    }
    if (!mode_provided) {
        std::cerr << "[landrun] missing --mode\n";
        return kLandrunHelperFailure;
    }

    auto driver = make_landrun_driver();
    const auto probe = driver->probe();
    if (!probe.available) {
        std::cerr << "[landrun] " << probe.backend << " unavailable: "
                  << probe.detail << '\n';
        return kLandrunHelperFailure;
    }

    std::vector<char*> target_argv;
    target_argv.reserve(static_cast<std::size_t>(argc - index) + 1);
    for (int i = index; i < argc; ++i) target_argv.push_back(argv[i]);
    target_argv.push_back(nullptr);

    if (const auto result = driver->apply(policy); !result.success) {
        std::cerr << "[landrun] " << driver->name() << ": " << result.detail << '\n';
        return kLandrunHelperFailure;
    }

    const std::string driver_name(driver->name());
    if (::setenv("FILO_LANDRUN", driver_name.c_str(), 1) != 0
        || ::setenv("HISTFILE", "/dev/null", 1) != 0
        || ::setenv("HISTSIZE", "0", 1) != 0) {
        std::cerr << "[landrun] cannot finalize the child environment: "
                  << std::strerror(errno) << '\n';
        return kLandrunHelperFailure;
    }
    ::execv(argv[index], target_argv.data());
    std::cerr << "[landrun] exec failed: " << std::strerror(errno) << '\n';
    return 127;
}

} // namespace core::landrun
