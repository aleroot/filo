#include "LandrunLaunch.hpp"

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace core::landrun {
namespace {

[[nodiscard]] std::string current_executable() {
#if defined(__APPLE__)
    std::uint32_t size = 0;
    (void)::_NSGetExecutablePath(nullptr, &size);
    std::string path(size, '\0');
    if (::_NSGetExecutablePath(path.data(), &size) != 0) return {};
    path.resize(std::char_traits<char>::length(path.c_str()));
    std::error_code ec;
    auto canonical = std::filesystem::weakly_canonical(path, ec);
    return ec ? path : canonical.string();
#elif defined(__linux__)
    std::array<char, 4096> path{};
    const auto count = ::readlink("/proc/self/exe", path.data(), path.size() - 1);
    return count < 0 ? std::string{} : std::string(path.data(), static_cast<std::size_t>(count));
#else
    return {};
#endif
}

} // namespace

LandrunLaunch prepare_landrun_launch(
    const LandrunPolicy& policy,
    const std::filesystem::path& target,
    std::vector<std::string> target_arguments)
{
    if (!target.is_absolute() || target_arguments.empty()) {
        throw std::invalid_argument(
            "landrun target must be absolute and provide argv[0]");
    }
    if (!policy.enabled()) {
        return {.executable = target.string(),
                .arguments = std::move(target_arguments)};
    }

    auto executable = current_executable();
    if (executable.empty()) {
        throw std::runtime_error("cannot resolve the Filo executable for landrun");
    }
    std::vector<std::string> arguments;
    arguments.reserve(
        5 + (policy.readable_roots.size()
             + policy.writable_roots.size()
             + policy.protected_read_paths.size()
             + policy.protected_write_paths.size()) * 2
        + target_arguments.size());
    arguments.push_back(executable);
    arguments.emplace_back("__landrun-exec");
    for (const auto& root : policy.readable_roots) {
        arguments.emplace_back("--ro");
        arguments.push_back(root.string());
    }
    for (const auto& root : policy.writable_roots) {
        arguments.emplace_back("--rw");
        arguments.push_back(root.string());
    }
    for (const auto& path : policy.protected_read_paths) {
        arguments.emplace_back("--deny-read");
        arguments.push_back(path.string());
    }
    for (const auto& path : policy.protected_write_paths) {
        arguments.emplace_back("--deny-write");
        arguments.push_back(path.string());
    }
    arguments.emplace_back(policy.allow_network ? "--network" : "--no-network");
    arguments.emplace_back("--");
    arguments.push_back(target.string());
    arguments.insert(
        arguments.end(), target_arguments.begin() + 1, target_arguments.end());
    return {.executable = std::move(executable), .arguments = std::move(arguments)};
}

LandrunLaunch prepare_shell_launch(const LandrunPolicy& policy) {
    const bool has_bash = ::access("/bin/bash", X_OK) == 0;
    const std::filesystem::path shell = has_bash ? "/bin/bash" : "/bin/sh";
    std::vector<std::string> arguments;
    arguments.push_back(has_bash ? "bash" : "sh");
    if (has_bash) {
        arguments.emplace_back("--norc");
        arguments.emplace_back("--noprofile");
    }
    return prepare_landrun_launch(policy, shell, std::move(arguments));
}

} // namespace core::landrun
