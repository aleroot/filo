#include "LandrunStatus.hpp"

#include "LandrunSettings.hpp"

namespace core::landrun {

namespace {

[[nodiscard]] std::string_view workspace_access_label(LandrunMode mode) {
    return landrun_workspace_writable(mode)
        ? "workspace/temp write"
        : "workspace read-only · private temp write";
}

} // namespace

std::string landrun_status_label(LandrunMode mode) {
    if (!landrun_enabled(mode)) return "sandbox: off";
#if defined(__APPLE__)
    return "sandbox: macOS Seatbelt · host read · "
        + std::string(workspace_access_label(mode))
        + " · no child network";
#elif defined(__linux__)
    return "sandbox: Linux Landlock+seccomp (partial) · policy-root read · "
        + std::string(workspace_access_label(mode))
        + " · no child network";
#else
    return "sandbox: unavailable";
#endif
}

std::string landrun_status_detail(LandrunMode mode) {
    if (!landrun_enabled(mode)) {
        return "Sandbox is off; agent-controlled commands are unrestricted.";
    }
    const std::string workspace_detail = landrun_workspace_writable(mode)
        ? "Workspace writes are allowed"
        : "Workspace writes are denied for native file tools and child processes";
    const std::string temp_detail = landrun_workspace_writable(mode)
        ? "private and shared temp roots remain writable"
        : "a private temp root remains writable";
#if defined(__APPLE__)
    return "Native Seatbelt is active. Host files are broadly readable for tool compatibility; "
        "known sensitive paths are denied. " + workspace_detail
        + "; " + temp_detail + ", and child networking is denied. "
          "This is write/network confinement, not complete confidentiality.";
#elif defined(__linux__)
    return "Native Landlock and seccomp are active. Reads are limited to policy roots. "
        + workspace_detail
        + "; " + temp_detail + ", inherited descriptors and child "
          "networking are denied. Landlock metadata-operation gaps make this a partial rather "
          "than strong boundary.";
#else
    return "No landrun backend is available on this platform.";
#endif
}

std::string landrun_status_label() {
    return landrun_status_label(LandrunSettings::instance().mode());
}

std::string landrun_status_detail() {
    return landrun_status_detail(LandrunSettings::instance().mode());
}

} // namespace core::landrun
