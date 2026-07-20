#include "LandrunStatus.hpp"

#include "LandrunSettings.hpp"

namespace core::landrun {

std::string landrun_status_label() {
    if (!LandrunSettings::instance().enabled()) return "sandbox: off";
#if defined(__APPLE__)
    return "sandbox: macOS Seatbelt · host read · workspace/temp write · no child network";
#elif defined(__linux__)
    return "sandbox: Linux Landlock+seccomp (partial) · policy-root read · workspace/temp write · no child network";
#else
    return "sandbox: unavailable";
#endif
}

std::string landrun_status_detail() {
    if (!LandrunSettings::instance().enabled()) {
        return "Sandbox is off; agent-controlled commands are unrestricted.";
    }
#if defined(__APPLE__)
    return "Native Seatbelt is active. Host files are broadly readable for tool compatibility; "
           "known sensitive paths are denied, writes are limited to granted roots, and child "
           "networking is denied. This is write/network confinement, not complete confidentiality.";
#elif defined(__linux__)
    return "Native Landlock and seccomp are active. Reads are limited to policy roots, writes are "
           "limited to granted roots, inherited descriptors and child networking are denied. "
           "Workspace contents and repository metadata remain readable and writable; Landlock "
           "metadata-operation gaps make this a partial rather than strong boundary.";
#else
    return "No landrun backend is available on this platform.";
#endif
}

} // namespace core::landrun
