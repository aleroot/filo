#include "LandrunRuntime.hpp"
#include "LandrunSettings.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>

namespace core::landrun {

LandrunRuntime::LandrunRuntime() {
    std::error_code ec;
    const std::filesystem::path temp_root{"/tmp"};
    std::string pattern = (temp_root / "filo-XXXXXX").string();
    char* created = ::mkdtemp(pattern.data());
    if (!created) {
        throw std::runtime_error(
            std::string("landrun cannot create private runtime: ") + std::strerror(errno));
    }
    root_ = std::filesystem::path(created).lexically_normal();
    if (::chmod(root_.c_str(), S_IRWXU) != 0) {
        std::filesystem::remove_all(root_, ec);
        throw std::runtime_error(
            std::string("landrun cannot secure private runtime: ") + std::strerror(errno));
    }

    constexpr std::array<const char*, 7> directories{
        "home", "tmp", "cache", "config", "data", "state", "run"};
    for (const char* directory : directories) {
        const auto path = root_ / directory;
        if (!std::filesystem::create_directory(path, ec) || ec) {
            std::filesystem::remove_all(root_, ec);
            throw std::runtime_error("landrun cannot initialize private runtime directories");
        }
        if (::chmod(path.c_str(), S_IRWXU) != 0) {
            std::filesystem::remove_all(root_, ec);
            throw std::runtime_error("landrun cannot secure private runtime directories");
        }
    }
    LandrunSettings::instance().set_runtime_root(root_);
}

LandrunRuntime::~LandrunRuntime() {
    if (root_.empty()) return;
    if (LandrunSettings::instance().runtime_root() == root_) {
        LandrunSettings::instance().set_runtime_root({});
    }
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
}

} // namespace core::landrun
