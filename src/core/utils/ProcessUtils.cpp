#include "ProcessUtils.hpp"

#include <cerrno>
#include <signal.h>

#if defined(__APPLE__)
#include <libproc.h>
#include <sys/proc.h>
#include <sys/proc_info.h>
#include <vector>
#else
#include <array>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#endif

namespace core::utils::process {
namespace {

#if defined(__APPLE__)
bool has_running_member(int group) noexcept {
    const int estimate = ::proc_listpids(PROC_PGRP_ONLY, static_cast<uint32_t>(group), nullptr, 0);
    if (estimate <= 0) return true;
    std::vector<pid_t> pids(static_cast<std::size_t>(estimate) / sizeof(pid_t) + 16);
    const int capacity = static_cast<int>(pids.size() * sizeof(pid_t));
    const int bytes = ::proc_listpids(PROC_PGRP_ONLY, static_cast<uint32_t>(group), pids.data(), capacity);
    if (bytes <= 0 || bytes >= capacity) return true;  // Unknown, or more members than listed.
    for (std::size_t i = 0; i < static_cast<std::size_t>(bytes) / sizeof(pid_t); ++i) {
        proc_bsdshortinfo info{};
        errno = 0;
        if (::proc_pidinfo(pids[i], PROC_PIDT_SHORTBSDINFO, 0, &info, sizeof info) == sizeof info) {
            if (info.pbsi_status != SZOMB) return true;
        } else if (errno != ESRCH) {
            return true;  // A zombie answers ESRCH; anything else is unknown.
        }
    }
    return false;
}
#else
/// The state and process group `/proc/<pid>/stat` records, or `false` when it
/// cannot be read: the process exited after it was listed.
bool read_stat(const char* pid, char& state, int& group) noexcept {
    std::array<char, 64> path{};
    std::snprintf(path.data(), path.size(), "/proc/%s/stat", pid);
    std::FILE* file = std::fopen(path.data(), "re");
    if (file == nullptr) return false;
    // "pid (comm) state ppid pgrp ...": comm is at most 64 bytes and holds no
    // NUL, but may hold ')', so the fields start after the last ')'.
    std::array<char, 512> line{};
    const bool read = std::fread(line.data(), 1, line.size() - 1, file) > 0;
    std::fclose(file);
    const char* close = read ? std::strrchr(line.data(), ')') : nullptr;
    return close != nullptr && std::sscanf(close + 1, " %c %*d %d", &state, &group) == 2;
}

bool has_running_member(int group) noexcept {
    DIR* proc = ::opendir("/proc");
    if (proc == nullptr) return true;
    bool listed = false;
    bool running = false;
    while (const dirent* entry = ::readdir(proc)) {
        if (entry->d_name[0] < '1' || entry->d_name[0] > '9') continue;
        char state = 0;
        int member_group = 0;
        if (!read_stat(entry->d_name, state, member_group) || member_group != group) continue;
        listed = true;
        if (state != 'Z' && state != 'X') {
            running = true;
            break;
        }
    }
    ::closedir(proc);
    // A group that answers a signal but lists no member is hidden from this
    // process (hidepid), not gone.
    return running || !listed;
}
#endif

} // namespace

bool group_may_exist(int group) noexcept {
    if (group <= 1) return true;
    if (::killpg(group, 0) != 0 && errno == ESRCH) return false;
    // A signal reaches zombies too, so only a running member proves the group live.
    return has_running_member(group);
}

} // namespace core::utils::process
