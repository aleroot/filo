#include "AuthBrowserLauncher.hpp"

#include <string>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace core::auth {

void open_browser(std::string_view url) {
#if defined(_WIN32)
    (void)url;
#else
    if (url.empty()) {
        return;
    }

    const std::string url_string(url);
    pid_t pid = fork();
    if (pid < 0) {
        return;
    }
    if (pid == 0) {
        if (fork() == 0) {
#if defined(__linux__)
            execlp("xdg-open", "xdg-open", url_string.c_str(), nullptr);
            execlp("open", "open", url_string.c_str(), nullptr);
#elif defined(__APPLE__)
            execlp("open", "open", url_string.c_str(), nullptr);
            execlp("xdg-open", "xdg-open", url_string.c_str(), nullptr);
#endif
            _exit(1);
        }
        _exit(0);
    }

    waitpid(pid, nullptr, 0);
#endif
}

} // namespace core::auth
