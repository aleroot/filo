#pragma once

#include <string>
#include <string_view>
#include <unistd.h>

namespace core::tools::detail {

// Escapes a string for safe use inside a single-quoted POSIX shell argument.
// Example: foo'bar → foo'\''bar, then wrapped in outer single quotes by the caller.
inline std::string shell_single_quote(std::string_view sv) {
    std::string out;
    out.reserve(sv.size() + 4);
    for (char c : sv) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    return out;
}

// Probe known install paths using access() — no shell spawn needed.
inline bool probe_binary(std::initializer_list<const char*> paths) {
    for (const char* p : paths)
        if (::access(p, X_OK) == 0) return true;
    return false;
}

} // namespace core::tools::detail
