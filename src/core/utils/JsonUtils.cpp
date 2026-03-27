#include "JsonUtils.hpp"
#include <algorithm>   // std::find_if
#include <cstdio>      // std::snprintf

namespace core::utils {

void append_escaped(std::string& out, std::string_view sv) {
    const char* p         = sv.data();
    const char* const end = p + sv.size();

    while (p != end) {
        const char* q = std::find_if(p, end,[](unsigned char c) noexcept { return kEscapeTable[c] != 0u; });

        // Bulk-append the clean run with a single memcpy-based append.
        out.append(p, q);
        if (q == end) break;

        switch (static_cast<unsigned char>(*q)) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default: {
                // Remaining control characters: emit \u00XX
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x",
                              static_cast<unsigned char>(*q));
                out.append(buf, 6);
            }
        }
        p = q + 1;
    }
}

} // namespace core::utils
