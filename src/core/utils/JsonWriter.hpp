#pragma once

// ---------------------------------------------------------------------------
// JsonWriter — header-only streaming JSON builder for C++26.
//
// Design goals
// ─────────────
//  • Zero per-call heap allocation: the builder owns a single pre-reserved
//    std::string; every write is an append.
//  • RAII scope guards: opening '{' / '[' on construction, closing '}' / ']'
//    on destruction — no manual balancing needed.
//  • Type safety via C++20 concepts (std::integral constraint on number()).
//  • std::to_chars for integer serialisation: no locale, no allocation, and
//    constant-time for any integral type.
//  • C++26 placeholder variable (_) in the to_chars structured binding to
//    discard the unused error code (P2169).
//  • = delete("reason") on Scope copy to produce helpful diagnostics (P2573).
//  • std::inplace_vector used by callers for zero-heap required-field lists.
// ---------------------------------------------------------------------------

#include "JsonUtils.hpp"    // append_escaped, kEscapeTable
#include <charconv>         // std::to_chars
#include <concepts>         // std::integral
#include <string>
#include <string_view>
#include <utility>          // std::exchange

namespace core::utils {

class JsonWriter {
public:
    // Pre-allocate the internal buffer to avoid small-string reallocations.
    explicit JsonWriter(std::size_t reserve = 512) {
        buf_.reserve(reserve);
    }

    // -----------------------------------------------------------------------
    // Primitive value emitters
    // -----------------------------------------------------------------------

    JsonWriter& raw(std::string_view v)  { buf_ += v;                  return *this; }
    JsonWriter& comma()                  { buf_ += ',';                 return *this; }
    JsonWriter& null_val()               { buf_ += "null";              return *this; }
    JsonWriter& boolean(bool v)          { buf_ += v ? "true" : "false"; return *this; }

    // Integer serialisation via std::to_chars (no locale, no allocation).
    // The C++26 placeholder variable _ discards the unused error-code field
    // of the to_chars_result structured binding (P2169 — placeholder with
    // no name, fully supported in GCC 15 / -std=c++26).
    JsonWriter& number(std::integral auto v) {
        char tmp[32];
        auto [ptr, _] = std::to_chars(tmp, tmp + sizeof(tmp), v);
        buf_.append(tmp, ptr);
        return *this;
    }

    JsonWriter& number(double v) {
        char tmp[64];
        int n = std::snprintf(tmp, sizeof(tmp), "%.6g", v);
        if (n > 0 && n < static_cast<int>(sizeof(tmp))) {
            buf_.append(tmp, n);
        }
        return *this;
    }

    // Floating-point with specified precision.
    JsonWriter& number(double v, int precision) {
        char tmp[64];
        int n = std::snprintf(tmp, sizeof(tmp), "%.*g", precision, v);
        if (n > 0 && n < static_cast<int>(sizeof(tmp))) {
            buf_.append(tmp, n);
        }
        return *this;
    }

    // Escaped string value — wraps the content in quotes and calls
    // append_escaped, which auto-vectorises with -O3 -march=native.
    JsonWriter& str(std::string_view v) {
        buf_ += '"';
        append_escaped(buf_, v);
        buf_ += '"';
        return *this;
    }

    // Object key: writes "key": (colon included, no trailing comma).
    JsonWriter& key(std::string_view k) {
        buf_ += '"';
        append_escaped(buf_, k);
        buf_ += "\":";
        return *this;
    }

    // -----------------------------------------------------------------------
    // Key-value shortcuts — write "key": <value> in one call.
    // -----------------------------------------------------------------------

    JsonWriter& kv_str(std::string_view k, std::string_view v)     { return key(k).str(v);     }
    JsonWriter& kv_raw(std::string_view k, std::string_view v)     { return key(k).raw(v);     }
    JsonWriter& kv_bool(std::string_view k, bool v)                { return key(k).boolean(v); }
    JsonWriter& kv_num(std::string_view k, std::integral auto v)   { return key(k).number(v);  }
    JsonWriter& kv_null(std::string_view k)                        { return key(k).null_val(); }

    // Floating-point key-value helpers (optimized, no locale overhead).
    JsonWriter& kv_float(std::string_view k, double v) {
        key(k);
        char tmp[64];
        int n = std::snprintf(tmp, sizeof(tmp), "%.6g", v);
        if (n > 0 && n < static_cast<int>(sizeof(tmp))) {
            buf_.append(tmp, n);
        }
        return *this;
    }
    JsonWriter& kv_float(std::string_view k, double v, int precision) {
        key(k);
        char tmp[64];
        int n = std::snprintf(tmp, sizeof(tmp), "%.*g", precision, v);
        if (n > 0 && n < static_cast<int>(sizeof(tmp))) {
            buf_.append(tmp, n);
        }
        return *this;
    }

    // -----------------------------------------------------------------------
    // RAII scope guard
    //
    // Appends the opening character on construction; appends the closing
    // character on destruction.  Holds a raw pointer to buf_ — do not move
    // the owning JsonWriter while a Scope is alive.
    //
    // Typical usage:
    //     {
    //         auto _obj = writer.object();   // writes '{'
    //         writer.kv_str("x", "y");
    //     }                                   // writes '}'
    //
    // In C++26 the placeholder variable _ can be reused across nested scopes
    // (P2169) and the correct destruction order (innermost first) is
    // guaranteed by the language.
    // -----------------------------------------------------------------------
    struct [[nodiscard]] Scope {
        std::string* buf_;
        char         close_;

        Scope(std::string* b, char c) noexcept : buf_(b), close_(c) {}
        Scope(const Scope&)            = delete("JsonWriter::Scope is not copyable");
        Scope& operator=(const Scope&) = delete("JsonWriter::Scope is not copyable");

        // Move constructor: transfer ownership and null the source so its
        // destructor becomes a no-op.
        Scope(Scope&& o) noexcept
            : buf_(std::exchange(o.buf_, nullptr)), close_(o.close_) {}

        ~Scope() { if (buf_) *buf_ += close_; }
    };

    [[nodiscard]] Scope object() { buf_ += '{'; return {&buf_, '}'}; }
    [[nodiscard]] Scope array()  { buf_ += '['; return {&buf_, ']'}; }

    // -----------------------------------------------------------------------
    // Extraction
    // -----------------------------------------------------------------------

    // Move the built string out (consumes the writer).
    [[nodiscard]] std::string   take()             && { return std::move(buf_); }
    // Non-owning view — valid until the next modification.
    [[nodiscard]] std::string_view view()    const noexcept { return buf_; }

private:
    std::string buf_;
};

} // namespace core::utils
