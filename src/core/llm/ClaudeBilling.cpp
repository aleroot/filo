#include "ClaudeBilling.hpp"

#include "../utils/Crypto.hpp"

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstdlib>
#include <format>
#include <vector>

namespace core::llm::claude {

namespace {

constexpr std::string_view kSuffixSalt = "59cf53e54c78";
constexpr std::array<std::size_t, 3> kSuffixPositions{4, 7, 20};
constexpr std::uint64_t kBodySignatureSeed = 0x4d659218e32a3268ULL;
constexpr std::string_view kUnsignedToken = "cch=00000;";
/// Claude Code's top-level body order; members it does not know follow.
constexpr std::array<std::string_view, 14> kBodyFieldOrder{
    "model", "messages", "system", "tools", "tool_choice", "metadata", "max_tokens",
    "temperature", "thinking", "context_management", "output_config", "diagnostics",
    "stream", "speed"};

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

/// UTF-16 code units of @p text, up to @p limit of them. Malformed bytes
/// decode to U+FFFD, as a JavaScript TextDecoder would.
[[nodiscard]] std::vector<char16_t> utf16_prefix(std::string_view text, std::size_t limit) {
    std::vector<char16_t> units;
    std::size_t i = 0;
    while (i < text.size() && units.size() < limit) {
        const auto lead = static_cast<unsigned char>(text[i]);
        std::size_t length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 0;
        char32_t cp = length == 1 ? lead : length == 2 ? (lead & 0x1F) : length == 3 ? (lead & 0x0F) : (lead & 0x07);
        bool valid = length != 0 && i + length <= text.size();
        for (std::size_t k = 1; valid && k < length; ++k) {
            const auto next = static_cast<unsigned char>(text[i + k]);
            if ((next & 0xC0) != 0x80) valid = false;
            else cp = (cp << 6) | (next & 0x3F);
        }
        if (!valid) {
            units.push_back(u'\uFFFD');
            ++i;
            continue;
        }
        i += length;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            units.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
            units.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
        } else {
            units.push_back(static_cast<char16_t>(cp));
        }
    }
    return units;
}

[[nodiscard]] std::uint64_t read64(const unsigned char* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

[[nodiscard]] std::uint32_t read32(const unsigned char* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8)
        | (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

constexpr std::uint64_t kP1 = 11400714785074694791ULL;
constexpr std::uint64_t kP2 = 14029467366897019727ULL;
constexpr std::uint64_t kP3 = 1609587929392839161ULL;
constexpr std::uint64_t kP4 = 9650029242287828579ULL;
constexpr std::uint64_t kP5 = 2870177450012600261ULL;

[[nodiscard]] constexpr std::uint64_t xxh_round(std::uint64_t acc, std::uint64_t input) noexcept {
    return std::rotl(acc + input * kP2, 31) * kP1;
}

[[nodiscard]] constexpr std::uint64_t xxh_merge(std::uint64_t acc, std::uint64_t value) noexcept {
    return (acc ^ xxh_round(0, value)) * kP1 + kP4;
}

/// JavaScript Number::toString for a finite double.
void append_js_number(std::string& out, double value) {
    if (value == 0) {
        out += '0';
        return;
    }
    if (value < 0) {
        out += '-';
        value = -value;
    }
    std::array<char, 64> buffer{};
    const auto [end, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                         std::chars_format::scientific);
    const std::string_view repr(buffer.data(), static_cast<std::size_t>(end - buffer.data()));
    const auto e_pos = repr.find('e');
    std::string digits;
    for (const char c : repr.substr(0, e_pos)) if (c != '.') digits += c;
    int exponent = 0;
    std::from_chars(repr.data() + e_pos + (repr[e_pos + 1] == '+' ? 2 : 1), repr.data() + repr.size(), exponent);
    const int k = static_cast<int>(digits.size());
    const int n = exponent + 1;
    if (k <= n && n <= 21) {
        out += digits;
        out.append(static_cast<std::size_t>(n - k), '0');
    } else if (0 < n && n <= 21) {
        out.append(digits, 0, static_cast<std::size_t>(n));
        out += '.';
        out.append(digits, static_cast<std::size_t>(n));
    } else if (-6 < n && n <= 0) {
        out += "0.";
        out.append(static_cast<std::size_t>(-n), '0');
        out += digits;
    } else {
        out += digits[0];
        if (k > 1) {
            out += '.';
            out.append(digits, 1);
        }
        out += 'e';
        out += n - 1 >= 0 ? '+' : '-';
        out += std::to_string(std::abs(n - 1));
    }
}

/// JSON.stringify string quoting: only `"`, `\` and control characters are escaped.
void append_js_string(std::string& out, std::string_view value) {
    out += '"';
    for (const char ch : value) {
        switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(ch)));
            } else {
                out += ch;
            }
        }
    }
    out += '"';
}

void append_js_json(std::string& out, simdjson::dom::element element) {
    switch (element.type()) {
    case simdjson::dom::element_type::OBJECT: {
        out += '{';
        bool first = true;
        for (const auto field : simdjson::dom::object(element)) {
            if (!first) out += ',';
            first = false;
            append_js_string(out, field.key);
            out += ':';
            append_js_json(out, field.value);
        }
        out += '}';
        break;
    }
    case simdjson::dom::element_type::ARRAY: {
        out += '[';
        bool first = true;
        for (const auto item : simdjson::dom::array(element)) {
            if (!first) out += ',';
            first = false;
            append_js_json(out, item);
        }
        out += ']';
        break;
    }
    case simdjson::dom::element_type::STRING:
        append_js_string(out, std::string_view(element));
        break;
    case simdjson::dom::element_type::INT64:
        append_js_number(out, static_cast<double>(std::int64_t(element)));
        break;
    case simdjson::dom::element_type::UINT64:
        append_js_number(out, static_cast<double>(std::uint64_t(element)));
        break;
    case simdjson::dom::element_type::DOUBLE:
        append_js_number(out, double(element));
        break;
    case simdjson::dom::element_type::BOOL:
        out += bool(element) ? "true" : "false";
        break;
    case simdjson::dom::element_type::NULL_VALUE:
    default:
        out += "null";
        break;
    }
}

/// Offset of the billing block's unsigned `cch=00000;`, or npos.
[[nodiscard]] std::size_t unsigned_token_offset(std::string_view body) {
    const auto block = body.find(std::string(kBillingMarker) + " cc_version=");
    if (block == std::string_view::npos) return std::string_view::npos;
    const auto token = body.find(kUnsignedToken, block);
    if (token == std::string_view::npos) return std::string_view::npos;
    // The token must sit inside the same JSON string as the marker.
    if (body.substr(block, token - block).find('"') != std::string_view::npos) return std::string_view::npos;
    return token;
}

} // namespace

std::string_view first_user_text(const ChatRequest& request) {
    for (const auto& message : request.messages) {
        // Filo's `tool` messages are serialized as user tool_result turns.
        if (message.role == "tool") return {};
        if (message.role != "user") continue;
        if (message.content_parts.empty()) return message.content;
        std::string_view last_visible;
        for (const auto& part : message.content_parts) {
            if (part.type != ContentPartType::Text) continue;
            std::string_view text = part.text;
            if (const auto command = text.find("<command-name>"); command != std::string_view::npos) {
                return text.substr(command);
            }
            if (!text.empty()) last_visible = text;
        }
        return last_visible;
    }
    return {};
}

std::string version_suffix(std::string_view first_user_text, std::string_view version) {
    const auto units = utf16_prefix(first_user_text, kSuffixPositions.back() + 1);
    std::string material(kSuffixSalt);
    for (const auto position : kSuffixPositions) {
        if (position >= units.size()) {
            material += '0';
            continue;
        }
        const char16_t unit = units[position];
        // A lone surrogate is encoded as U+FFFD when the string becomes UTF-8.
        append_utf8(material, unit >= 0xD800 && unit <= 0xDFFF ? U'\uFFFD' : static_cast<char32_t>(unit));
    }
    material += version;
    return core::utils::crypto::sha256_hex(material).substr(0, 3);
}

bool is_valid_request_id(std::string_view id) noexcept {
    if (!id.starts_with("req_")) return false;
    const auto rest = id.substr(4);
    return rest.size() >= 8 && rest.size() <= 128
        && std::ranges::all_of(rest, [](char c) {
               return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
                   || c == '_' || c == '-';
           });
}

int count_user_prompts(const ChatRequest& request) noexcept {
    return static_cast<int>(std::ranges::count_if(
        request.messages, [](const Message& message) { return message.role == "user"; }));
}

std::string billing_block(std::string_view first_user_text, const BillingLineage& lineage) {
    auto block = std::format("{} cc_version={}.{}; cc_entrypoint={}; {}", kBillingMarker, kClaudeCodeVersion,
                             version_suffix(first_user_text), kBillingEntrypoint, kUnsignedToken);
    if (is_valid_request_id(lineage.previous_request_id)) {
        block += std::format(" cc_prev_req={};", lineage.previous_request_id);
    }
    if (!lineage.prompt_id.empty()) {
        block += std::format(" cc_prompt_id={};", lineage.prompt_id);
        if (lineage.prompt_index > 0) {
            block += std::format(" cc_turn_origin=human; cc_prompt_index={0}; cc_turn_index={0};",
                                 lineage.prompt_index);
        }
    }
    return block;
}

std::uint64_t xxh64(std::string_view data, std::uint64_t seed) noexcept {
    const auto* p = reinterpret_cast<const unsigned char*>(data.data());
    const auto* const end = p + data.size();
    std::uint64_t h = 0;
    if (data.size() >= 32) {
        std::uint64_t v1 = seed + kP1 + kP2;
        std::uint64_t v2 = seed + kP2;
        std::uint64_t v3 = seed;
        std::uint64_t v4 = seed - kP1;
        for (; p + 32 <= end; p += 32) {
            v1 = xxh_round(v1, read64(p));
            v2 = xxh_round(v2, read64(p + 8));
            v3 = xxh_round(v3, read64(p + 16));
            v4 = xxh_round(v4, read64(p + 24));
        }
        h = std::rotl(v1, 1) + std::rotl(v2, 7) + std::rotl(v3, 12) + std::rotl(v4, 18);
        h = xxh_merge(h, v1);
        h = xxh_merge(h, v2);
        h = xxh_merge(h, v3);
        h = xxh_merge(h, v4);
    } else {
        h = seed + kP5;
    }
    h += data.size();
    for (; p + 8 <= end; p += 8) h = std::rotl(h ^ xxh_round(0, read64(p)), 27) * kP1 + kP4;
    if (p + 4 <= end) {
        h = std::rotl(h ^ (static_cast<std::uint64_t>(read32(p)) * kP1), 23) * kP2 + kP3;
        p += 4;
    }
    for (; p < end; ++p) h = std::rotl(h ^ (*p * kP5), 11) * kP1;
    h ^= h >> 33;
    h *= kP2;
    h ^= h >> 29;
    h *= kP3;
    h ^= h >> 32;
    return h;
}

std::optional<std::string> body_signature(std::string_view unsigned_body) {
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(unsigned_body);
    simdjson::dom::object root;
    if (parser.parse(padded).get_object().get(root) != simdjson::SUCCESS) return std::nullopt;

    std::string canonical;
    canonical.reserve(unsigned_body.size());
    canonical += '{';
    bool first = true;
    for (const auto field : root) {
        if (field.key == "max_tokens") continue;
        if (!first) canonical += ',';
        first = false;
        append_js_string(canonical, field.key);
        canonical += ':';
        if (field.key == "model") canonical += "\"\"";
        else append_js_json(canonical, field.value);
    }
    canonical += '}';
    return std::format("{:05x}", xxh64(canonical, kBodySignatureSeed) & 0xFFFFFULL);
}

void sign_body(std::string& body) {
    if (unsigned_token_offset(body) == std::string::npos) return;
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(body);
    simdjson::dom::object root;
    if (parser.parse(padded).get_object().get(root) != simdjson::SUCCESS) return;

    // The wire body, plus the spans the signature leaves out: the model value
    // (signed as "") and the whole max_tokens member with its separator.
    std::string wire;
    wire.reserve(body.size() + 64);
    std::size_t model_begin = std::string::npos, model_end = 0;
    std::size_t max_tokens_begin = std::string::npos, max_tokens_end = 0;
    wire += '{';
    const auto emit = [&](std::string_view key, simdjson::dom::element value) {
        const bool first = wire.size() == 1;
        const std::size_t member_begin = wire.size();
        if (!first) wire += ',';
        append_js_string(wire, key);
        wire += ':';
        const std::size_t value_begin = wire.size();
        append_js_json(wire, value);
        if (key == "model") {
            model_begin = value_begin;
            model_end = wire.size();
        } else if (key == "max_tokens") {
            max_tokens_begin = member_begin;
            // A leading member drops the separator that follows it instead.
            max_tokens_end = wire.size() + (first ? 1 : 0);
        }
    };
    for (const auto key : kBodyFieldOrder) {
        simdjson::dom::element value;
        if (root[key].get(value) == simdjson::SUCCESS) emit(key, value);
    }
    for (const auto field : root) {
        if (std::ranges::find(kBodyFieldOrder, field.key) == kBodyFieldOrder.end()) emit(field.key, field.value);
    }
    wire += '}';

    std::string canonical = wire;
    const auto drop_max_tokens = [&] {
        if (max_tokens_begin != std::string::npos) {
            canonical.erase(max_tokens_begin, std::min(max_tokens_end, canonical.size() - 1) - max_tokens_begin);
        }
    };
    const auto blank_model = [&] {
        if (model_begin != std::string::npos) canonical.replace(model_begin, model_end - model_begin, "\"\"");
    };
    // Edit the later span first so the earlier offsets stay valid.
    if (max_tokens_begin != std::string::npos && max_tokens_begin > model_begin) {
        drop_max_tokens();
        blank_model();
    } else {
        blank_model();
        drop_max_tokens();
    }

    const auto token = unsigned_token_offset(wire);
    if (token == std::string::npos) return;
    wire.replace(token + 4, 5, std::format("{:05x}", xxh64(canonical, kBodySignatureSeed) & 0xFFFFFULL));
    body = std::move(wire);
}

} // namespace core::llm::claude
