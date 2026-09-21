#include "ZaiModelTraits.hpp"

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <optional>

namespace core::llm {
namespace {

enum class ZaiReasoningEffort {
    ProviderDefault,
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Ultra,
};

[[nodiscard]] ZaiReasoningEffort parse_reasoning_effort(
    std::string_view raw_effort) noexcept {
    const std::string_view effort = core::utils::str::trim_ascii_view(raw_effort);
    using core::utils::ascii::iequals;
    if (effort.empty() || iequals(effort, "auto") || iequals(effort, "unset")
        || iequals(effort, "default")) {
        return ZaiReasoningEffort::ProviderDefault;
    }
    if (iequals(effort, "off") || iequals(effort, "none")
        || iequals(effort, "disabled") || iequals(effort, "disable")) {
        return ZaiReasoningEffort::Off;
    }
    if (iequals(effort, "minimal")) return ZaiReasoningEffort::Minimal;
    if (iequals(effort, "low")) return ZaiReasoningEffort::Low;
    if (iequals(effort, "medium")) return ZaiReasoningEffort::Medium;
    if (iequals(effort, "high")) return ZaiReasoningEffort::High;
    if (iequals(effort, "xhigh")) return ZaiReasoningEffort::XHigh;
    if (iequals(effort, "max")) return ZaiReasoningEffort::Max;
    if (iequals(effort, "ultra")) return ZaiReasoningEffort::Ultra;
    return ZaiReasoningEffort::ProviderDefault;
}

[[nodiscard]] ZaiReasoningEffort effort_for_model(
    ZaiReasoningEffort effort,
    std::string_view model) noexcept {
    if (is_glm_53_model(model)) {
        switch (effort) {
        case ZaiReasoningEffort::Minimal:
        case ZaiReasoningEffort::Off:
        case ZaiReasoningEffort::Low:
            return ZaiReasoningEffort::Low;
        case ZaiReasoningEffort::Medium:
        case ZaiReasoningEffort::XHigh:
        case ZaiReasoningEffort::High:
            return ZaiReasoningEffort::High;
        case ZaiReasoningEffort::Ultra:
        case ZaiReasoningEffort::Max:
            return ZaiReasoningEffort::Max;
        case ZaiReasoningEffort::ProviderDefault:
            return effort;
        }
    }
    if (is_glm_52_model(model)) {
        switch (effort) {
        case ZaiReasoningEffort::Low:
        case ZaiReasoningEffort::Medium:
            return ZaiReasoningEffort::High;
        case ZaiReasoningEffort::XHigh:
        case ZaiReasoningEffort::Ultra:
            return ZaiReasoningEffort::Max;
        default:
            return effort;
        }
    }
    return effort;
}

[[nodiscard]] std::string_view reasoning_effort_name(
    ZaiReasoningEffort effort) noexcept {
    switch (effort) {
    case ZaiReasoningEffort::Minimal: return "minimal";
    case ZaiReasoningEffort::Low: return "low";
    case ZaiReasoningEffort::Medium: return "medium";
    case ZaiReasoningEffort::High: return "high";
    case ZaiReasoningEffort::XHigh: return "xhigh";
    case ZaiReasoningEffort::Max: return "max";
    case ZaiReasoningEffort::Ultra: return "ultra";
    case ZaiReasoningEffort::ProviderDefault:
    case ZaiReasoningEffort::Off:
        return {};
    }
    return {};
}

void append_thinking_fields(
    std::string& payload,
    std::string_view model,
    std::string_view effort,
    bool openai_wire) {
    if (!is_glm_model(model)) return;
    const ReasoningCapabilities capabilities = glm_reasoning_capabilities(model);
    ZaiReasoningEffort parsed = parse_reasoning_effort(effort);

    const bool skip_glm_52_thinking = is_glm_52_model(model)
        && parsed == ZaiReasoningEffort::Minimal;
    if (parsed == ZaiReasoningEffort::Off || skip_glm_52_thinking) {
        if (capabilities.supports(ReasoningCapability::Disable)) {
            payload += R"(,"thinking":{"type":"disabled"})";
            return;
        }
        if (capabilities.supports(ReasoningCapability::Required)
            && capabilities.supports_effort()) {
            parsed = ZaiReasoningEffort::Low;
        } else {
            parsed = ZaiReasoningEffort::ProviderDefault;
        }
    }

    if (openai_wire) {
        payload += R"(,"thinking":{"type":"enabled","clear_thinking":false})";
    } else {
        payload += R"(,"thinking":{"type":"enabled"})";
    }

    if (parsed == ZaiReasoningEffort::ProviderDefault
        || !capabilities.supports_effort()) {
        return;
    }
    parsed = effort_for_model(parsed, model);
    const std::string_view effort_name = reasoning_effort_name(parsed);
    if (effort_name.empty()) return;
    if (openai_wire) {
        payload += R"(,"reasoning_effort":")";
        payload += effort_name;
        payload += '"';
        return;
    }
    payload += R"(,"output_config":{"effort":")";
    payload += effort_name;
    payload += "\"}";
}

} // namespace

bool is_glm_model(std::string_view model) noexcept {
    return core::utils::ascii::istarts_with(
        core::utils::str::trim_ascii_view(model), "glm-");
}

bool is_glm_model_family(
    std::string_view model,
    std::string_view family) noexcept {
    const std::string_view normalized = core::utils::str::trim_ascii_view(model);
    if (!core::utils::ascii::istarts_with(normalized, family)) return false;
    if (normalized.size() == family.size()) return true;
    const char suffix = normalized[family.size()];
    return suffix == '-' || suffix == '.' || suffix == ':'
        || suffix == '/' || suffix == '[';
}

bool is_glm_53_model(std::string_view model) noexcept {
    return is_glm_model_family(model, "glm-5.3");
}

bool is_glm_52_model(std::string_view model) noexcept {
    return is_glm_model_family(model, "glm-5.2");
}

bool is_glm_multimodal_model(std::string_view model) noexcept {
    // ZCode capability matrix: video/pdf input is advertised for the Flash
    // variant of GLM-5.3 only ("GLM-5.3-Flash", including dated/latest
    // suffixes). Base GLM-5.3 stays text+image.
    return is_glm_model_family(model, "glm-5.3-flash");
}

ReasoningCapabilities glm_reasoning_capabilities(
    std::string_view model) noexcept {
    if (!is_glm_model(model)) return {};
    if (is_glm_53_model(model)) {
        return ReasoningCapability::Effort
            | ReasoningCapability::MaxEffort
            | ReasoningCapability::Required
            | ReasoningCapability::MapsMediumToHigh
            | ReasoningCapability::MapsMinimalToLow
            | ReasoningCapability::MapsXHighToHigh;
    }
    if (is_glm_52_model(model)) {
        return ReasoningCapability::Effort
            | ReasoningCapability::MaxEffort
            | ReasoningCapability::XHighEffort
            | ReasoningCapability::Disable
            | ReasoningCapability::MapsLowToHigh
            | ReasoningCapability::MapsMediumToHigh
            | ReasoningCapability::MapsXHighToMax
            | ReasoningCapability::MapsMinimalToOff;
    }
    if (is_glm_model_family(model, "glm-5")
        || is_glm_model_family(model, "glm-4.7")
        || is_glm_model_family(model, "glm-4.6")
        || is_glm_model_family(model, "glm-4.5")) {
        return ReasoningCapabilities{ReasoningCapability::Disable};
    }
    return {};
}

void append_glm_openai_thinking_fields(
    std::string& payload,
    std::string_view model,
    std::string_view effort) {
    append_thinking_fields(payload, model, effort, true);
}

void append_glm_anthropic_thinking_fields(
    std::string& payload,
    std::string_view model,
    std::string_view effort) {
    append_thinking_fields(payload, model, effort, false);
}

bool zai_is_transient_business_code(int code) noexcept {
    switch (code) {
    case 1120:
    case 1230:
    case 1234:
    case 1302:
    case 1303:
    case 1305:
    case 1312:
    case 2007:
    case 3002:
        return true;
    default:
        return false;
    }
}

bool zai_is_resettable_limit(int code) noexcept {
    return code == 1302
        || code == 1304
        || code == 1308
        || code == 1310
        || code == 1313
        || (code >= 1316 && code <= 1321);
}

[[nodiscard]] std::optional<int> bracketed_business_code(
    std::string_view text) noexcept {
    // ZCode parity: SSE error chunks may carry only "[1302][…][request_id]".
    // The closing bracket must be followed by another bracket so ordinary
    // markdown references are never mistaken for a business code.
    for (std::size_t at = text.find('['); at != std::string_view::npos;
         at = text.find('[', at + 1)) {
        std::size_t digit_end = at + 1;
        while (digit_end < text.size()
               && text[digit_end] >= '0'
               && text[digit_end] <= '9') {
            ++digit_end;
        }
        const std::size_t digits = digit_end - (at + 1);
        if (digits != 4) continue;
        if (digit_end >= text.size() || text[digit_end] != ']') continue;
        if (digit_end + 1 >= text.size() || text[digit_end + 1] != '[') continue;

        int code = 0;
        for (std::size_t i = at + 1; i < digit_end; ++i) {
            code = code * 10 + (text[i] - '0');
        }
        return code;
    }
    return std::nullopt;
}

int zai_stream_business_code(
    std::string_view error_type,
    std::string_view error_message) noexcept {
    // Some gateways send the bare 4-digit business code as the error type.
    if (error_type.size() == 4) {
        const bool all_digits = std::ranges::all_of(error_type, [](char ch) {
            return ch >= '0' && ch <= '9';
        });
        if (all_digits) {
            int code = 0;
            for (const char ch : error_type) code = code * 10 + (ch - '0');
            return code;
        }
    }

    if (const auto bracketed = bracketed_business_code(error_type)) {
        return *bracketed;
    }
    if (const auto bracketed = bracketed_business_code(error_message)) {
        return *bracketed;
    }
    return 0;
}

bool zai_is_context_overflow_code(int code) noexcept {
    return code == 1261;
}

bool zai_error_mentions_context_overflow(std::string_view message) noexcept {
    static constexpr std::array<std::string_view, 6> kOverflowPhrases{{
        "context window",
        "context_length_exceeded",
        "prompt is too long",
        "maximum context length",
        "input length exceeds",
        "too many tokens",
    }};
    return std::ranges::any_of(
        kOverflowPhrases,
        [&](std::string_view phrase) {
            return core::utils::ascii::icontains(message, phrase);
        });
}

bool zai_error_is_signature_rejection(
    int http_status,
    std::string_view message) noexcept {
    if (http_status != 400) return false;

    // Provider does not give this 400 a dedicated business code; only the
    // confirmed narrow wording may trigger history repair (ZCode parity).
    if (core::utils::ascii::icontains(message, "signature in thinking block")) {
        return true;
    }

    const bool names_thinking_block =
        core::utils::ascii::icontains(message, "thinking block")
        || core::utils::ascii::icontains(message, "`thinking`")
        || core::utils::ascii::icontains(message, "redacted_thinking");
    const bool names_signature_failure =
        core::utils::ascii::icontains(message, "cannot be modified")
        || core::utils::ascii::icontains(message, "invalid signature");
    return names_thinking_block && names_signature_failure;
}

} // namespace core::llm
