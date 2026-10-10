#include "GeminiAntigravityProtocol.hpp"
#include "AntigravityConversationState.hpp"
#include "../../utils/JsonUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace core::llm::protocols {

namespace {

constexpr int kClaudeMaxOutputTokens = 64000;

// Per-wire-id constants captured from the real `antigravity/hub` client.
// `model_enum` is telemetry-only (the backend does not require it) but the
// real client never omits it for these ids, so neither does Filo.
struct WireProfileEntry {
    std::string_view model;
    AntigravityModelWireProfile profile;
};

constexpr std::array<WireProfileEntry, 7> kWireProfiles{{
    {"gemini-3.5-flash-extra-low", {"MODEL_PLACEHOLDER_M187", 65536}},
    {"gemini-3.5-flash-low", {"MODEL_PLACEHOLDER_M20", 65536}},
    {"gemini-3-flash-agent", {"MODEL_PLACEHOLDER_M132", 65536}},
    {"gemini-3.1-pro-low", {"MODEL_PLACEHOLDER_M36", 65535}},
    {"gemini-pro-agent", {"MODEL_PLACEHOLDER_M16", 65535}},
    // Claude on `daily-cloudcode-pa` rejects maxOutputTokens > 64000 with a
    // 400; these ids carry no model_enum label.
    {"claude-sonnet-4-6", {{}, 64000}},
    {"claude-opus-4-6-thinking", {{}, 64000}},
}};

[[nodiscard]] std::uint64_t fnv1a64(std::string_view text) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::string uuid_from_seed(std::string_view seed) {
    constexpr char kHex[] = "0123456789abcdef";
    const std::uint64_t hi = fnv1a64(std::string(seed) + "#hi");
    const std::uint64_t lo = fnv1a64(std::string(seed) + "#lo");
    std::string hex;
    hex.reserve(32);
    for (int shift = 60; shift >= 0; shift -= 4) hex += kHex[(hi >> shift) & 0xF];
    for (int shift = 60; shift >= 0; shift -= 4) hex += kHex[(lo >> shift) & 0xF];
    hex[12] = '4';                                    // version 4
    hex[16] = kHex[8 + (std::string_view(kHex).find(hex[16]) & 0x3)]; // RFC 4122 variant
    return hex.substr(0, 8) + '-' + hex.substr(8, 4) + '-' + hex.substr(12, 4)
        + '-' + hex.substr(16, 4) + '-' + hex.substr(20, 12);
}

[[nodiscard]] std::string conversation_anchor(const ChatRequest& req) {
    if (!req.session_id.empty()) return req.session_id;
    for (const auto& msg : req.messages) {
        if (msg.role != "user") continue;
        if (!msg.content.empty()) return msg.content;
        for (const auto& part : msg.content_parts) {
            if (part.type == ContentPartType::Text && !part.text.empty()) return part.text;
        }
        break;
    }
    return "filo-antigravity";
}

[[nodiscard]] std::string next_request_timestamp_ms() {
    static std::atomic<std::int64_t> last{0};
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::int64_t previous = last.load();
    std::int64_t next;
    do {
        next = std::max<std::int64_t>(now, previous + 1);
    } while (!last.compare_exchange_weak(previous, next));
    return std::to_string(next);
}

[[nodiscard]] bool is_claude_model(std::string_view model) {
    return core::utils::str::to_lower_ascii_copy(model).starts_with("claude-");
}

[[nodiscard]] bool is_gemini3_or_newer(std::string_view model) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(model);
    if (!lowered.starts_with("gemini-")) return false;
    const auto digit = lowered.find_first_of("0123456789", 7);
    return digit != std::string::npos && lowered[digit] >= '3';
}

std::string trim_trailing_slashes(std::string_view url) {
    while (!url.empty() && url.back() == '/') url.remove_suffix(1);
    return std::string(url);
}

// One `"key":"value"` labels entry with the value JSON-escaped.
[[nodiscard]] std::string labels_entry(std::string_view key, std::string_view value) {
    std::string entry = "\"";
    entry += key;
    entry += "\":\"";
    entry += core::utils::escape_json_string(value);
    entry += '\"';
    return entry;
}

} // namespace

std::optional<AntigravityModelWireProfile>
antigravity_wire_profile(std::string_view model) {
    for (const auto& entry : kWireProfiles) {
        if (entry.model == model) return entry.profile;
    }
    return std::nullopt;
}

void GeminiAntigravityProtocol::prepare_request(ChatRequest& request) {
    GeminiCodeAssistProtocol::prepare_request(request);
    // Latch the conversation anchor so parse_event() can attribute the
    // response's responseId to this trajectory (see stream_anchor_).
    stream_anchor_ = conversation_anchor(request);
}

std::string GeminiAntigravityProtocol::serialize(const ChatRequest& req) const {
    const std::string anchor = conversation_anchor(req);
    const std::string model = req.model;
    const bool claude = is_claude_model(model);

    // One prior assistant message == one earlier request of this trajectory.
    const auto prior_steps = static_cast<int>(std::count_if(
        req.messages.begin(), req.messages.end(),
        [](const Message& m) { return m.role == "assistant"; }));
    const int step = 2 + prior_steps;

    const std::string trajectory_id = uuid_from_seed(anchor + "#trajectory");
    const std::string agent_id = uuid_from_seed(anchor + "#agent");
    const std::string used_claude = claude ? "true" : "false";
    const auto profile = antigravity_wire_profile(model);

    // Label order matches the real hub client. The previous response's
    // execution handle appears only once a response has been observed for
    // this conversation.
    std::string labels = "{";
    bool first_label = true;
    const auto append_label = [&](std::string_view key, std::string_view value) {
        if (!first_label) labels += ',';
        labels += labels_entry(key, value);
        first_label = false;
    };
    const std::string last_execution_id =
        AntigravityConversationState::instance().execution_id(anchor);
    if (!last_execution_id.empty()) {
        append_label("last_execution_id", last_execution_id);
    }
    append_label("last_step_index", std::to_string(step - 1));
    if (profile.has_value() && !profile->model_enum.empty()) {
        append_label("model_enum", profile->model_enum);
    }
    append_label("trajectory_id", trajectory_id);
    append_label("used_claude", used_claude);
    append_label("used_claude_conservative", used_claude);
    labels += '}';

    GeminiRequestExtras extras;
    extras.system_instruction_role = "user";
    extras.validated_tool_mode = true;
    extras.skip_thought_signature_on_first_call = is_gemini3_or_newer(model);
    extras.session_id = "-" + std::to_string(fnv1a64(anchor) & 0x7FFFFFFFFFFFFFFFULL);
    extras.labels_json = std::move(labels);
    if (profile.has_value()) {
        extras.max_output_tokens_cap = profile->max_output_tokens;
    } else if (claude) {
        extras.max_output_tokens_cap = kClaudeMaxOutputTokens;
    }

    std::string payload = serialize_gemini_code_assist_request(req, req.model, &extras);
    // serialize_gemini_code_assist_request always emits a single top-level
    // JSON object terminated by '}'; splice in the Antigravity envelope
    // rather than re-implementing Gemini request serialization here.
    if (!payload.empty() && payload.back() == '}') {
        payload.pop_back();
        payload += R"(,"requestId":"agent/)" + agent_id + '/' + next_request_timestamp_ms()
            + '/' + trajectory_id + '/' + std::to_string(step);
        payload += R"(","userAgent":"antigravity","requestType":"agent"})";
    }
    return payload;
}

cpr::Header GeminiAntigravityProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    // The real hub client sends only its `User-Agent` beyond the standard
    // auth/content headers — no X-Goog-Api-Client, no Client-Metadata.
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"User-Agent", core::auth::antigravity::user_agent()},
    };
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

std::string GeminiAntigravityProtocol::build_url(std::string_view base_url,
                                                 std::string_view model) const {
    // Computed from per-stream state: the first host until an attempt fails,
    // then the ordered hosts cycle. A user-supplied endpoint never rotates.
    static constexpr std::array<std::string_view, 2> kHosts{
        kAntigravityEndpoint, kAntigravitySandboxEndpoint};
    if (trim_trailing_slashes(base_url) == kHosts.front()) {
        base_url = kHosts[static_cast<std::size_t>(failed_attempts_) % kHosts.size()];
    }
    return GeminiCodeAssistProtocol::build_url(base_url, model);
}

ParseResult GeminiAntigravityProtocol::parse_event(std::string_view raw_event) {
    ParseResult result = GeminiCodeAssistProtocol::parse_event(raw_event);
    // The hub client echoes the previous responseId as labels.last_execution_id
    // on the next request of the trajectory; remember it for this conversation.
    if (!result.response_id.empty()) {
        AntigravityConversationState::instance().record_execution_id(
            stream_anchor_, result.response_id);
    }
    return result;
}

void GeminiAntigravityProtocol::on_response(const HttpResponse& response) {
    // Status 0 is a transport failure.
    if (response.status_code != 200 && is_retryable(response)) ++failed_attempts_;
}

bool GeminiAntigravityProtocol::is_retryable(const HttpResponse& response) const noexcept {
    const int status = response.status_code;
    return status == 0 || status == 408 || status == 429 || status >= 500;
}

} // namespace core::llm::protocols
