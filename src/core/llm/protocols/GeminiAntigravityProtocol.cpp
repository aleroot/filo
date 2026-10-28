#include "GeminiAntigravityProtocol.hpp"
#include "AntigravityConversationState.hpp"
#include "GeminiProtocol.hpp"
#include "../AntigravityModelTraits.hpp"
#include "../../utils/JsonUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace core::llm::protocols {

namespace {

constexpr int kClaudeMaxOutputTokens = 128000;
constexpr int kGeminiMaxOutputTokens = 65536;

// Per-wire-id constants snapshotted from `v1internal:fetchAvailableModels` as
// served to the 2.27.0 hub client (2026-10-28). `model_enum` is telemetry-only
// (the backend does not require it) but the real client never omits it for the
// ids it lists, so neither does Filo; an id absent from this snapshot simply
// sends no label. `max_output_tokens` is the fixed `generationConfig.
// maxOutputTokens` the client sends for that SKU.
struct WireProfileEntry {
    std::string_view model;
    AntigravityModelWireProfile profile;
};

constexpr std::array<WireProfileEntry, 22> kWireProfiles{{
    // Gemini 4 Argon — the frontier tier of the 2.27.0 client.
    {"gemini-4-argon-low", {"MODEL_PLACEHOLDER_M419", kGeminiMaxOutputTokens}},
    {"gemini-4-argon-medium", {"MODEL_PLACEHOLDER_M418", kGeminiMaxOutputTokens}},
    {"gemini-4-argon-high", {"MODEL_PLACEHOLDER_M417", kGeminiMaxOutputTokens}},
    // Gemini 3.8 Flash. `-tiered` is the SKU the hub's own agent runs: it
    // carries no display name and asks for a dynamic thinking budget.
    {"gemini-3.8-flash-low", {"MODEL_PLACEHOLDER_M320", kGeminiMaxOutputTokens}},
    {"gemini-3.8-flash-medium", {"MODEL_PLACEHOLDER_M319", kGeminiMaxOutputTokens}},
    {"gemini-3.8-flash-high", {"MODEL_PLACEHOLDER_M318", kGeminiMaxOutputTokens}},
    {"gemini-3.8-flash-tiered", {"MODEL_PLACEHOLDER_M322", kGeminiMaxOutputTokens}},
    {"gemini-3.5-flash-lite", {"MODEL_PLACEHOLDER_M198", 65535}},
    {"gemini-3.1-flash-lite", {"MODEL_PLACEHOLDER_M50", 65535}},
    {"gemini-3.1-flash-image", {"MODEL_PLACEHOLDER_M21", kGeminiMaxOutputTokens}},
    {"gemini-3-flash", {"MODEL_PLACEHOLDER_M18", kGeminiMaxOutputTokens}},
    {"gemini-2.5-pro", {"MODEL_GOOGLE_GEMINI_2_5_PRO", 65535}},
    {"gemini-2.5-flash", {"MODEL_GOOGLE_GEMINI_2_5_FLASH", 65535}},
    {"gemini-2.5-flash-lite", {"MODEL_GOOGLE_GEMINI_2_5_FLASH_LITE", 65535}},
    {"gemini-2.5-flash-thinking", {"MODEL_GOOGLE_GEMINI_2_5_FLASH_THINKING", 65535}},
    // Anthropic-backed SKUs: measured 128000 maxOutputTokens on
    // `daily-cloudcode-pa` (the 64000 ceiling of the 4.x ids is gone).
    {"claude-opus-5-5-low", {"MODEL_PLACEHOLDER_M400", kClaudeMaxOutputTokens}},
    {"claude-opus-5-5-medium", {"MODEL_PLACEHOLDER_M401", kClaudeMaxOutputTokens}},
    {"claude-opus-5-5-high", {"MODEL_PLACEHOLDER_M402", kClaudeMaxOutputTokens}},
    {"claude-sonnet-5-5-low", {"MODEL_PLACEHOLDER_M403", kClaudeMaxOutputTokens}},
    {"claude-sonnet-5-5-medium", {"MODEL_PLACEHOLDER_M404", kClaudeMaxOutputTokens}},
    {"claude-sonnet-5-5-high", {"MODEL_PLACEHOLDER_M405", kClaudeMaxOutputTokens}},
    {"gpt-oss-120b-medium", {"MODEL_OPENAI_GPT_OSS_120B_MEDIUM", 32768}},
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

/// The hub derives one signed 64-bit `sessionId` per installation and reuses it
/// for every conversation *and* every restart (a single value appears across
/// all captured cascades and language-server runs), so Filo derives one per
/// user/machine rather than per conversation.
[[nodiscard]] std::string installation_session_id() {
    static const std::string value = [] {
        std::string seed = "filo-antigravity";
        for (const char* key : {"USER", "HOME"}) {
            if (const char* raw = std::getenv(key); raw && raw[0] != '\0') {
                seed += '|';
                seed += raw;
            }
        }
        return std::to_string(static_cast<std::int64_t>(fnv1a64(seed)));
    }();
    return value;
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

/// `thinkingConfig.thinkingLevel` for an id that is a single SKU: the caller's
/// effort hint wins, else the SKU's own `-low` / `-medium` / `-high` suffix.
/// Empty when neither names a level, in which case the backend applies its
/// default.
[[nodiscard]] std::string thinking_level_for(std::string_view model,
                                             std::string_view effort) {
    const std::string hint = core::utils::str::to_lower_ascii_copy(effort);
    if (hint == "low") return "LOW";
    if (hint == "medium") return "MEDIUM";
    if (hint == "high" || hint == "max") return "HIGH";

    const std::string lowered = core::utils::str::to_lower_ascii_copy(model);
    if (lowered.ends_with("-low")) return "LOW";
    if (lowered.ends_with("-medium")) return "MEDIUM";
    if (lowered.ends_with("-high")) return "HIGH";
    return {};
}

/// The profile of `family` in the wire-profile snapshot: the fallback for
/// when the live catalog has not described it. Empty unless the snapshot sells
/// it at two or more levels.
[[nodiscard]] ModelReasoningProfile snapshot_family_profile(std::string_view family) {
    std::vector<std::string_view> levels;
    for (const std::string_view level : antigravity::kThinkingLevels) {
        if (antigravity_wire_profile(antigravity::thinking_variant_id(family, level))) {
            levels.push_back(level);
        }
    }
    return levels.size() >= 2 ? antigravity::thinking_family_profile(levels) : ModelReasoningProfile{};
}

/// The SKU a request is sent to and the `thinkingLevel` that goes with it.
struct WireModel {
    std::string id;
    std::string thinking_level;
};

/// Resolves the requested model to the SKU on the wire. A family sold at
/// several thinking levels (`claude-opus-5-5`, or a configured variant id such
/// as `claude-opus-5-5-high`) becomes the SKU of the level the request's effort
/// selects, so the id, its `model_enum` and `thinkingLevel` always agree. The
/// levels come from the live catalog when it described the family, else from
/// the wire-profile snapshot. Any other id is a single SKU and is sent as is.
[[nodiscard]] WireModel resolve_wire_model(const ChatRequest& req) {
    const std::string model = normalize_requested_gemini_model(req.model);
    const auto variant = antigravity::split_thinking_variant(model);
    const std::string_view family = variant ? variant->family : std::string_view(model);

    const ModelReasoningProfile profile =
        req.catalog_reasoning && req.catalog_reasoning->supported_efforts.size() >= 2
        ? *req.catalog_reasoning
        : snapshot_family_profile(family);
    if (profile.supported_efforts.empty()) {
        return {model, thinking_level_for(model, req.effort)};
    }

    const std::string level = antigravity::select_thinking_level(
        profile, req.effort, variant ? variant->level : std::string_view{});
    std::string thinking_level = level;
    std::ranges::transform(thinking_level, thinking_level.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return {antigravity::thinking_variant_id(family, level), std::move(thinking_level)};
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

GeminiAntigravityProtocol::GeminiAntigravityProtocol(std::shared_ptr<antigravity::IUsageSource> usage)
    : usage_(usage ? std::move(usage) : antigravity::make_usage_source()) {}

void GeminiAntigravityProtocol::prepare_request(ChatRequest& request) {
    GeminiCodeAssistProtocol::prepare_request(request);
    // Latch the conversation anchor so parse_event() can attribute the
    // response's responseId to this trajectory (see stream_anchor_).
    stream_anchor_ = conversation_anchor(request);
    wire_model_ = resolve_wire_model(request).id;
    const auto project = request.auth_properties.find("project_id");
    project_ = project != request.auth_properties.end() ? project->second : std::string{};
}

ReasoningCapabilities
GeminiAntigravityProtocol::reasoning_capabilities(std::string_view model) const noexcept {
    try {
        const std::string id = normalize_requested_gemini_model(model);
        const auto variant = antigravity::split_thinking_variant(id);
        return snapshot_family_profile(variant ? variant->family : std::string_view(id)).effort;
    } catch (...) {
        // A capability query must not fail; the live catalog still answers.
    }
    return {};
}

std::string GeminiAntigravityProtocol::serialize(const ChatRequest& req) const {
    const std::string anchor = conversation_anchor(req);
    const WireModel wire = resolve_wire_model(req);
    const std::string& model = wire.id;
    const std::string lowered = core::utils::str::to_lower_ascii_copy(model);
    const bool claude = is_claude_model(model);
    const bool gemini = lowered.starts_with("gemini-");

    // The hub numbers a trajectory by the `contents` it sends: one content per
    // turn, so the first request of a cascade carries 1, the request after one
    // tool round carries 3, and so on. `requestId` ends with that count and
    // `labels.last_step_index` with count-1, while `labels.request_id` ends
    // with the 0-based index of this request inside the trajectory — one prior
    // assistant message == one prior request.
    const auto contents = static_cast<int>(std::count_if(
        req.messages.begin(), req.messages.end(),
        [](const Message& m) { return m.role != "system"; }));
    const auto prior_requests = static_cast<int>(std::count_if(
        req.messages.begin(), req.messages.end(),
        [](const Message& m) { return m.role == "assistant"; }));
    const int step = std::max(contents, 1);

    const std::string trajectory_id = uuid_from_seed(anchor + "#trajectory");
    const std::string cascade_id = uuid_from_seed(anchor + "#agent");
    const std::string used_claude = claude ? "true" : "false";
    const auto profile = antigravity_wire_profile(model);

    // Alphabetical, exactly as the hub serialises it. The previous response's
    // execution handle appears only once a response has been observed for
    // this conversation.
    std::string labels = "{";
    bool first_label = true;
    const auto append_label = [&](std::string_view key, std::string_view value) {
        if (!first_label) labels += ',';
        labels += labels_entry(key, value);
        first_label = false;
    };
    append_label("cascade_id", cascade_id);
    if (const std::string last_execution_id =
            AntigravityConversationState::instance().execution_id(anchor);
        !last_execution_id.empty()) {
        append_label("last_execution_id", last_execution_id);
    }
    append_label("last_step_index", std::to_string(step - 1));
    if (profile.has_value() && !profile->model_enum.empty()) {
        append_label("model_enum", profile->model_enum);
    }
    append_label("request_id", trajectory_id + "-" + std::to_string(prior_requests));
    append_label("root_cascade_id", cascade_id);
    append_label("trajectory_id", trajectory_id);
    append_label("used_claude", used_claude);
    append_label("used_claude_conservative", used_claude);
    append_label("used_non_gemini_model", gemini ? "false" : "true");
    labels += '}';

    GeminiRequestExtras extras;
    extras.system_instruction_role = "user";
    // The hub sends no `toolConfig` at all, one `functionDeclarations` element
    // per tool, and one thought signature per replayed call part.
    extras.tool_declaration_per_element = true;
    extras.skip_thought_signature_on_all_calls = is_gemini3_or_newer(model);
    extras.session_id = installation_session_id();
    extras.labels_json = std::move(labels);
    if (profile.has_value()) {
        extras.max_output_tokens_cap = profile->max_output_tokens;
    } else if (claude) {
        extras.max_output_tokens_cap = kClaudeMaxOutputTokens;
    }
    if (antigravity::takes_thinking_config(model)) {
        // includeThoughts is the only way the backend surfaces reasoning: the
        // model thinks either way, but without it no thought parts come back.
        extras.include_thoughts = true;
        if (lowered.ends_with("-tiered")) {
            extras.thinking_budget = -1;
        } else {
            extras.thinking_level = wire.thinking_level;
        }
    }

    // Envelope in the hub's own key order: project, requestId, request, model,
    // userAgent, requestType.
    std::string payload = "{";
    if (const auto project = req.auth_properties.find("project_id");
        project != req.auth_properties.end() && !project->second.empty()) {
        payload += R"("project":")";
        payload += core::utils::escape_json_string(project->second);
        payload += "\",";
    }
    payload += R"("requestId":"agent/)" + cascade_id + '/' + next_request_timestamp_ms()
        + '/' + trajectory_id + '/' + std::to_string(step);
    payload += R"(","request":)";
    payload += serialize_gemini_request(req, req.model, &extras);
    payload += R"(,"model":")";
    payload += core::utils::escape_json_string(model);
    payload += R"(","userAgent":"antigravity","requestType":"agent"})";
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
    // No `retry_after`: every 429 is retryable here, so a server delay (an
    // exhausted quota's reset) would hold the request instead of failing it,
    // and the retry goes to the next host anyway.
    rate_limit_ = {};
    rate_limit_.is_rate_limited = response.status_code == 429;
}

void GeminiAntigravityProtocol::enrich_rate_limit(std::string_view base_url,
                                                  const cpr::Header& headers,
                                                  const HttpResponse& response) {
    // After a transport failure the read would most likely wait out another
    // timeout, and its failure would then be cached.
    if (!usage_ || wire_model_.empty() || response.status_code == 0) return;
    try {
        if (auto quota = usage_->read(base_url, headers, project_, wire_model_)) {
            quota->is_rate_limited |= rate_limit_.is_rate_limited;
            rate_limit_ = std::move(*quota);
        }
    } catch (...) { /* Usage is best-effort and must not fail generation. */ }
}

bool GeminiAntigravityProtocol::is_retryable(const HttpResponse& response) const noexcept {
    const int status = response.status_code;
    return status == 0 || status == 408 || status == 429 || status >= 500;
}

} // namespace core::llm::protocols
