#pragma once

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <fstream>
#include "../tools/Tool.hpp"
#include "../tools/ToolSchema.hpp"
#include "../context/PromptPlan.hpp"
#include "../utils/Base64.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/MimeUtils.hpp"
#include "ResponseFormat.hpp"

namespace core::llm {

inline constexpr std::uintmax_t kMaxLocalVideoBytes = 100ULL * 1024ULL * 1024ULL;
inline constexpr std::uintmax_t kMaxInlineVideoBytes = 20ULL * 1024ULL * 1024ULL;

// ---------------------------------------------------------------------------
// TokenUsage — populated by each provider after every LLM call.
// ---------------------------------------------------------------------------
struct TokenUsage {
    int32_t prompt_tokens     = 0;
    int32_t completion_tokens = 0;
    int32_t total_tokens      = 0;  // may be computed as prompt + completion
    int32_t cached_prompt_tokens = 0;
    int32_t cache_creation_prompt_tokens = 0;
    int32_t reasoning_tokens = 0;

    [[nodiscard]] TokenUsage operator+(const TokenUsage& o) const noexcept {
        return {
            .prompt_tokens = prompt_tokens + o.prompt_tokens,
            .completion_tokens = completion_tokens + o.completion_tokens,
            .total_tokens = total_tokens + o.total_tokens,
            .cached_prompt_tokens = cached_prompt_tokens + o.cached_prompt_tokens,
            .cache_creation_prompt_tokens =
                cache_creation_prompt_tokens + o.cache_creation_prompt_tokens,
            .reasoning_tokens = reasoning_tokens + o.reasoning_tokens,
        };
    }
    TokenUsage& operator+=(const TokenUsage& o) noexcept {
        prompt_tokens     += o.prompt_tokens;
        completion_tokens += o.completion_tokens;
        total_tokens      += o.total_tokens;
        cached_prompt_tokens += o.cached_prompt_tokens;
        cache_creation_prompt_tokens += o.cache_creation_prompt_tokens;
        reasoning_tokens += o.reasoning_tokens;
        return *this;
    }
    [[nodiscard]] bool has_data() const noexcept { return prompt_tokens > 0 || completion_tokens > 0; }
};

struct ToolCall {
    int index = -1; // Critical for OpenAI SSE streaming multiple tool calls
    std::string id = {};
    std::string type = "function";
    struct Function {
        std::string name = {};
        std::string arguments = {}; // JSON string
    } function;
};

enum class ContentPartType {
    Text,
    Image,
    Video,
};

struct ContentPart {
    ContentPartType type = ContentPartType::Text;
    std::string text = {};
    std::string path = {};
    std::string url = {};
    std::string media_id = {};
    std::string mime_type = {};
    std::string detail = "auto";

    [[nodiscard]] static ContentPart make_text(std::string value) {
        return ContentPart{
            .type = ContentPartType::Text,
            .text = std::move(value),
        };
    }

    [[nodiscard]] static ContentPart make_image(std::string value,
                                                std::string mime = {},
                                                std::string level = "auto") {
        return ContentPart{
            .type = ContentPartType::Image,
            .path = std::move(value),
            .mime_type = std::move(mime),
            .detail = std::move(level),
        };
    }

    [[nodiscard]] static ContentPart make_video(std::string value,
                                                std::string mime = {}) {
        return ContentPart{
            .type = ContentPartType::Video,
            .path = std::move(value),
            .mime_type = std::move(mime),
        };
    }

    [[nodiscard]] static ContentPart make_image_url(std::string value,
                                                    std::string id = {},
                                                    std::string level = "auto") {
        return ContentPart{
            .type = ContentPartType::Image,
            .url = std::move(value),
            .media_id = std::move(id),
            .detail = std::move(level),
        };
    }

    [[nodiscard]] static ContentPart make_video_url(std::string value,
                                                    std::string id = {}) {
        return ContentPart{
            .type = ContentPartType::Video,
            .url = std::move(value),
            .media_id = std::move(id),
        };
    }
};

[[nodiscard]] inline std::string_view media_kind(ContentPartType type) noexcept {
    switch (type) {
        case ContentPartType::Image:
            return "image";
        case ContentPartType::Video:
            return "video";
        case ContentPartType::Text:
            break;
    }
    return "text";
}

[[nodiscard]] inline bool is_media_part(ContentPartType type) noexcept {
    return type == ContentPartType::Image || type == ContentPartType::Video;
}

[[nodiscard]] inline bool is_video_mime(std::string_view mime_type) noexcept {
    return mime_type.starts_with("video/");
}

[[nodiscard]] inline bool is_data_url_for_media(ContentPartType type,
                                                std::string_view url) noexcept {
    if (type == ContentPartType::Image) {
        return url.starts_with("data:image/");
    }
    if (type == ContentPartType::Video) {
        return url.starts_with("data:video/");
    }
    return false;
}

[[nodiscard]] inline bool is_media_reference_url(ContentPartType type,
                                                 std::string_view url) noexcept {
    if (url.empty() || type == ContentPartType::Text) {
        return false;
    }
    return is_data_url_for_media(type, url)
        || url.starts_with("ms://")
        || url.starts_with("http://")
        || url.starts_with("https://");
}

[[nodiscard]] inline std::string_view media_reference(const ContentPart& part) noexcept {
    if (!part.path.empty()) {
        return part.path;
    }
    if (!part.url.empty()) {
        return part.url;
    }
    return {};
}

[[nodiscard]] inline std::string describe_image_attachment(std::string_view path) {
    return "[Attached image: " + std::string(path) + "]";
}

[[nodiscard]] inline std::string describe_video_attachment(std::string_view path) {
    return "[Attached video: " + std::string(path) + "]";
}

[[nodiscard]] inline std::string describe_media_attachment(ContentPartType type,
                                                           std::string_view path) {
    if (type == ContentPartType::Video) {
        return describe_video_attachment(path);
    }
    return describe_image_attachment(path);
}

[[nodiscard]] inline std::string unavailable_image_attachment_text(std::string_view path) {
    return "[Attached image unavailable: " + std::string(path) + "]";
}

[[nodiscard]] inline std::string unavailable_video_attachment_text(std::string_view path) {
    return "[Attached video unavailable: " + std::string(path) + "]";
}

[[nodiscard]] inline std::string unavailable_media_attachment_text(ContentPartType type,
                                                                   std::string_view path) {
    if (type == ContentPartType::Video) {
        return unavailable_video_attachment_text(path);
    }
    return unavailable_image_attachment_text(path);
}

struct EncodedMediaPart {
    ContentPartType type = ContentPartType::Image;
    std::string path = {};
    std::string url = {};
    std::string media_id = {};
    std::string mime_type = {};
    std::string base64_data = {};
    std::string detail = "auto";

    [[nodiscard]] std::string data_url() const {
        if (!url.empty()) {
            return url;
        }
        return "data:" + mime_type + ";base64," + base64_data;
    }

    [[nodiscard]] bool is_url_reference() const noexcept {
        return !url.empty();
    }
};

using EncodedImagePart = EncodedMediaPart;

[[nodiscard]] inline std::optional<EncodedMediaPart> encode_media_part(
    const ContentPart& part) {
    if (!is_media_part(part.type)) {
        return std::nullopt;
    }

    if (!part.url.empty()) {
        if (!is_media_reference_url(part.type, part.url)) {
            return std::nullopt;
        }
        return EncodedMediaPart{
            .type = part.type,
            .path = part.path,
            .url = part.url,
            .media_id = part.media_id,
            .mime_type = part.mime_type,
            .detail = part.detail.empty() ? "auto" : part.detail,
        };
    }

    if (part.path.empty()) {
        return std::nullopt;
    }

    const std::filesystem::path path = part.path;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        return std::nullopt;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::nullopt;
    }

    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        return std::nullopt;
    }
    if (part.type == ContentPartType::Video
        && static_cast<std::uintmax_t>(size) > kMaxInlineVideoBytes) {
        return std::nullopt;
    }
    input.seekg(0, std::ios::beg);

    std::string bytes(static_cast<std::size_t>(size), '\0');
    if (size > 0) {
        input.read(bytes.data(), static_cast<std::streamsize>(size));
        if (!input) {
            return std::nullopt;
        }
    }

    std::string mime_type = part.mime_type.empty()
        ? core::utils::mime::guess_type(path, false)
        : part.mime_type;
    if (part.type == ContentPartType::Image && !mime_type.starts_with("image/")) {
        return std::nullopt;
    }
    if (part.type == ContentPartType::Video && !is_video_mime(mime_type)) {
        return std::nullopt;
    }

    return EncodedMediaPart{
        .type = part.type,
        .path = part.path,
        .url = part.url,
        .media_id = part.media_id,
        .mime_type = std::move(mime_type),
        .base64_data = core::utils::Base64::encode(bytes),
        .detail = part.detail.empty() ? "auto" : part.detail,
    };
}

[[nodiscard]] inline std::optional<EncodedImagePart> encode_image_part(
    const ContentPart& part) {
    if (part.type != ContentPartType::Image) {
        return std::nullopt;
    }
    return encode_media_part(part);
}

[[nodiscard]] inline bool message_has_image_input(const std::vector<ContentPart>& parts) noexcept {
    for (const auto& part : parts) {
        if (part.type == ContentPartType::Image) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool message_has_video_input(const std::vector<ContentPart>& parts) noexcept {
    for (const auto& part : parts) {
        if (part.type == ContentPartType::Video) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool message_has_media_input(const std::vector<ContentPart>& parts) noexcept {
    for (const auto& part : parts) {
        if (is_media_part(part.type)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline std::string render_content_parts_as_text(
    const std::vector<ContentPart>& parts) {
    std::string text;
    for (const auto& part : parts) {
        if (part.type == ContentPartType::Text) {
            text += part.text;
            continue;
        }

        text += describe_media_attachment(
            part.type,
            media_reference(part).empty()
                ? std::string_view{part.type == ContentPartType::Video ? "<video>" : "<image>"}
                : media_reference(part));
    }
    return text;
}

[[nodiscard]] inline std::string collapse_text_parts(const std::vector<ContentPart>& parts) {
    std::string text;
    for (const auto& part : parts) {
        if (part.type == ContentPartType::Text) {
            text += part.text;
        }
    }
    return text;
}

// ---------------------------------------------------------------------------
// StreamChunk — represents a single chunk from an LLM streaming response.
// ---------------------------------------------------------------------------
struct ContinuationItem {
    std::string provider = {}; ///< Wire-protocol owner (for example "openai" or "anthropic").
    std::string kind = {};     ///< Provider item/block type; informational and forward-compatible.
    std::string payload = {};  ///< Complete provider JSON item, replayed without interpretation.
};

[[nodiscard]] inline bool has_valid_continuation_payload(
    const ContinuationItem& item) {
    if (item.payload.empty()) return false;
    simdjson::dom::parser parser;
    simdjson::padded_string padded(item.payload);
    simdjson::dom::object object;
    return parser.parse(padded).get(object) == simdjson::SUCCESS;
}

struct StreamChunk {
    std::string content;           // Text content from the model
    std::string reasoning_content; // Provider-specific thinking/reasoning content
    std::string stop_reason;       // Provider terminal reason, populated on final chunks when available
    std::vector<ToolCall> tools;   // Tool calls (if any)
    bool is_final = false;         // True if this is the last chunk
    bool is_error = false;         // True if this chunk represents an API/HTTP error
    bool incomplete_tool_call = false; // True if the stream ended mid-tool_use block
    std::vector<ContinuationItem> continuation_items; // Opaque signed/encrypted reasoning state

    // Factory methods for common cases
    [[nodiscard]] static StreamChunk make_final(
        std::string stop_reason = {},
        bool incomplete_tool_call = false) {
        return StreamChunk{
            .content = {},
            .reasoning_content = {},
            .stop_reason = std::move(stop_reason),
            .tools = {},
            .is_final = true,
            .is_error = false,
            .incomplete_tool_call = incomplete_tool_call,
        };
    }
    [[nodiscard]] static StreamChunk make_error(std::string message) {
        return StreamChunk{
            .content = std::move(message),
            .reasoning_content = {},
            .stop_reason = {},
            .tools = {},
            .is_final = true,
            .is_error = true,
        };
    }
    [[nodiscard]] static StreamChunk make_content(std::string text) {
        return StreamChunk{
            .content = std::move(text),
            .reasoning_content = {},
            .stop_reason = {},
            .tools = {},
            .is_final = false,
            .is_error = false,
        };
    }
    [[nodiscard]] static StreamChunk make_tools(std::vector<ToolCall> tool_calls) {
        return StreamChunk{
            .content = {},
            .reasoning_content = {},
            .stop_reason = {},
            .tools = std::move(tool_calls),
            .is_final = false,
            .is_error = false,
        };
    }
    [[nodiscard]] static StreamChunk make_reasoning(std::string text) {
        return StreamChunk{
            .content = {},
            .reasoning_content = std::move(text),
            .stop_reason = {},
            .tools = {},
            .is_final = false,
            .is_error = false,
        };
    }
};

struct Message {
    std::string role = {};
    std::string content = {};
    std::string name = {};                 // Optional, for tool role
    std::string tool_call_id = {};         // Optional, for tool role
    std::vector<ToolCall> tool_calls = {}; // Optional, for assistant role
    std::string reasoning_content = {};    // Optional provider-specific reasoning text
    std::vector<ContentPart> content_parts = {}; // Optional, for multimodal user input
    std::string input_text = {};            // Original editable prompt before mention/media expansion
    bool synthetic = false;                // Internal context record, not a visible user prompt
    std::vector<ContinuationItem> continuation_items = {}; // Opaque provider continuation state
};

struct Tool {
    std::string type = "function";
    core::tools::ToolDefinition function;
};

struct ChatRequest {
    std::string model;
    std::vector<Message> messages;
    bool stream = true;
    std::optional<float> temperature;
    std::optional<int> max_tokens;
    std::vector<Tool> tools;
    ResponseFormat response_format;  ///< For JSON mode / structured outputs
    std::string previous_response_id = {}; ///< Responses API incremental continuation id
    std::string prompt_cache_key = {};     ///< Responses API prompt cache key
    std::string service_tier = {};         ///< Responses API service_tier (e.g. "priority")
    std::string effort = {};               ///< Optional effort hint ("low"/"medium"/"high"/"max")
    std::string session_id = {};            ///< Internal session scope for accounting/routing decisions
    std::string transport_turn_id = {};     ///< Short-lived provider transport scope for one agent turn
    bool stream_include_usage = false;      ///< Per-request stream usage request (OpenAI-compatible APIs)
    std::unordered_map<std::string, std::string> auth_properties = {}; ///< Provider auth metadata projected into the request lifecycle
    core::context::PromptPlan prompt_plan; ///< Structured stable-to-dynamic system prompt.
};

[[nodiscard]] inline bool message_has_image_input(const Message& msg) noexcept {
    return message_has_image_input(msg.content_parts);
}

[[nodiscard]] inline bool message_has_video_input(const Message& msg) noexcept {
    return message_has_video_input(msg.content_parts);
}

[[nodiscard]] inline bool message_has_media_input(const Message& msg) noexcept {
    return message_has_media_input(msg.content_parts);
}

[[nodiscard]] inline bool request_has_image_input(const ChatRequest& req) noexcept {
    for (const auto& msg : req.messages) {
        if (message_has_image_input(msg)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool request_has_video_input(const ChatRequest& req) noexcept {
    for (const auto& msg : req.messages) {
        if (message_has_video_input(msg)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool request_has_media_input(const ChatRequest& req) noexcept {
    for (const auto& msg : req.messages) {
        if (message_has_media_input(msg)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool latest_user_message_has_image_input(const ChatRequest& req) noexcept {
    for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
        if (it->role == "user") {
            return message_has_image_input(*it);
        }
    }
    return false;
}

[[nodiscard]] inline bool latest_user_message_has_video_input(const ChatRequest& req) noexcept {
    for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
        if (it->role == "user") {
            return message_has_video_input(*it);
        }
    }
    return false;
}

[[nodiscard]] inline bool latest_user_message_has_media_input(const ChatRequest& req) noexcept {
    for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
        if (it->role == "user") {
            return message_has_media_input(*it);
        }
    }
    return false;
}

[[nodiscard]] inline std::string message_text_for_display(const Message& msg) {
    if (!msg.content.empty()) {
        return msg.content;
    }
    return render_content_parts_as_text(msg.content_parts);
}

inline void degrade_message_to_text_only(Message& msg) {
    if (msg.content.empty()) {
        msg.content = render_content_parts_as_text(msg.content_parts);
    }
    msg.content_parts.clear();
}

struct MediaDegradationOptions {
    bool images = true;
    bool videos = true;
};

[[nodiscard]] inline bool message_has_degradable_media_input(
    const Message& msg,
    const MediaDegradationOptions options) noexcept {
    for (const auto& part : msg.content_parts) {
        if ((options.images && part.type == ContentPartType::Image)
            || (options.videos && part.type == ContentPartType::Video)) {
            return true;
        }
    }
    return false;
}

inline void degrade_historical_media_inputs(
    ChatRequest& req,
    MediaDegradationOptions options = {}) {
    bool kept_latest_user = false;
    for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
        if (it->role == "user" && !kept_latest_user) {
            kept_latest_user = true;
            continue;
        }

        if (message_has_degradable_media_input(*it, options)) {
            degrade_message_to_text_only(*it);
        }
    }
}

inline void degrade_historical_image_inputs(ChatRequest& req) {
    degrade_historical_media_inputs(req, {.images = true, .videos = false});
}

inline void degrade_historical_video_inputs(ChatRequest& req) {
    degrade_historical_media_inputs(req, {.images = false, .videos = true});
}

struct Serializer {
    struct Options {
        // Vendor extension: some providers require assistant.reasoning_content
        // to be echoed back on follow-up tool turns. Disabled by default.
        bool include_reasoning_content = false;
        // OpenAI-compatible providers do not all agree on completion-budget
        // spelling.
        std::string max_tokens_field = "max_tokens";
        // Some providers need provider-specific schema normalization.
        std::function<std::string(std::string_view, bool)> transform_tool_schema;
        std::function<std::optional<std::string>(const Tool&)> serialize_tool_override;
    };

    static std::string serialize(const ChatRequest& req) {
        return serialize(req, Options{});
    }

    static std::string serialize(const ChatRequest& req, const Options& options) {
        std::string payload;
        payload.reserve(4096);
        payload += R"({"model":")";
        payload += core::utils::escape_json_string(req.model);
        payload += R"(","stream":)";
        payload += req.stream ? "true" : "false";
        
        // C++26 optimization: Use std::to_chars when available (faster than std::to_string).
        if (req.temperature.has_value()) {
            payload += R"(,"temperature":)";
            // Note: C++26 adds floating-point to_chars; use snprintf as high-performance fallback
            char tmp[32];
            int n = std::snprintf(tmp, sizeof(tmp), "%.6g", req.temperature.value());
            if (n > 0 && n < static_cast<int>(sizeof(tmp))) {
                payload.append(tmp, n);
            }
        }
        if (req.max_tokens.has_value()) {
            payload += R"(,")";
            payload += core::utils::escape_json_string(options.max_tokens_field);
            payload += R"(":)";
            char tmp[32];
            auto [ptr, _] = std::to_chars(tmp, tmp + sizeof(tmp), req.max_tokens.value());
            payload.append(tmp, ptr);
        }
        
        // Response format (JSON mode / structured outputs)
        if (req.response_format.is_structured()) {
            payload += R"(,"response_format":{"type":")";
            payload += req.response_format.to_string();
            payload += '"';
            if (req.response_format.type == ResponseFormat::Type::JsonSchema && !req.response_format.schema.empty()) {
                payload += R"(,"schema":)" + req.response_format.schema;
            }
            payload += "}";
        }

        if (!req.tools.empty()) {
            payload += R"(,"tools":[)";
            for (size_t i = 0; i < req.tools.size(); ++i) {
                const auto& def = req.tools[i].function;
                if (options.serialize_tool_override) {
                    if (auto serialized = options.serialize_tool_override(req.tools[i]);
                        serialized.has_value()) {
                        payload += *serialized;
                        if (i < req.tools.size() - 1) payload += ",";
                        continue;
                    }
                }

                const auto transform_schema = [&options](std::string_view schema,
                                                         bool root_is_property) {
                    if (options.transform_tool_schema) {
                        return options.transform_tool_schema(schema, root_is_property);
                    }
                    return std::string(schema);
                };

                payload += R"({"type":")" + req.tools[i].type + R"(","function":{"name":")" + core::utils::escape_json_string(def.name) + R"(","description":")" + core::utils::escape_json_string(def.description) + R"(","parameters":)";
                const std::string canonical_schema =
                    core::tools::schema::canonical_input_schema(def);
                payload += transform_schema(canonical_schema, false);
                payload += "}}";
                if (i < req.tools.size() - 1) payload += ",";
            }
            payload += "]";
        }

        payload += R"(,"messages":[)";
        for (size_t i = 0; i < req.messages.size(); ++i) {
            payload += R"({"role":")";
            payload += core::utils::escape_json_string(req.messages[i].role);
            payload += "\"";

            if (!req.messages[i].name.empty()) {
                payload += R"(,"name":")" + core::utils::escape_json_string(req.messages[i].name) + "\"";
            }
            if (!req.messages[i].tool_call_id.empty()) {
                payload += R"(,"tool_call_id":")" + core::utils::escape_json_string(req.messages[i].tool_call_id) + "\"";
            }
            if (!req.messages[i].content_parts.empty()) {
                payload += R"(,"content":[)";
                bool first_part = true;
                bool has_text_part = false;
                for (const auto& part : req.messages[i].content_parts) {
                    if (part.type == ContentPartType::Text && !part.text.empty()) {
                        has_text_part = true;
                        break;
                    }
                }
                if (!has_text_part && !req.messages[i].content.empty()
                    && req.messages[i].content
                        != render_content_parts_as_text(req.messages[i].content_parts)) {
                    payload += R"({"type":"text","text":")";
                    payload += core::utils::escape_json_string(req.messages[i].content);
                    payload += R"("})";
                    first_part = false;
                }
                for (const auto& part : req.messages[i].content_parts) {
                    if (part.type == ContentPartType::Text) {
                        if (part.text.empty()) continue;
                        if (!first_part) payload += ",";
                        payload += R"({"type":"text","text":")";
                        payload += core::utils::escape_json_string(part.text);
                        payload += R"("})";
                        first_part = false;
                        continue;
                    }

                    if (const auto encoded = encode_media_part(part); encoded.has_value()) {
                        if (!first_part) payload += ",";
                        const std::string_view kind = media_kind(encoded->type);
                        payload += R"({"type":")";
                        payload += kind;
                        payload += R"(_url",")";
                        payload += kind;
                        payload += R"(_url":{"url":")";
                        payload += core::utils::escape_json_string(encoded->data_url());
                        payload += R"(")";
                        if (!encoded->media_id.empty()) {
                            payload += R"(,"id":")";
                            payload += core::utils::escape_json_string(encoded->media_id);
                            payload += R"(")";
                        }
                        if (encoded->type == ContentPartType::Image) {
                            payload += R"(,"detail":")";
                            payload += core::utils::escape_json_string(encoded->detail);
                            payload += R"(")";
                        }
                        payload += "}}";
                        first_part = false;
                    } else {
                        if (!first_part) payload += ",";
                        payload += R"({"type":"text","text":")";
                        payload += core::utils::escape_json_string(
                            unavailable_media_attachment_text(part.type, media_reference(part)));
                        payload += R"("})";
                        first_part = false;
                    }
                }
                if (first_part) {
                    payload += R"({"type":"text","text":")";
                    payload += core::utils::escape_json_string(req.messages[i].content);
                    payload += R"("})";
                }
                payload += "]";
            } else if (!req.messages[i].content.empty()) {
                payload += R"(,"content":")" + core::utils::escape_json_string(req.messages[i].content) + "\"";
            } else if (req.messages[i].tool_calls.empty()) {
                payload += R"(,"content":null)";
            }

            if (!req.messages[i].tool_calls.empty()) {
                payload += R"(,"tool_calls":[)";
                for (size_t j = 0; j < req.messages[i].tool_calls.size(); ++j) {
                    const auto& tc = req.messages[i].tool_calls[j];
                    payload += R"({"id":")" + core::utils::escape_json_string(tc.id) + R"(","type":")" + core::utils::escape_json_string(tc.type) + R"(","function":{"name":")" + core::utils::escape_json_string(tc.function.name) + R"(","arguments":")" + core::utils::escape_json_string(tc.function.arguments) + R"("}})";
                    if (j < req.messages[i].tool_calls.size() - 1) payload += ",";
                }
                payload += "]";
            }

            // Vendor extension: only emitted when explicitly enabled.
            if (options.include_reasoning_content
                && !req.messages[i].reasoning_content.empty()) {
                payload += ",\"reasoning_content\":\"" + core::utils::escape_json_string(req.messages[i].reasoning_content) + "\"";
            }

            payload += "}";
            if (i < req.messages.size() - 1) payload += ",";
        }
        payload += "]}";
        return payload;
    }
};

} // namespace core::llm
