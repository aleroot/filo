#pragma once

#include <string>
#include <vector>
#include <optional>
#include <sstream>
#include <iomanip>
#include <charconv>
#include <filesystem>
#include <fstream>
#include "../tools/Tool.hpp"
#include "../utils/Base64.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/MimeUtils.hpp"

namespace core::llm {

// ---------------------------------------------------------------------------
// TokenUsage — populated by each provider after every LLM call.
// ---------------------------------------------------------------------------
struct TokenUsage {
    int32_t prompt_tokens     = 0;
    int32_t completion_tokens = 0;
    int32_t total_tokens      = 0;  // may be computed as prompt + completion

    [[nodiscard]] TokenUsage operator+(const TokenUsage& o) const noexcept {
        return { prompt_tokens     + o.prompt_tokens,
                 completion_tokens + o.completion_tokens,
                 total_tokens      + o.total_tokens };
    }
    TokenUsage& operator+=(const TokenUsage& o) noexcept {
        prompt_tokens     += o.prompt_tokens;
        completion_tokens += o.completion_tokens;
        total_tokens      += o.total_tokens;
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
};

struct ContentPart {
    ContentPartType type = ContentPartType::Text;
    std::string text = {};
    std::string path = {};
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
};

[[nodiscard]] inline std::string describe_image_attachment(std::string_view path) {
    return "[Attached image: " + std::string(path) + "]";
}

[[nodiscard]] inline std::string unavailable_image_attachment_text(std::string_view path) {
    return "[Attached image unavailable: " + std::string(path) + "]";
}

struct EncodedImagePart {
    std::string path = {};
    std::string mime_type = {};
    std::string base64_data = {};
    std::string detail = "auto";

    [[nodiscard]] std::string data_url() const {
        return "data:" + mime_type + ";base64," + base64_data;
    }
};

[[nodiscard]] inline std::optional<EncodedImagePart> encode_image_part(
    const ContentPart& part) {
    if (part.type != ContentPartType::Image || part.path.empty()) {
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
    if (!mime_type.starts_with("image/")) {
        return std::nullopt;
    }

    return EncodedImagePart{
        .path = part.path,
        .mime_type = std::move(mime_type),
        .base64_data = core::utils::Base64::encode(bytes),
        .detail = part.detail.empty() ? "auto" : part.detail,
    };
}

[[nodiscard]] inline bool message_has_image_input(const std::vector<ContentPart>& parts) noexcept {
    for (const auto& part : parts) {
        if (part.type == ContentPartType::Image) {
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

        text += describe_image_attachment(
            part.path.empty() ? std::string_view{"<image>"} : std::string_view{part.path});
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
struct StreamChunk {
    std::string content;           // Text content from the model
    std::string reasoning_content; // Thinking/reasoning content (Kimi K2.5)
    std::vector<ToolCall> tools;   // Tool calls (if any)
    bool is_final = false;         // True if this is the last chunk
    bool is_error = false;         // True if this chunk represents an API/HTTP error

    // Factory methods for common cases
    [[nodiscard]] static StreamChunk make_final() {
        return StreamChunk{ .content = {}, .reasoning_content = {}, .tools = {}, .is_final = true, .is_error = false };
    }
    [[nodiscard]] static StreamChunk make_error(std::string message) {
        return StreamChunk{ .content = std::move(message), .reasoning_content = {}, .tools = {}, .is_final = true, .is_error = true };
    }
    [[nodiscard]] static StreamChunk make_content(std::string text) {
        return StreamChunk{ .content = std::move(text), .reasoning_content = {}, .tools = {}, .is_final = false, .is_error = false };
    }
    [[nodiscard]] static StreamChunk make_tools(std::vector<ToolCall> tool_calls) {
        return StreamChunk{ .content = {}, .reasoning_content = {}, .tools = std::move(tool_calls), .is_final = false, .is_error = false };
    }
    [[nodiscard]] static StreamChunk make_reasoning(std::string text) {
        return StreamChunk{ .content = {}, .reasoning_content = std::move(text), .tools = {}, .is_final = false, .is_error = false };
    }
};

struct Message {
    std::string role = {};
    std::string content = {};
    std::string name = {};                 // Optional, for tool role
    std::string tool_call_id = {};         // Optional, for tool role
    std::vector<ToolCall> tool_calls = {}; // Optional, for assistant role
    std::string reasoning_content = {};    // Optional, for Kimi thinking mode
    std::vector<ContentPart> content_parts = {}; // Optional, for multimodal user input
};

struct Tool {
    std::string type = "function";
    core::tools::ToolDefinition function;
};

/**
 * @brief Response format configuration for structured outputs (JSON mode).
 */
struct ResponseFormat {
    enum class Type { Text, JsonObject, JsonSchema };
    
    Type type = Type::Text;
    std::string schema;  // JSON schema when type is JsonSchema
    
    [[nodiscard]] std::string to_string() const noexcept {
        switch (type) {
            case Type::JsonObject: return "json_object";
            case Type::JsonSchema: return "json_schema";
            default: return "text";
        }
    }
    
    [[nodiscard]] bool is_structured() const noexcept {
        return type == Type::JsonObject || type == Type::JsonSchema;
    }
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
};

[[nodiscard]] inline bool message_has_image_input(const Message& msg) noexcept {
    return message_has_image_input(msg.content_parts);
}

[[nodiscard]] inline bool request_has_image_input(const ChatRequest& req) noexcept {
    for (const auto& msg : req.messages) {
        if (message_has_image_input(msg)) {
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

inline void degrade_historical_image_inputs(ChatRequest& req) {
    bool kept_latest_user = false;
    for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
        if (it->role == "user" && !kept_latest_user) {
            kept_latest_user = true;
            continue;
        }

        if (message_has_image_input(*it)) {
            degrade_message_to_text_only(*it);
        }
    }
}

struct Serializer {
    struct Options {
        // Vendor extension: Kimi thinking models require assistant.reasoning_content
        // to be echoed back on follow-up tool turns. Disabled by default so the
        // base OpenAI serializer remains spec-clean.
        bool include_reasoning_content = false;
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
            payload += R"(,"max_tokens":)";
            char tmp[32];
            auto [ptr, _] = std::to_chars(tmp, tmp + sizeof(tmp), req.max_tokens.value());
            payload.append(tmp, ptr);
        }
        
        // Response format (JSON mode / structured outputs)
        if (req.response_format.is_structured()) {
            payload += R"(,"response_format":{"type":")" + req.response_format.to_string() + '"';
            if (req.response_format.type == ResponseFormat::Type::JsonSchema && !req.response_format.schema.empty()) {
                payload += R"(,"schema":)" + req.response_format.schema;
            }
            payload += "}";
        }

        if (!req.tools.empty()) {
            payload += R"(,"tools":[)";
            for (size_t i = 0; i < req.tools.size(); ++i) {
                const auto& def = req.tools[i].function;
                payload += R"({"type":")" + req.tools[i].type + R"(","function":{"name":")" + core::utils::escape_json_string(def.name) + R"(","description":")" + core::utils::escape_json_string(def.description) + R"(","parameters":{"type":"object","properties":{)";
                
                for (size_t j = 0; j < def.parameters.size(); ++j) {
                    const auto& p = def.parameters[j];
                    payload += "\"" + core::utils::escape_json_string(p.name) + R"(":{"type":")" + core::utils::escape_json_string(p.type) + R"(","description":")" + core::utils::escape_json_string(p.description) + "\"";
                    if (!p.items_schema.empty()) {
                        payload += R"(,"items":)" + p.items_schema;
                    }
                    payload += "}";
                    if (j < def.parameters.size() - 1) payload += ",";
                }
                
                payload += R"(},"required":[)";
                bool first_req = true;
                for (const auto& p : def.parameters) {
                    if (p.required) {
                        if (!first_req) payload += ",";
                        payload += "\"" + core::utils::escape_json_string(p.name) + "\"";
                        first_req = false;
                    }
                }
                payload += "]}}}";
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

                    if (const auto encoded = encode_image_part(part); encoded.has_value()) {
                        if (!first_part) payload += ",";
                        payload += R"({"type":"image_url","image_url":{"url":")";
                        payload += core::utils::escape_json_string(encoded->data_url());
                        payload += R"(","detail":")";
                        payload += core::utils::escape_json_string(encoded->detail);
                        payload += R"("}})";
                        first_part = false;
                    } else {
                        if (!first_part) payload += ",";
                        payload += R"({"type":"text","text":")";
                        payload += core::utils::escape_json_string(
                            unavailable_image_attachment_text(part.path));
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

            // Vendor extension (Kimi): only emitted when explicitly enabled.
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
