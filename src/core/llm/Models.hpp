#pragma once

#include <string>
#include <vector>
#include <optional>
#include <sstream>
#include <iomanip>
#include "../tools/Tool.hpp"
#include "../utils/JsonUtils.hpp"

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
};

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
            if (!req.messages[i].content.empty()) {
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
