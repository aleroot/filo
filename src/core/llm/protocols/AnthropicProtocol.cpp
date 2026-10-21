#include "AnthropicProtocol.hpp"
#include "SseUtils.hpp"
#include "../Models.hpp"
#include "../ModelRegistry.hpp"
#include "../../logging/Logger.hpp"
#include "../../utils/JsonUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include "../../utils/AsciiUtils.hpp"
#include "../../tools/StrictToolSchema.hpp"
#include "../../tools/ToolSchema.hpp"
#include <simdjson.h>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <optional>
#include <utility>

namespace core::llm::protocols {

namespace {
    [[nodiscard]] std::string infer_anthropic_event_type(std::string_view payload) {
        thread_local simdjson::dom::parser parser;
        const core::utils::json::ParserRetentionGuard parser_guard{parser};
        simdjson::padded_string padded(payload);
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return {};

        std::string_view type;
        if (doc["type"].get(type) != simdjson::SUCCESS) return {};
        return std::string(type);
    }

    [[nodiscard]] bool is_retryable_anthropic_stream_error(std::string_view type) {
        return type == "overloaded_error"
            || type == "rate_limit_error"
            || type == "api_error"
            || type == "timeout_error";
    }

    [[nodiscard]] int model_default_max_tokens(std::string_view model,
                                                int fallback) {
        const auto info = ModelRegistry::instance().lookup(model);
        if (!info) return fallback;
        const int32_t max_tokens = info->effective_max_tokens();
        return max_tokens > 0 ? max_tokens : fallback;
    }

void append_system_field(std::string& payload, const ChatRequest& req,
                         const AnthropicSerializationOptions& options) {
    std::vector<std::string_view> layers;
    std::optional<std::size_t> cache_boundary;
    if (!options.system_prefix.empty()) layers.push_back(options.system_prefix);
    if (req.prompt_plan.empty()) {
        for (const auto& msg : req.messages) {
            if (msg.role == "system" && !msg.content.empty()) {
                layers.push_back(msg.content);
            }
        }
    } else {
        for (const auto& layer : req.prompt_plan.layers()) {
            layers.push_back(layer.content);
            if (layer.stability <= core::context::PromptStability::Workspace) cache_boundary = layers.size() - 1;
        }
    }
    if (layers.empty()) return;
    if (req.prompt_plan.empty() && options.system_as_text) {
        std::string text;
        for (const auto layer : layers) { if (!text.empty()) text += "\n\n"; text += layer; }
        payload += R"(,"system":")" + core::utils::escape_json_string(text) + '"';
        return;
    }
    payload += R"(,"system":[)";
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (i) payload += ',';
        payload += R"({"type":"text","text":")";
        payload += core::utils::escape_json_string(layers[i]);
        payload += '"';
        if (options.cache_control && cache_boundary == i) payload += R"(,"cache_control":{"type":"ephemeral"})";
        payload += '}';
    }
    payload += ']';
}

}

namespace {

// Messages API image sources accept image/jpeg, image/png, image/gif, and
// image/webp. `source.data` is raw base64, never a data: URL.
[[nodiscard]] std::optional<std::string> anthropic_image_media_type(
    std::string_view mime) {
    std::string normalized;
    normalized.reserve(mime.size());
    for (const unsigned char ch : mime) {
        if (ch == ' ' || ch == '\t') continue;
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    if (normalized == "image/jpg" || normalized == "image/pjpeg") {
        return std::string("image/jpeg");
    }
    if (normalized == "image/x-png") {
        return std::string("image/png");
    }
    if (normalized == "image/jpeg" || normalized == "image/png"
        || normalized == "image/gif" || normalized == "image/webp") {
        return normalized;
    }
    return std::nullopt;
}

// Replay native block order only while the visible transcript still matches it.
// Session edits fall back to the normalized representation and binding controls.
[[nodiscard]] std::optional<std::string> native_assistant_content(
    const Message& message, bool omit_thinking) {
    for (const auto& item : message.continuation_items) {
        if (item.provider != "anthropic" || item.kind != "assistant_content") continue;
        simdjson::dom::parser parser;
        simdjson::padded_string payload(item.payload);
        simdjson::dom::element doc;
        simdjson::dom::array blocks;
        if (parser.parse(payload).get(doc) != simdjson::SUCCESS
            || doc["content"].get(blocks) != simdjson::SUCCESS) continue;
        std::string text;
        std::string content = "[";
        std::size_t tool_index = 0;
        bool matches = true;
        for (const auto block : blocks) {
            std::string_view type;
            if (block["type"].get(type) != simdjson::SUCCESS) { matches = false; break; }
            if (type == "text") {
                std::string_view value;
                if (block["text"].get(value) != simdjson::SUCCESS) { matches = false; break; }
                text += value;
            } else if (type == "tool_use") {
                if (tool_index >= message.tool_calls.size()) { matches = false; break; }
                const auto& tool = message.tool_calls[tool_index++];
                std::string_view id, name;
                simdjson::dom::object input;
                simdjson::dom::parser arguments_parser;
                simdjson::padded_string arguments(tool.function.arguments.empty() ? "{}" : tool.function.arguments);
                simdjson::dom::object expected;
                if (block["id"].get(id) != simdjson::SUCCESS || id != tool.id
                    || block["name"].get(name) != simdjson::SUCCESS || name != tool.function.name
                    || block["input"].get(input) != simdjson::SUCCESS
                    || arguments_parser.parse(arguments).get_object().get(expected) != simdjson::SUCCESS
                    || simdjson::to_string(input) != simdjson::to_string(expected)) { matches = false; break; }
            } else if (type != "thinking" && type != "redacted_thinking") {
                matches = false; break;
            }
            if (omit_thinking && (type == "thinking" || type == "redacted_thinking")) continue;
            if (content.back() != '[') content += ',';
            content += simdjson::to_string(block);
        }
        if (matches && text == message.content && tool_index == message.tool_calls.size()
            && content.size() > 1) return content + "]";
    }
    return std::nullopt;
}

} // namespace

std::string AnthropicSerializer::serialize(
    const ChatRequest& req,
    int default_max_tokens,
    const AnthropicSerializationOptions& options) {
    std::string payload;
    payload.reserve(estimated_request_bytes(req));

    payload += R"({"model":")";
    payload += core::utils::escape_json_string(req.model);
    payload += R"(","stream":)";
    payload += req.stream ? "true" : "false";

    // max_tokens is mandatory in the Anthropic API.
    payload += R"(,"max_tokens":)";
    payload += std::to_string(req.max_tokens.value_or(
        AnthropicSerializer::max_tokens(req, default_max_tokens)));

    // Automatic prompt caching materially reduces repeated-prefix cost for
    // tool-heavy multi-step agent loops while remaining transparent to callers.
    const bool should_enable_auto_cache =
        options.cache_control && (!req.tools.empty() || req.messages.size() > 2);
    if (should_enable_auto_cache) {
        payload += R"(,"cache_control":{"type":"ephemeral"})";
    }

    if (options.extra_fields) options.extra_fields(payload, req);
    else if (req.temperature) {
        payload += R"(,"temperature":)" + std::to_string(*req.temperature);
    }
    append_system_field(payload, req, options);
    const bool omit_historical_thinking = options.omit_historical_thinking;

    // Tools (Anthropic uses "input_schema" instead of "parameters").
    if (!req.tools.empty()) {
        // Strict tool use constrains the sampler to the declared shape, so a
        // mis-typed argument cannot be generated in the first place. A tool
        // whose contract does not survive the projection is sent unflagged.
        const auto strict_dialect = options.tool_schema_dialect;
        payload += R"(,"tools":[)";
        for (size_t i = 0; i < req.tools.size(); ++i) {
            const auto& def = req.tools[i].function;
            payload += R"({"name":")";
            payload += core::utils::escape_json_string(def.name);
            payload += R"(","description":")";
            payload += core::utils::escape_json_string(def.description);
            payload += R"(","input_schema":)";
            const std::string canonical_schema =
                core::tools::schema::canonical_input_schema(def);
            const auto strict_schema = strict_dialect.has_value()
                ? core::tools::schema::strict_input_schema(
                      canonical_schema, *strict_dialect)
                : std::nullopt;
            const bool use_strict = strict_schema.has_value() && (!options.admit_strict_tool || options.admit_strict_tool(*strict_schema));
            payload += use_strict ? *strict_schema : canonical_schema;
            if (use_strict) {
                payload += R"(,"strict":true)";
            }
            if (options.cache_control && i + 1 == req.tools.size()) {
                payload += R"(,"cache_control":{"type":"ephemeral"})";
            }
            payload += '}';
            if (i + 1 < req.tools.size()) payload += ',';
        }
        payload += ']';
    }

    // Messages — system messages are skipped (handled above as top-level field).
    payload += R"(,"messages":[)";
    bool first_msg = true;
    for (std::size_t message_index = 0;
         message_index < req.messages.size();
         ++message_index) {
        const auto& msg = req.messages[message_index];
        if (msg.role == "system") continue;
        if (omit_historical_thinking && msg.role == "assistant"
            && msg.content.empty() && msg.tool_calls.empty() && msg.content_parts.empty()) continue;

        if (!first_msg) payload += ',';
        first_msg = false;

        if (msg.role == "tool") {
            // Every result for a parallel assistant tool-use step belongs in
            // the immediately following Messages user message. Filo stores one
            // provider-neutral `tool` message per result, so coalesce the
            // consecutive run into one Anthropic content array.
            payload += R"({"role":"user","content":[)";
            bool first_result = true;
            while (message_index < req.messages.size()
                   && req.messages[message_index].role == "tool") {
                const auto& result = req.messages[message_index];
                if (!first_result) payload += ',';
                first_result = false;
                payload += R"({"type":"tool_result","tool_use_id":")";
                payload += core::utils::escape_json_string(result.tool_call_id);
                payload += R"(","content":")";
                payload += core::utils::escape_json_string(result.content);
                payload += "\"}";
                ++message_index;
            }
            --message_index;
            payload += "]}";

        } else if (msg.role == "assistant"
                   && (!msg.tool_calls.empty() || !msg.continuation_items.empty())) {
            if (const auto content = native_assistant_content(msg, omit_historical_thinking)) {
                payload += R"({"role":"assistant","content":)";
                payload += *content;
                payload += '}';
                continue;
            }
            // Assistant tool_calls → Messages content blocks.
            payload += R"({"role":"assistant","content":[)";
            bool first_block = true;

            for (const auto& continuation : msg.continuation_items) {
                if (continuation.kind == "assistant_content") continue;
                // These low-thinking modes reject block_binding. Remove historical
                // reasoning as documented so edited prefixes remain resumable.
                if (omit_historical_thinking
                    && (continuation.kind == "thinking" || continuation.kind == "redacted_thinking")) continue;
                if ((!continuation.provider.empty()
                     && continuation.provider != "anthropic")
                    || !has_valid_continuation_payload(continuation)) {
                    continue;
                }
                if (!first_block) payload += ',';
                payload += continuation.payload;
                first_block = false;
            }

            if (!msg.content.empty()) {
                if (!first_block) payload += ',';
                payload += R"({"type":"text","text":")";
                payload += core::utils::escape_json_string(msg.content);
                payload += "\"}";
                first_block = false;
            }

            for (const auto& tc : msg.tool_calls) {
                if (!first_block) payload += ',';
                first_block = false;
                payload += R"({"type":"tool_use","id":")";
                payload += core::utils::escape_json_string(tc.id);
                payload += R"(","name":")";
                payload += core::utils::escape_json_string(tc.function.name);
                // Only validated JSON objects may be embedded as native JSON.
                payload += R"(","input":)";
                payload += core::utils::json::object_or_empty(
                    tc.function.arguments);
                payload += '}';
            }
            payload += "]}";

        } else {
            payload += R"({"role":")";
            payload += core::utils::escape_json_string(msg.role);
            if (!msg.content_parts.empty()) {
                payload += R"(","content":[)";
                bool first_block = true;
                for (const auto& part : msg.content_parts) {
                    if (!first_block) payload += ',';
                    first_block = false;

                    if (part.type == ContentPartType::Text) {
                        payload += R"({"type":"text","text":")";
                        payload += core::utils::escape_json_string(part.text);
                        payload += R"("})";
                        continue;
                    }

                    const auto encoded = encode_image_part(part);
                    const auto media_type = encoded.has_value() && !encoded->is_url_reference()
                        ? anthropic_image_media_type(encoded->mime_type)
                        : std::nullopt;
                    if (options.image_urls && encoded && encoded->is_url_reference()) {
                        payload += R"({"type":"image","source":{"type":"url","url":")";
                        payload += core::utils::escape_json_string(encoded->url);
                        payload += R"("}})";
                    } else if (media_type.has_value()) {
                        payload += R"({"type":"image","source":{"type":"base64","media_type":")";
                        payload += core::utils::escape_json_string(*media_type);
                        payload += R"(","data":")";
                        payload += core::utils::escape_json_string(encoded->base64_data);
                        payload += R"("}})";
                    } else {
                        payload += R"({"type":"text","text":")";
                        payload += core::utils::escape_json_string(
                            unavailable_media_attachment_text(part.type, media_reference(part)));
                        payload += R"("})";
                    }
                }
                payload += "]}";
            } else {
                payload += R"(","content":")";
                payload += core::utils::escape_json_string(msg.content);
                payload += "\"}";
            }
        }
    }
    payload += "]}";
    return payload;
}

AnthropicSSEParser::Result AnthropicSSEParser::process_event(std::string_view event_type,
                                                              std::string_view json_str) {
    Result result;
    if (json_str.empty()) return result;
    if (event_type == "ping") return result;

    thread_local simdjson::dom::parser parser;
    const core::utils::json::ParserRetentionGuard parser_guard{parser};
    simdjson::padded_string input(json_str);
    simdjson::dom::element doc;
    if (parser.parse(input).get(doc) != simdjson::SUCCESS) {
        if (event_type == "message_start" || event_type == "message_delta"
            || event_type == "message_stop" || event_type == "content_block_start"
            || event_type == "content_block_delta" || event_type == "content_block_stop"
            || event_type == "error") {
            result.stream_error = true;
            result.retryable_stream_error = true;
            result.error_type = "invalid_stream_event";
            result.error_message = "Messages endpoint returned malformed streaming JSON";
        }
        return result;
    }

    const auto read_usage = [&](simdjson::dom::object usage) {
        const auto read_input = [&](std::string_view key, int64_t& value) {
            int64_t reported = 0;
            if (usage[key].get(reported) == simdjson::SUCCESS) {
                value = std::clamp<int64_t>(reported, 0, std::numeric_limits<int32_t>::max());
                result.input_usage_reported = true;
            }
        };
        read_input("input_tokens", uncached_input_);
        read_input("cache_read_input_tokens", cached_input_);
        read_input("cache_creation_input_tokens", cache_creation_input_);
        if (result.input_usage_reported) {
            result.input_tokens = static_cast<int32_t>(std::min<int64_t>(
                uncached_input_ + cached_input_ + cache_creation_input_, std::numeric_limits<int32_t>::max()));
            result.cached_input_tokens = static_cast<int32_t>(cached_input_);
            result.cache_creation_input_tokens = static_cast<int32_t>(cache_creation_input_);
        }
        int64_t output = 0;
        if (usage["output_tokens"].get(output) == simdjson::SUCCESS) {
            result.output_tokens = static_cast<int32_t>(std::clamp<int64_t>(output, 0, std::numeric_limits<int32_t>::max()));
        }
    };

    if (event_type == "error") {
        result.stream_error = true;
        simdjson::dom::object error_obj;
        if (doc["error"].get(error_obj) == simdjson::SUCCESS) {
            std::string_view type;
            if (error_obj["type"].get(type) == simdjson::SUCCESS) {
                result.error_type = std::string(type);
                result.retryable_stream_error =
                    is_retryable_anthropic_stream_error(type);
            }
            std::string_view message;
            if (error_obj["message"].get(message) == simdjson::SUCCESS) {
                result.error_message = std::string(message);
            }
        }
        return result;
    }

    if (event_type == "message_start") {
        result.stream_started = true;
        simdjson::dom::array transformations;
        if (doc["input_transformations"].get(transformations) == simdjson::SUCCESS) {
            for (const auto item : transformations) {
                std::string_view type;
                std::string_view reason;
                if (item["type"].get(type) != simdjson::SUCCESS
                    || type != "thinking_dropped"
                    || item["reason"].get(reason) != simdjson::SUCCESS) continue;
                if (reason == "prefix_binding_mismatch") ++result.prefix_binding_mismatches;
                if (reason == "model_binding_mismatch") ++result.model_binding_mismatches;
            }
        }
        simdjson::dom::object msg_obj;
        if (doc["message"].get(msg_obj) == simdjson::SUCCESS) {
            simdjson::dom::object usage_obj;
            if (msg_obj["usage"].get(usage_obj) == simdjson::SUCCESS) {
                read_usage(usage_obj);
            }
        }
        return result;
    }

    if (event_type == "message_delta") {
        simdjson::dom::object delta_obj;
        if (doc["delta"].get(delta_obj) == simdjson::SUCCESS) {
            std::string_view stop_reason;
            if (delta_obj["stop_reason"].get(stop_reason) == simdjson::SUCCESS) {
                result.stop_reason = std::string(stop_reason);
                // Classifier refusals are HTTP 200 with an empty content array.
                // Surface the API's explanation so the turn is not a blank stop.
                if (stop_reason == "refusal") {
                    simdjson::dom::object details;
                    if (delta_obj["stop_details"].get(details) == simdjson::SUCCESS) {
                        std::string_view explanation;
                        std::string_view category;
                        if (details["explanation"].get(explanation) == simdjson::SUCCESS
                            && !explanation.empty()) {
                            result.text = std::string(explanation);
                        } else if (details["category"].get(category) == simdjson::SUCCESS
                                   && !category.empty()) {
                            result.text = "Request declined (";
                            result.text.append(category);
                            result.text += ").";
                        }
                    }
                }
            }
        }

        simdjson::dom::object usage_obj;
        if (doc["usage"].get(usage_obj) == simdjson::SUCCESS) {
            read_usage(usage_obj);
        }
        return result;
    }

    if (event_type == "content_block_start") {
        simdjson::dom::object content_block;
        if (doc["content_block"].get(content_block) != simdjson::SUCCESS) return result;

        std::string_view type_v;
        if (content_block["type"].get(type_v) != simdjson::SUCCESS) return result;

        if (type_v == "tool_use") {
            AnthropicToolBlockState state;
            int64_t index_v;
            if (doc["index"].get(index_v) == simdjson::SUCCESS)
                state.index = static_cast<int>(index_v);
            std::string_view id_v;
            if (content_block["id"].get(id_v) == simdjson::SUCCESS)
                state.id = std::string(id_v);
            std::string_view name_v;
            if (content_block["name"].get(name_v) == simdjson::SUCCESS)
                state.name = std::string(name_v);
            simdjson::dom::object initial_input;
            if (content_block["input"].get(initial_input) == simdjson::SUCCESS) {
                state.initial_args = simdjson::to_string(initial_input);
            }
            current_tool_ = std::move(state);
        } else if (type_v == "thinking" || type_v == "redacted_thinking") {
            AnthropicContinuationBlockState state;
            int64_t index = -1;
            if (doc["index"].get(index) == simdjson::SUCCESS) state.index = static_cast<int>(index);
            state.type = std::string(type_v);
            state.initial_payload = simdjson::to_string(content_block);
            std::string_view value;
            if (content_block["thinking"].get(value) == simdjson::SUCCESS) {
                state.thinking = std::string(value);
            }
            if (content_block["signature"].get(value) == simdjson::SUCCESS) {
                state.signature = std::string(value);
            }
            current_continuation_ = std::move(state);
        } else if (type_v == "text") {
            int64_t index = -1;
            if (doc["index"].get(index) == simdjson::SUCCESS) text_index_ = static_cast<int>(index);
            std::string_view text;
            current_text_ = content_block["text"].get(text) == simdjson::SUCCESS ? std::string(text) : "";
            result.text = *current_text_;
        } else {
            replay_complete_ = false;
        }
        return result;
    }

    if (event_type == "content_block_delta") {
        simdjson::dom::object delta;
        if (doc["delta"].get(delta) != simdjson::SUCCESS) return result;

        std::string_view delta_type;
        if (delta["type"].get(delta_type) != simdjson::SUCCESS) return result;

        int64_t index = -1;
        const bool has_index = doc["index"].get(index) == simdjson::SUCCESS;
        if (delta_type == "text_delta" && (!current_text_ || !has_index || index == text_index_)) {
            std::string_view text;
            if (delta["text"].get(text) == simdjson::SUCCESS) {
                result.text = std::string(text);
                if (current_text_) *current_text_ += text;
            }
        } else if (delta_type == "input_json_delta" && current_tool_.has_value()
                   && (!has_index || index == current_tool_->index)) {
            std::string_view partial_json;
            if (delta["partial_json"].get(partial_json) == simdjson::SUCCESS)
                current_tool_->accumulated_args += partial_json;
        } else if (delta_type == "thinking_delta" && current_continuation_.has_value()
                   && (!has_index || index == current_continuation_->index)) {
            std::string_view thinking;
            if (delta["thinking"].get(thinking) == simdjson::SUCCESS) {
                // Authoritative, signed thinking accumulated for upstream replay.
                current_continuation_->thinking += thinking;
                // Display-only mirror surfaced to the UI reasoning channel. This
                // is a copy: the signed block above remains the source of truth
                // for API continuation and is never derived from this field.
                result.reasoning_delta.append(thinking.data(), thinking.size());
            }
        } else if (delta_type == "signature_delta" && current_continuation_.has_value()
                   && (!has_index || index == current_continuation_->index)) {
            std::string_view signature;
            if (delta["signature"].get(signature) == simdjson::SUCCESS) {
                current_continuation_->signature += signature;
            }
        }
        return result;
    }

    if (event_type == "content_block_stop") {
        int64_t index = -1;
        const bool has_index = doc["index"].get(index) == simdjson::SUCCESS;
        if (current_text_ && (!has_index || index == text_index_)) {
            replay_blocks_.push_back(R"({"type":"text","text":")"
                + core::utils::escape_json_string(*current_text_) + R"("})");
            current_text_.reset();
        }
        if (current_tool_.has_value() && (!has_index || index == current_tool_->index)) {
            const std::string args = current_tool_->accumulated_args.empty()
                ? (current_tool_->initial_args.empty() ? "{}" : current_tool_->initial_args)
                : current_tool_->accumulated_args;
            simdjson::dom::parser arguments_parser;
            simdjson::dom::object arguments;
            simdjson::padded_string padded_args(args);
            if (arguments_parser.parse(padded_args).get_object().get(arguments) != simdjson::SUCCESS) {
                result.stream_error = true;
                result.retryable_stream_error = true;
                result.error_type = "invalid_tool_input";
                result.error_message = "Messages endpoint returned incomplete or malformed tool arguments";
                current_tool_.reset();
                return result;
            }
            replay_blocks_.push_back(R"({"type":"tool_use","id":")"
                + core::utils::escape_json_string(current_tool_->id) + R"(","name":")"
                + core::utils::escape_json_string(current_tool_->name) + R"(","input":)" + args + "}");
            ToolCall tc;
            tc.index              = current_tool_->index;
            tc.id                 = std::move(current_tool_->id);
            tc.type               = "function";
            tc.function.name      = std::move(current_tool_->name);
            tc.function.arguments = args;
            result.completed_tools.push_back(std::move(tc));
            current_tool_.reset();
        }
        if (current_continuation_.has_value() && (!has_index || index == current_continuation_->index)) {
            std::string block;
            if (current_continuation_->type == "redacted_thinking") {
                block = std::move(current_continuation_->initial_payload);
            } else {
                block = R"({"type":"thinking","thinking":")";
                block += core::utils::escape_json_string(current_continuation_->thinking);
                block += R"(","signature":")";
                block += core::utils::escape_json_string(current_continuation_->signature);
                block += R"("})";
            }
            replay_blocks_.push_back(block);
            result.continuation_items.push_back(ContinuationItem{
                .provider = "anthropic",
                .kind = current_continuation_->type,
                .payload = std::move(block),
            });
            current_continuation_.reset();
        }
        return result;
    }

    if (event_type == "message_stop") {
        result.done = true;
        result.incomplete_tool_call = current_tool_.has_value();
        // A terminal max_tokens stop can legitimately interrupt a block. The
        // agent already recovers such turns and never executes unfinished tools.
        if (!current_tool_ && !current_continuation_ && !current_text_
            && replay_complete_ && !replay_blocks_.empty()) {
            std::string content = R"({"content":[)";
            for (const auto& block : replay_blocks_) {
                if (content.back() != '[') content += ',';
                content += block;
            }
            content += "]}";
            result.continuation_items.push_back({.provider = "anthropic", .kind = "assistant_content", .payload = std::move(content)});
        }
        current_tool_.reset();
        current_continuation_.reset();
        current_text_.reset();
        replay_blocks_.clear();
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicProtocol
// ─────────────────────────────────────────────────────────────────────────────

std::string AnthropicProtocol::serialize(const ChatRequest& req) const {
    AnthropicSerializationOptions options;
    options.extra_fields = reasoning_emitter();
    return AnthropicSerializer::serialize(req, default_max_tokens_, options);
}
AnthropicReasoningEmitter AnthropicProtocol::reasoning_emitter() const { return {}; }
int AnthropicSerializer::max_tokens(const ChatRequest& req, int fallback) {
    return req.max_tokens.value_or(model_default_max_tokens(req.model, fallback));
}

cpr::Header AnthropicProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{{"anthropic-version", "2023-06-01"},
                        {"Content-Type", "application/json"}, {"Accept", "text/event-stream"}};
    for (const auto& [key, value] : auth.headers) headers[key] = value;
    return headers;
}

std::string AnthropicProtocol::build_url(std::string_view base_url,
                                          [[maybe_unused]] std::string_view model) const {
    while (base_url.ends_with('/')) base_url.remove_suffix(1);
    if (base_url.ends_with("/v1/messages")) return std::string(base_url);
    if (base_url.ends_with("/v1")) return std::string(base_url) + "/messages";
    return std::string(base_url) + "/v1/messages";
}

ParseResult AnthropicProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    if (parsed.is_done || parsed.data.empty()) return {};

    const std::string event_data = std::string(parsed.data);
    std::string event_type = std::string(parsed.event);
    if (event_type.empty()) {
        event_type = infer_anthropic_event_type(event_data);
    }
    if (event_type.empty()) return {};

    auto r = sse_parser_.process_event(event_type, event_data);

    ParseResult result;
    result.stream_started = r.stream_started;
    result.stream_error = r.stream_error;
    result.retryable_stream_error = r.retryable_stream_error;
    result.stream_error_type = std::move(r.error_type);
    result.stream_error_message = std::move(r.error_message);
    if (r.input_usage_reported) {
        accumulated_input_ = r.input_tokens;
        accumulated_cached_input_ = r.cached_input_tokens;
        accumulated_cache_creation_ = r.cache_creation_input_tokens;
    }
    if (r.output_tokens > 0) accumulated_output_ = r.output_tokens;
    if (!r.stop_reason.empty()) last_stop_reason_ = r.stop_reason;
    if (r.prefix_binding_mismatches > 0 || r.model_binding_mismatches > 0) {
        core::logging::debug(
            "[Anthropic] Dropped thinking blocks: {} after context edits, {} after model switches",
            r.prefix_binding_mismatches, r.model_binding_mismatches);
    }

    if (!r.text.empty())           result.chunks.push_back(StreamChunk::make_content(r.text));
    if (!r.reasoning_delta.empty()) {
        auto chunk = StreamChunk::make_reasoning(std::move(r.reasoning_delta));
        chunk.reasoning_protocol = std::string(name());
        result.chunks.push_back(std::move(chunk));
    }
    if (!r.completed_tools.empty()) result.chunks.push_back(StreamChunk::make_tools(r.completed_tools));
    if (!preserves_native_assistant_content()) {
        std::erase_if(r.continuation_items, [](const auto& item) { return item.kind == "assistant_content"; });
    }
    if (!r.continuation_items.empty()) {
        StreamChunk chunk;
        chunk.continuation_items = std::move(r.continuation_items);
        result.chunks.push_back(std::move(chunk));
    }

    if (r.done) {
        result.done              = true;
        result.prompt_tokens     = accumulated_input_;
        result.completion_tokens = accumulated_output_;
        result.cached_prompt_tokens = accumulated_cached_input_;
        result.cache_creation_prompt_tokens = accumulated_cache_creation_;
        result.stop_reason       = last_stop_reason_;
        result.incomplete_tool_call = r.incomplete_tool_call;
    }

    return result;
}

std::unique_ptr<ApiProtocolBase> AnthropicProtocol::clone() const {
    return std::make_unique<AnthropicProtocol>(default_max_tokens_);
}

void AnthropicProtocol::reset_state() {
    sse_parser_ = AnthropicSSEParser{};
    accumulated_input_ = 0;
    accumulated_output_ = 0;
    accumulated_cached_input_ = 0;
    accumulated_cache_creation_ = 0;
    last_stop_reason_.clear();
}


} // namespace core::llm::protocols
