#include "GeminiProtocol.hpp"
#include "SseUtils.hpp"
#include "../Models.hpp"
#include "../../utils/JsonUtils.hpp"
#include <atomic>
#include <cctype>
#include <charconv>
#include <simdjson.h>

namespace core::llm::protocols {

namespace {

std::atomic_uint64_t g_gemini_call_counter{0};

[[nodiscard]] bool gemini_model_requires_property_ordering(std::string_view model) {
    return model.starts_with("gemini-2.0");
}

[[nodiscard]] std::string trim_ascii_copy(std::string_view input) {
    std::size_t begin = 0;
    while (begin < input.size()
        && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }

    std::size_t end = input.size();
    while (end > begin
        && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return std::string(input.substr(begin, end - begin));
}

[[nodiscard]] std::string lower_ascii_copy(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (const unsigned char ch : input) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

[[nodiscard]] std::string_view stable_gemini_pro_model() {
    return "gemini-2.5-pro";
}

[[nodiscard]] std::string_view stable_gemini_flash_model() {
    return "gemini-2.5-flash";
}

[[nodiscard]] std::string_view stable_gemini_flash_lite_model() {
    return "gemini-2.5-flash-lite";
}

[[nodiscard]] std::string_view preview_gemini_pro_model() {
    return "gemini-3.1-pro-preview";
}

[[nodiscard]] std::string_view preview_gemini_flash_model() {
    return "gemini-3-flash-preview";
}

[[nodiscard]] std::string_view preview_gemini_flash_lite_model() {
    return "gemini-3.1-flash-lite-preview";
}

[[nodiscard]] std::string format_json_double(double value) {
    char buffer[64];
    const auto [ptr, ec] = std::to_chars(
        buffer,
        buffer + sizeof(buffer),
        value,
        std::chars_format::general);
    if (ec == std::errc{}) {
        return std::string(buffer, ptr);
    }
    return std::to_string(value);
}

void append_json_escaped_string(std::string& out, std::string_view value) {
    out.push_back('"');
    out += core::utils::escape_json_string(value);
    out.push_back('"');
}

void append_schema_json(std::string& out,
                        simdjson::dom::element element,
                        bool inject_property_ordering);

void append_schema_object(std::string& out,
                          simdjson::dom::object object,
                          bool inject_property_ordering) {
    out.push_back('{');
    bool first = true;
    bool has_property_ordering = false;
    std::vector<std::string> property_order;

    simdjson::dom::object properties_obj;
    const bool has_properties =
        object["properties"].get(properties_obj) == simdjson::SUCCESS;
    if (inject_property_ordering && has_properties) {
        for (auto property : properties_obj) {
            property_order.push_back(std::string(property.key));
        }
    }

    for (auto field : object) {
        const std::string_view key = field.key;
        if (!first) out.push_back(',');
        first = false;

        append_json_escaped_string(out, key);
        out.push_back(':');

        if (key == "propertyOrdering") {
            has_property_ordering = true;
        }

        if (key == "properties") {
            simdjson::dom::object props;
            if (field.value.get(props) == simdjson::SUCCESS) {
                out.push_back('{');
                bool first_prop = true;
                for (auto prop : props) {
                    if (!first_prop) out.push_back(',');
                    first_prop = false;
                    append_json_escaped_string(out, prop.key);
                    out.push_back(':');
                    append_schema_json(out, prop.value, inject_property_ordering);
                }
                out.push_back('}');
                continue;
            }
        }

        append_schema_json(out, field.value, inject_property_ordering);
    }

    if (inject_property_ordering && has_properties && !has_property_ordering) {
        if (!first) out.push_back(',');
        append_json_escaped_string(out, "propertyOrdering");
        out.push_back(':');
        out.push_back('[');
        for (std::size_t i = 0; i < property_order.size(); ++i) {
            if (i > 0) out.push_back(',');
            append_json_escaped_string(out, property_order[i]);
        }
        out.push_back(']');
    }

    out.push_back('}');
}

void append_schema_array(std::string& out,
                         simdjson::dom::array array,
                         bool inject_property_ordering) {
    out.push_back('[');
    bool first = true;
    for (auto entry : array) {
        if (!first) out.push_back(',');
        first = false;
        append_schema_json(out, entry, inject_property_ordering);
    }
    out.push_back(']');
}

void append_schema_json(std::string& out,
                        simdjson::dom::element element,
                        bool inject_property_ordering) {
    if (element.is_null()) {
        out += "null";
        return;
    }

    std::string_view string_value;
    if (element.get(string_value) == simdjson::SUCCESS) {
        append_json_escaped_string(out, string_value);
        return;
    }

    bool bool_value = false;
    if (element.get(bool_value) == simdjson::SUCCESS) {
        out += bool_value ? "true" : "false";
        return;
    }

    int64_t int_value = 0;
    if (element.get(int_value) == simdjson::SUCCESS) {
        out += std::to_string(int_value);
        return;
    }

    uint64_t uint_value = 0;
    if (element.get(uint_value) == simdjson::SUCCESS) {
        out += std::to_string(uint_value);
        return;
    }

    double double_value = 0.0;
    if (element.get(double_value) == simdjson::SUCCESS) {
        out += format_json_double(double_value);
        return;
    }

    simdjson::dom::array array;
    if (element.get(array) == simdjson::SUCCESS) {
        append_schema_array(out, array, inject_property_ordering);
        return;
    }

    simdjson::dom::object object;
    if (element.get(object) == simdjson::SUCCESS) {
        append_schema_object(out, object, inject_property_ordering);
        return;
    }

    out += simdjson::minify(element);
}

[[nodiscard]] std::string serialize_gemini_response_schema(std::string_view raw_schema,
                                                           bool inject_property_ordering) {
    const std::string trimmed = trim_ascii_copy(raw_schema);
    if (trimmed.empty()) {
        return R"({"type":"object"})";
    }

    try {
        simdjson::dom::parser parser;
        simdjson::padded_string padded(trimmed.data(), trimmed.size());
        simdjson::dom::element doc = parser.parse(padded);
        std::string out;
        out.reserve(trimmed.size() + 64);
        append_schema_json(out, doc, inject_property_ordering);
        return out;
    } catch (...) {
        return trimmed;
    }
}

} // namespace

std::string normalize_requested_gemini_model(std::string_view raw_model) {
    std::string normalized = trim_ascii_copy(raw_model);
    if (normalized.empty()) {
        return normalized;
    }

    const std::string lowered = lower_ascii_copy(normalized);

    if (lowered == "auto-gemini-2.5") {
        return std::string(stable_gemini_pro_model());
    }
    if (lowered == "auto-gemini-3") {
        return std::string(preview_gemini_pro_model());
    }
    if (lowered == "gemini-pro-latest") {
        return std::string(stable_gemini_pro_model());
    }
    if (lowered == "gemini-flash-latest") {
        return std::string(stable_gemini_flash_model());
    }
    if (lowered == "gemini-flash-lite-latest") {
        return std::string(stable_gemini_flash_lite_model());
    }

    // Accept the current concrete model strings directly once alias handling
    // has had a chance to rewrite documented `*-latest` variants.
    if (lowered.starts_with("gemini-")) {
        return normalized;
    }

    // `auto` is reserved globally by Filo's router selection, but if a caller
    // explicitly routes to Gemini (for example `gemini/auto`) we still accept it.
    if (lowered == "auto" || lowered == "pro") {
        return std::string(preview_gemini_pro_model());
    }
    if (lowered == "flash") {
        return std::string(preview_gemini_flash_model());
    }
    if (lowered == "flash-lite") {
        return std::string(preview_gemini_flash_lite_model());
    }

    return normalized;
}

// ─────────────────────────────────────────────────────────────────────────────
// serialize_gemini_request
// ─────────────────────────────────────────────────────────────────────────────

std::string serialize_gemini_request(const ChatRequest& req,
                                      const std::string& default_model) {
    const std::string normalized_effective_model = normalize_requested_gemini_model(
        req.model.empty() ? std::string_view(default_model) : std::string_view(req.model));
    const std::string_view effective_model = normalized_effective_model;
    std::string payload;
    payload.reserve(8192);
    payload += "{";

    // System instruction (top-level, not a content turn).
    std::string system_prompt;
    std::vector<Message> filtered_messages;
    for (const auto& msg : req.messages) {
        if (msg.role == "system") {
            system_prompt += msg.content + "\n";
        } else {
            filtered_messages.push_back(msg);
        }
    }

    if (!system_prompt.empty()) {
        payload += R"("systemInstruction":{"parts":[{"text":")" +
                   core::utils::escape_json_string(system_prompt) +
                   R"("}]},)";
    }

    // Tools (Gemini uses functionDeclarations with uppercase type names).
    if (!req.tools.empty()) {
        payload += R"("tools":[{"functionDeclarations":[)";
        for (size_t i = 0; i < req.tools.size(); ++i) {
            const auto& def = req.tools[i].function;
            payload += R"({"name":")";
            payload += core::utils::escape_json_string(def.name);
            payload += R"(","description":")";
            payload += core::utils::escape_json_string(def.description);
            payload += R"(","parameters":{"type":"OBJECT","properties":{)";

            for (size_t j = 0; j < def.parameters.size(); ++j) {
                const auto& p = def.parameters[j];
                payload += '"';
                payload += core::utils::escape_json_string(p.name);
                payload += R"(":{"type":")";
                payload += core::utils::escape_json_string(p.type);
                payload += R"(","description":")";
                payload += core::utils::escape_json_string(p.description);
                payload += "\"}";
                if (j < def.parameters.size() - 1) payload += ',';
            }

            payload += R"(},"required":[)";
            bool first_req = true;
            for (const auto& p : def.parameters) {
                if (p.required) {
                    if (!first_req) payload += ',';
                    payload += '"';
                    payload += core::utils::escape_json_string(p.name);
                    payload += '"';
                    first_req = false;
                }
            }
            payload += "]}}";
            if (i < req.tools.size() - 1) payload += ',';
        }
        payload += "]}],";
    }

    // generationConfig (temperature, maxOutputTokens).
    payload += R"("generationConfig":{)";
    bool has_config = false;
    if (req.temperature.has_value()) {
        payload += R"("temperature":)" + std::to_string(req.temperature.value());
        has_config = true;
    }
    if (req.max_tokens.has_value()) {
        if (has_config) payload += ',';
        payload += R"("maxOutputTokens":)" + std::to_string(req.max_tokens.value());
        has_config = true;
    }
    if (req.response_format.is_structured()) {
        if (has_config) payload += ',';
        payload += R"("responseMimeType":"application/json")";
        has_config = true;
        if (req.response_format.type == ResponseFormat::Type::JsonSchema) {
            payload += R"(,"responseJsonSchema":)";
            payload += serialize_gemini_response_schema(
                req.response_format.schema,
                gemini_model_requires_property_ordering(effective_model));
        }
    }
    payload += "},";

    // Contents (the conversation).
    payload += R"("contents":[)";
    for (size_t i = 0; i < filtered_messages.size(); ++i) {
        const auto& msg = filtered_messages[i];
        std::string role = msg.role;
        if (role == "assistant") role = "model";
        if (role == "tool")      role = "function";

        payload += R"({"role":")";
        payload += core::utils::escape_json_string(role);
        payload += R"(","parts":[)";

        if (role == "function") {
            payload += R"({"functionResponse":{"name":")";
            payload += core::utils::escape_json_string(msg.name);
            payload += R"(","response":{"result":)";
            payload += msg.content;
            payload += "}}}";
        } else if (!msg.tool_calls.empty()) {
            for (size_t j = 0; j < msg.tool_calls.size(); ++j) {
                const auto& tc = msg.tool_calls[j];
                std::string args = tc.function.arguments.empty() ? "{}" : tc.function.arguments;
                payload += R"({"functionCall":{"name":")";
                payload += core::utils::escape_json_string(tc.function.name);
                payload += R"(","args":)";
                payload += args;
                payload += "}}";
                if (j < msg.tool_calls.size() - 1) payload += ',';
            }
        } else if (!msg.content_parts.empty()) {
            bool first_part = true;
            for (const auto& part : msg.content_parts) {
                if (part.type == ContentPartType::Text) {
                    if (part.text.empty()) continue;
                    if (!first_part) payload += ',';
                    payload += R"({"text":")";
                    payload += core::utils::escape_json_string(part.text);
                    payload += R"("})";
                    first_part = false;
                    continue;
                }

                if (!first_part) payload += ',';
                if (const auto encoded = encode_image_part(part); encoded.has_value()) {
                    payload += R"({"inlineData":{"mimeType":")";
                    payload += core::utils::escape_json_string(encoded->mime_type);
                    payload += R"(","data":")";
                    payload += core::utils::escape_json_string(encoded->base64_data);
                    payload += R"("}})";
                } else {
                    payload += R"({"text":")";
                    payload += core::utils::escape_json_string(
                        unavailable_image_attachment_text(part.path));
                    payload += R"("})";
                }
                first_part = false;
            }
        } else {
            payload += R"({"text":")";
            payload += core::utils::escape_json_string(msg.content);
            payload += "\"}";
        }

        payload += "]}";
        if (i < filtered_messages.size() - 1) payload += ',';
    }
    payload += "]}";
    return payload;
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_gemini_sse_chunk
// ─────────────────────────────────────────────────────────────────────────────

std::pair<std::string, std::vector<ToolCall>>
parse_gemini_sse_chunk(std::string_view json_sv) {
    if (json_sv.empty()) return {};

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_sv);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return {};

    std::string           content_str;
    std::vector<ToolCall> tools;

    simdjson::ondemand::array candidates;
    if (doc["candidates"].get_array().get(candidates) != simdjson::SUCCESS) return {};

    for (auto candidate : candidates) {
        simdjson::ondemand::object content_obj;
        if (candidate["content"].get_object().get(content_obj) != simdjson::SUCCESS) continue;
        simdjson::ondemand::array parts;
        if (content_obj["parts"].get_array().get(parts) != simdjson::SUCCESS) continue;

        for (auto part : parts) {
            std::string_view text_v;
            if (part["text"].get_string().get(text_v) == simdjson::SUCCESS) {
                content_str += std::string(text_v);
                continue;
            }

            simdjson::ondemand::object function_call;
            if (part["functionCall"].get_object().get(function_call) == simdjson::SUCCESS) {
                ToolCall call;
                call.id   = "call_gemini_" + std::to_string(++g_gemini_call_counter);
                call.type = "function";

                std::string_view name_v;
                if (function_call["name"].get_string().get(name_v) == simdjson::SUCCESS)
                    call.function.name = std::string(name_v);

                simdjson::ondemand::object args_obj;
                if (function_call["args"].get_object().get(args_obj) == simdjson::SUCCESS) {
                    std::string_view raw_v;
                    if (args_obj.raw_json().get(raw_v) == simdjson::SUCCESS)
                        call.function.arguments = std::string(raw_v);
                }
                tools.push_back(std::move(call));
            }
        }
    }

    return {std::move(content_str), std::move(tools)};
}

GeminiUsageMetadata extract_gemini_usage_metadata(std::string_view json_sv) {
    GeminiUsageMetadata usage;
    if (json_sv.empty()) return usage;

    simdjson::dom::parser parser;
    simdjson::padded_string padded(json_sv.data(), json_sv.size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return usage;
    }

    int64_t prompt = 0;
    int64_t completion = 0;
    [[maybe_unused]] auto prompt_err =
        doc["usageMetadata"]["promptTokenCount"].get(prompt);
    [[maybe_unused]] auto completion_err =
        doc["usageMetadata"]["candidatesTokenCount"].get(completion);
    if (completion <= 0) {
        [[maybe_unused]] auto output_err =
            doc["usageMetadata"]["outputTokenCount"].get(completion);
    }

    usage.prompt_tokens = prompt > 0 ? static_cast<int32_t>(prompt) : 0;
    usage.completion_tokens = completion > 0 ? static_cast<int32_t>(completion) : 0;
    return usage;
}

// ─────────────────────────────────────────────────────────────────────────────
// GeminiProtocol
// ─────────────────────────────────────────────────────────────────────────────

void GeminiProtocol::prepare_request(ChatRequest& request) {
    if (request.model.empty()) return;
    request.model = normalize_requested_gemini_model(request.model);
}

std::string GeminiProtocol::serialize(const ChatRequest& req) const {
    return serialize_gemini_request(req, req.model);
}

cpr::Header GeminiProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{{"Content-Type", "application/json"}};
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

std::string GeminiProtocol::build_url(std::string_view base_url,
                                       std::string_view model) const {
    const std::string normalized_model = normalize_requested_gemini_model(model);
    return std::string(base_url)
         + "/v1beta/models/"
         + normalized_model
         + ":streamGenerateContent?alt=sse";
}

ParseResult GeminiProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    const std::string_view json_sv = parsed.data;

    if (parsed.is_done) {
        ParseResult r;
        r.done = true;
        return r;
    }

    auto [content, tools] = parse_gemini_sse_chunk(json_sv);

    ParseResult result;
    const GeminiUsageMetadata usage = extract_gemini_usage_metadata(json_sv);
    result.prompt_tokens = usage.prompt_tokens;
    result.completion_tokens = usage.completion_tokens;
    if (!content.empty() || !tools.empty()) {
        StreamChunk chunk;
        chunk.content = std::move(content);
        chunk.tools   = std::move(tools);
        result.chunks.push_back(std::move(chunk));
    }
    return result;
}

} // namespace core::llm::protocols
