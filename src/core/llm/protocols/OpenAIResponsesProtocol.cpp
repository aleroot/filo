#include "OpenAIResponsesProtocol.hpp"
#include "SseUtils.hpp"
#include "../Models.hpp"
#include "../transport/HttpHeaderUtils.hpp"
#include "../../logging/Logger.hpp"
#include "../../utils/JsonUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include "../../utils/Uuid.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <fstream>
#include <format>
#include <random>

namespace core::llm::protocols {

namespace {

[[nodiscard]] std::string extract_assistant_output_text(simdjson::dom::object item);

[[nodiscard]] std::string generate_prompt_cache_key() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    const uint64_t hi = dist(rng);
    const uint64_t lo = dist(rng);
    return std::format("filo-{0:016x}{1:016x}", hi, lo);
}

[[nodiscard]] std::string load_or_create_codex_installation_id(
    const std::filesystem::path& config_dir) {
    const std::string generated = core::utils::random_uuid_v4();
    if (config_dir.empty()) return generated;

    try {
        const auto path = config_dir / "codex_installation_id";

        {
            std::ifstream in(path);
            std::string value;
            if (in && std::getline(in, value)) {
                value = core::utils::str::trim_ascii_copy(value);
                if (!value.empty()) return value;
            }
        }

        std::error_code ec;
        std::filesystem::create_directories(config_dir, ec);
        std::ofstream out(path, std::ios::trunc);
        if (out) {
            out << generated << '\n';
        }
    } catch (...) {
        // Identity persistence should never make the transport unusable.
    }

    return generated;
}

void append_member_before_object_end(std::string& object_json, std::string_view member_json) {
    if (object_json.empty() || object_json.back() != '}') return;
    object_json.pop_back();
    object_json += ',';
    object_json += member_json;
    object_json += '}';
}

[[nodiscard]] std::string websocket_url_for_responses(std::string_view base_url) {
    std::string url(base_url);
    if (url.starts_with("https://")) {
        url.replace(0, std::string("https://").size(), "wss://");
    } else if (url.starts_with("http://")) {
        url.replace(0, std::string("http://").size(), "ws://");
    }
    return url + "/responses";
}

void append_tool_schema(std::string& payload, const std::vector<Tool>& tools) {
    payload += R"(,"tools":[)";
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const auto& def = tools[i].function;
        payload += R"({"type":"function","name":")";
        payload += core::utils::escape_json_string(def.name);
        payload += R"(","description":")";
        payload += core::utils::escape_json_string(def.description);
        payload += R"(","strict":false,"parameters":{"type":"object","properties":{)";

        for (std::size_t j = 0; j < def.parameters.size(); ++j) {
            const auto& p = def.parameters[j];
            payload += "\"";
            payload += core::utils::escape_json_string(p.name);
            payload += R"(":{"type":")";
            payload += core::utils::escape_json_string(p.type);
            payload += R"(","description":")";
            payload += core::utils::escape_json_string(p.description);
            payload += "\"";
            if (!p.items_schema.empty()) {
                payload += R"(,"items":)";
                payload += p.items_schema;
            }
            payload += "}";
            if (j + 1 < def.parameters.size()) payload += ",";
        }

        payload += R"(},"required":[)";
        bool first_req = true;
        for (const auto& p : def.parameters) {
            if (!p.required) continue;
            if (!first_req) payload += ",";
            payload += "\"";
            payload += core::utils::escape_json_string(p.name);
            payload += "\"";
            first_req = false;
        }
        payload += "]}}";
        if (i + 1 < tools.size()) payload += ",";
    }
    payload += "]";
}

[[nodiscard]] std::string collect_instructions(const ChatRequest& req) {
    std::string instructions;
    bool first = true;
    for (const auto& msg : req.messages) {
        if (msg.role != "system" || msg.content.empty()) continue;
        if (!first) instructions += "\n\n";
        instructions += msg.content;
        first = false;
    }
    return instructions;
}

void append_message_item(std::string& payload,
                         const Message& msg) {
    payload += R"({"type":"message","role":")";
    payload += core::utils::escape_json_string(msg.role);
    payload += R"(","content":[)";

    bool first_part = true;
    const auto append_part_sep = [&]() {
        if (!first_part) payload += ",";
        first_part = false;
    };

    const auto append_text_part = [&](std::string_view text) {
        append_part_sep();
        payload += R"({"type":")";
        payload += (msg.role == "assistant") ? "output_text" : "input_text";
        payload += R"(","text":")";
        payload += core::utils::escape_json_string(text);
        payload += R"("})";
    };

    if (!msg.content_parts.empty()) {
        for (const auto& part : msg.content_parts) {
            if (part.type == ContentPartType::Text) {
                if (!part.text.empty()) {
                    append_text_part(part.text);
                }
                continue;
            }

            if (const auto encoded = encode_image_part(part); encoded.has_value()) {
                append_part_sep();
                payload += R"({"type":"input_image","image_url":")";
                payload += core::utils::escape_json_string(encoded->data_url());
                payload += R"("})";
            } else {
                append_text_part(unavailable_media_attachment_text(part.type, media_reference(part)));
            }
        }
    } else if (!msg.content.empty()) {
        append_text_part(msg.content);
    }

    payload += R"(]})";
}

void append_function_call_item(std::string& payload,
                               const ToolCall& call,
                               std::size_t fallback_idx) {
    const std::string call_id = !call.id.empty()
        ? call.id
        : "call_" + std::to_string(fallback_idx);
    const std::string arguments = call.function.arguments.empty()
        ? "{}"
        : call.function.arguments;

    payload += R"({"type":"function_call","call_id":")";
    payload += core::utils::escape_json_string(call_id);
    payload += R"(","name":")";
    payload += core::utils::escape_json_string(call.function.name);
    payload += R"(","arguments":")";
    payload += core::utils::escape_json_string(arguments);
    payload += R"("})";
}

void append_function_call_output_item(std::string& payload,
                                      const Message& msg,
                                      std::size_t fallback_idx) {
    const std::string call_id = !msg.tool_call_id.empty()
        ? msg.tool_call_id
        : "call_" + std::to_string(fallback_idx);
    payload += R"({"type":"function_call_output","call_id":")";
    payload += core::utils::escape_json_string(call_id);
    payload += R"(","output":")";
    payload += core::utils::escape_json_string(msg.content);
    payload += R"("})";
}

[[nodiscard]] std::vector<std::string> build_input_items(
    const std::vector<Message>& messages) {
    std::vector<std::string> items;
    std::size_t fallback_idx = 0;

    for (const auto& msg : messages) {
        if (msg.role == "system") continue;

        if (msg.role == "tool") {
            if (msg.content.empty() && msg.tool_call_id.empty()) continue;
            std::string item;
            append_function_call_output_item(item, msg, ++fallback_idx);
            items.push_back(std::move(item));
            continue;
        }

        if (!msg.content.empty() || !msg.content_parts.empty()) {
            std::string item;
            append_message_item(item, msg);
            items.push_back(std::move(item));
        }

        if (msg.role == "assistant" && !msg.tool_calls.empty()) {
            for (const auto& tc : msg.tool_calls) {
                if (tc.function.name.empty()) continue;
                std::string item;
                append_function_call_item(item, tc, ++fallback_idx);
                items.push_back(std::move(item));
            }
        }
    }

    return items;
}

void append_input_items(std::string& payload, const std::vector<std::string>& input_items) {
    payload += R"(,"input":[)";
    for (std::size_t i = 0; i < input_items.size(); ++i) {
        if (i > 0) payload += ',';
        payload += input_items[i];
    }
    payload += "]";
}

[[nodiscard]] std::vector<Message> system_messages_only(const std::vector<Message>& messages) {
    std::vector<Message> system_messages;
    for (const auto& message : messages) {
        if (message.role == "system") {
            system_messages.push_back(message);
        }
    }
    return system_messages;
}

[[nodiscard]] bool has_prefix(const std::vector<std::string>& values,
                              const std::vector<std::string>& prefix) {
    return values.size() >= prefix.size()
        && std::equal(prefix.begin(), prefix.end(), values.begin());
}

[[nodiscard]] std::string serialize_assistant_output_item(simdjson::dom::object item) {
    std::string_view item_type;
    if (item["type"].get(item_type) != simdjson::SUCCESS) return {};

    if (item_type == "function_call") {
        ToolCall call;
        call.type = "function";

        std::string_view call_id;
        if (item["call_id"].get(call_id) == simdjson::SUCCESS) {
            call.id = std::string(call_id);
        } else {
            std::string_view id;
            if (item["id"].get(id) == simdjson::SUCCESS) {
                call.id = std::string(id);
            }
        }

        std::string_view name;
        if (item["name"].get(name) == simdjson::SUCCESS) {
            call.function.name = std::string(name);
        }
        if (call.function.name.empty()) return {};

        std::string_view arguments;
        if (item["arguments"].get(arguments) == simdjson::SUCCESS) {
            call.function.arguments = std::string(arguments);
        }

        std::string serialized;
        append_function_call_item(serialized, call, 0);
        return serialized;
    }

    if (item_type == "message") {
        std::string_view role;
        if (item["role"].get(role) != simdjson::SUCCESS || role != "assistant") {
            return {};
        }

        const std::string text = extract_assistant_output_text(item);
        if (text.empty()) return {};

        Message message;
        message.role = "assistant";
        message.content = text;

        std::string serialized;
        append_message_item(serialized, message);
        return serialized;
    }

    return {};
}

[[nodiscard]] std::string lower_ascii(std::string_view value) {
    std::string lowered;
    lowered.reserve(value.size());
    for (const char ch : value) {
        lowered.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }
    return lowered;
}

[[nodiscard]] std::string normalize_openai_effort(std::string_view raw_effort) {
    std::string effort = lower_ascii(raw_effort);
    std::erase_if(effort, [](unsigned char ch) {
        return std::isspace(ch);
    });
    if (effort == "auto" || effort == "unset" || effort == "default") {
        return {};
    }
    if (effort == "low" || effort == "medium" || effort == "high") {
        return effort;
    }
    if (effort == "max") {
        return "high";
    }
    return {};
}

[[nodiscard]] bool model_supports_openai_reasoning_effort(std::string_view model) {
    const std::string lowered = lower_ascii(model);
    return lowered.starts_with("gpt-5")
        || lowered.starts_with("o1")
        || lowered.starts_with("o3")
        || lowered.starts_with("o4");
}

void extract_usage_from_completed(simdjson::dom::element doc, ParseResult& result) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return;

    simdjson::dom::object usage_obj;
    if (response_obj["usage"].get(usage_obj) != simdjson::SUCCESS) return;

    int64_t input_tokens = 0;
    int64_t output_tokens = 0;

    if (usage_obj["input_tokens"].get(input_tokens) != simdjson::SUCCESS) {
        [[maybe_unused]] auto _ = usage_obj["prompt_tokens"].get(input_tokens);
    }
    if (usage_obj["output_tokens"].get(output_tokens) != simdjson::SUCCESS) {
        [[maybe_unused]] auto _ = usage_obj["completion_tokens"].get(output_tokens);
    }

    if (input_tokens > 0) {
        result.prompt_tokens = static_cast<int32_t>(
            std::min<int64_t>(input_tokens, INT32_MAX));
    }
    if (output_tokens > 0) {
        result.completion_tokens = static_cast<int32_t>(
            std::min<int64_t>(output_tokens, INT32_MAX));
    }
}

[[nodiscard]] std::string parse_response_id(simdjson::dom::element doc) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return {};

    std::string_view id;
    if (response_obj["id"].get(id) == simdjson::SUCCESS && !id.empty()) {
        return std::string(id);
    }
    return {};
}

[[nodiscard]] std::string extract_assistant_output_text(simdjson::dom::object item) {
    simdjson::dom::array content_items;
    if (item["content"].get(content_items) != simdjson::SUCCESS) return {};

    std::string text;
    for (simdjson::dom::element content_el : content_items) {
        simdjson::dom::object content_obj;
        if (content_el.get(content_obj) != simdjson::SUCCESS) continue;

        std::string_view content_type;
        if (content_obj["type"].get(content_type) != simdjson::SUCCESS
            || content_type != "output_text") {
            continue;
        }

        std::string_view text_part;
        if (content_obj["text"].get(text_part) == simdjson::SUCCESS) {
            text.append(text_part);
        }
    }
    return text;
}

[[nodiscard]] std::string parse_failed_message(simdjson::dom::element doc) {
    auto from_error_obj = [](simdjson::dom::object error_obj) -> std::string {
        std::string_view message;
        if (error_obj["message"].get(message) == simdjson::SUCCESS && !message.empty()) {
            return std::string(message);
        }
        return {};
    };

    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) == simdjson::SUCCESS) {
        simdjson::dom::object error_obj;
        if (response_obj["error"].get(error_obj) == simdjson::SUCCESS) {
            if (std::string msg = from_error_obj(error_obj); !msg.empty()) return msg;
        }

        std::string_view top_message;
        if (response_obj["message"].get(top_message) == simdjson::SUCCESS && !top_message.empty()) {
            return std::string(top_message);
        }
    }

    simdjson::dom::object error_obj;
    if (doc["error"].get(error_obj) == simdjson::SUCCESS) {
        if (std::string msg = from_error_obj(error_obj); !msg.empty()) return msg;
    }

    std::string_view top_message;
    if (doc["message"].get(top_message) == simdjson::SUCCESS && !top_message.empty()) {
        return std::string(top_message);
    }

    return "Responses API stream failed.";
}

[[nodiscard]] std::string parse_incomplete_reason(simdjson::dom::element doc) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return {};

    simdjson::dom::object details_obj;
    if (response_obj["incomplete_details"].get(details_obj) != simdjson::SUCCESS) return {};

    std::string_view reason;
    if (details_obj["reason"].get(reason) == simdjson::SUCCESS && !reason.empty()) {
        return std::string(reason);
    }
    return {};
}

} // namespace

void OpenAIResponsesProtocol::prepare_request(ChatRequest& req) {
    std::scoped_lock lock(shared_state_->mutex);

    active_session_id_ = req.session_id;
    auto& session = shared_state_->sessions[active_session_id_];
    if (session.prompt_cache_key.empty()) {
        session.prompt_cache_key = req.session_id.empty()
            ? generate_prompt_cache_key()
            : req.session_id;
    }
    if (req.prompt_cache_key.empty()) {
        req.prompt_cache_key = session.prompt_cache_key;
    }
    if (req.previous_response_id.empty() && !session.previous_response_id.empty()) {
        req.previous_response_id = session.previous_response_id;
    }
}

void OpenAIResponsesProtocol::reset_state() {
    std::scoped_lock lock(shared_state_->mutex);
    shared_state_->sessions.clear();
    active_session_id_.clear();
    in_progress_response_id_.clear();
    last_response_id_.clear();
    saw_text_delta_ = false;
}

std::string OpenAIResponsesProtocol::serialize_with_input_items(
    const ChatRequest& req,
    const std::vector<std::string>& input_items,
    std::optional<std::string_view> previous_response_id_override) const {
    std::string payload;
    payload.reserve(8192);

    payload += R"({"model":")";
    payload += core::utils::escape_json_string(req.model);
    payload += R"(","instructions":")";
    payload += core::utils::escape_json_string(collect_instructions(req));
    payload += R"(","stream":)";
    payload += req.stream ? "true" : "false";
    payload += R"(,"store":false)";

    const std::string_view previous_response_id = previous_response_id_override.has_value()
        ? *previous_response_id_override
        : std::string_view{req.previous_response_id};
    if (!previous_response_id.empty()) {
        payload += R"(,"previous_response_id":")";
        payload += core::utils::escape_json_string(previous_response_id);
        payload += '"';
    }

    if (req.temperature.has_value()) {
        payload += R"(,"temperature":)";
        payload += std::to_string(*req.temperature);
    }
    if (req.max_tokens.has_value()) {
        payload += R"(,"max_output_tokens":)";
        payload += std::to_string(*req.max_tokens);
    }
    if (model_supports_openai_reasoning_effort(req.model)) {
        const std::string effort = normalize_openai_effort(req.effort);
        if (!effort.empty()) {
            payload += R"(,"reasoning":{"effort":")";
            payload += core::utils::escape_json_string(effort);
            payload += R"("})";
        }
    }

    append_input_items(payload, input_items);

    payload += R"(,"tool_choice":"auto","parallel_tool_calls":)";
    payload += req.tools.empty() ? "false" : "true";

    if (!req.tools.empty()) {
        append_tool_schema(payload, req.tools);
    }

    if (!req.prompt_cache_key.empty()) {
        payload += R"(,"prompt_cache_key":")";
        payload += core::utils::escape_json_string(req.prompt_cache_key);
        payload += '"';
    }

    const std::string_view service_tier = req.service_tier.empty()
        ? std::string_view{default_service_tier_}
        : std::string_view{req.service_tier};
    if (!service_tier.empty()) {
        payload += R"(,"service_tier":")";
        payload += core::utils::escape_json_string(service_tier);
        payload += '"';
    }

    if (include_reasoning_encrypted_) {
        payload += R"(,"include":["reasoning.encrypted_content"])";
    } else {
        payload += R"(,"include":[])";
    }

    if (req.response_format.type == ResponseFormat::Type::JsonSchema) {
        payload += R"(,"text":{"format":{"type":"json_schema","name":"structured_output","strict":true,"schema":)";
        payload += req.response_format.schema.empty() ? R"({"type":"object"})" : req.response_format.schema;
        payload += "}}";
    } else if (req.response_format.type == ResponseFormat::Type::JsonObject) {
        payload += R"(,"text":{"format":{"type":"json_schema","name":"json_object","strict":false,"schema":{"type":"object","additionalProperties":true}}})";
    }

    payload += "}";
    return payload;
}

std::string OpenAIResponsesProtocol::serialize(const ChatRequest& req) const {
    return serialize_with_input_items(req, build_input_items(req.messages));
}

cpr::Header OpenAIResponsesProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept",       "text/event-stream"},
    };
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }

    // OpenAI Codex backend account-scoped tokens require this header.
    if (auto it = auth.properties.find("account_id");
        it != auth.properties.end() && !it->second.empty()
        && headers.count("chatgpt-account-id") == 0) {
        headers["chatgpt-account-id"] = it->second;
    }

    return headers;
}

std::string OpenAIResponsesProtocol::build_url(std::string_view base_url,
                                               [[maybe_unused]] std::string_view model) const {
    return std::string(base_url) + "/responses";
}

CodexResponsesProtocol::CodexResponsesProtocol(bool include_reasoning_encrypted,
                                               std::string default_service_tier,
                                               std::filesystem::path config_dir)
    : OpenAIResponsesProtocol(include_reasoning_encrypted, std::move(default_service_tier))
    , config_dir_(std::move(config_dir)) {}

std::string CodexResponsesProtocol::serialize(const ChatRequest& request) const {
    return serialize_codex_with_input_items(request, build_input_items(request.messages));
}

std::string CodexResponsesProtocol::serialize_codex_with_input_items(
    const ChatRequest& request,
    const std::vector<std::string>& input_items,
    std::optional<std::string_view> previous_response_id_override) const {
    std::string payload = serialize_with_input_items(
        request, input_items, previous_response_id_override);
    const std::string window_id = (request.session_id.empty() ? std::string("filo") : request.session_id) + ":0";
    std::string metadata = R"("client_metadata":{"x-codex-installation-id":")";
    metadata += core::utils::escape_json_string(installation_id());
    metadata += R"(","x-codex-window-id":")";
    metadata += core::utils::escape_json_string(window_id);
    metadata += R"("})";
    append_member_before_object_end(payload, metadata);
    return payload;
}

void CodexResponsesProtocol::prepare_headers(cpr::Header& headers,
                                             const ChatRequest& request,
                                             [[maybe_unused]] std::string_view base_url) {
    const std::string thread_id = request.session_id.empty()
        ? std::string("filo")
        : request.session_id;

    std::string turn_state;
    {
        std::lock_guard lock(transport_state_->mutex);
        if (!request.transport_turn_id.empty()) {
            if (auto it = transport_state_->turn_states.find(request.transport_turn_id);
                it != transport_state_->turn_states.end()) {
                turn_state = it->second;
            }
        }
    }

    headers["x-client-request-id"] = thread_id;
    headers["x-codex-installation-id"] = installation_id();
    headers["x-codex-window-id"] = thread_id + ":0";
    headers["x-responsesapi-include-timing-metrics"] = "true";
    if (!turn_state.empty()) {
        headers["x-codex-turn-state"] = turn_state;
    } else {
        headers.erase("x-codex-turn-state");
    }
}

std::string CodexResponsesProtocol::build_websocket_url(
    std::string_view base_url,
    [[maybe_unused]] std::string_view model) const {
    return websocket_url_for_responses(base_url);
}

void CodexResponsesProtocol::prepare_websocket_headers(cpr::Header& headers,
                                                       const ChatRequest& request,
                                                       std::string_view base_url) {
    headers.erase("Content-Type");
    headers.erase("Accept");
    prepare_headers(headers, request, base_url);
    headers["OpenAI-Beta"] = "responses_websockets=2026-02-06";
}

std::string CodexResponsesProtocol::serialize_websocket_request(
    const ChatRequest& request) const {
    const auto full_input_items = build_input_items(request.messages);

    ChatRequest signature_request = request;
    signature_request.messages = system_messages_only(request.messages);
    signature_request.previous_response_id.clear();
    const std::string request_signature = serialize_codex_with_input_items(
        signature_request, build_input_items(signature_request.messages), std::string_view{});

    std::string incremental_previous_response_id;
    std::vector<std::string> incremental_input_items;
    if (!request.transport_turn_id.empty()) {
        std::lock_guard lock(transport_state_->mutex);
        if (auto last = transport_state_->last_ws_requests.find(request.transport_turn_id);
            last != transport_state_->last_ws_requests.end()
            && last->second.request_signature == request_signature
            && !last->second.response_id.empty()) {
            std::vector<std::string> baseline = last->second.request_input_items;
            baseline.insert(
                baseline.end(),
                last->second.response_items_added.begin(),
                last->second.response_items_added.end());

            if (has_prefix(full_input_items, baseline)
                && full_input_items.size() > baseline.size()) {
                incremental_input_items = std::vector<std::string>{
                    full_input_items.begin() + static_cast<std::ptrdiff_t>(baseline.size()),
                    full_input_items.end()};
                incremental_previous_response_id = last->second.response_id;
            }
        }

        transport_state_->pending_ws_requests[request.transport_turn_id] =
            TransportState::PendingWebSocketRequest{
                .request_signature = request_signature,
                .request_input_items = full_input_items,
            };
    }

    std::string payload = incremental_input_items.empty()
        ? serialize_codex_with_input_items(request, full_input_items)
        : serialize_codex_with_input_items(
            request, incremental_input_items, std::string_view{incremental_previous_response_id});

    active_transport_turn_id_ = request.transport_turn_id;
    active_response_items_.clear();

    if (payload.empty() || payload.front() != '{') {
        return {};
    }
    payload.erase(payload.begin());
    return std::string{R"({"type":"response.create",)"} + payload;
}

void CodexResponsesProtocol::abandon_websocket_request(const ChatRequest& request) {
    if (request.transport_turn_id.empty()) {
        return;
    }

    std::lock_guard lock(transport_state_->mutex);
    transport_state_->pending_ws_requests.erase(request.transport_turn_id);
    if (active_transport_turn_id_ == request.transport_turn_id) {
        active_transport_turn_id_.clear();
        active_response_items_.clear();
    }
}

ParseResult CodexResponsesProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    std::string data_scratch;
    std::string event_type;
    std::string serialized_output_item;

    if (sse::parse_event_payload(raw_event, parsed, data_scratch)) {
        event_type = std::string(parsed.event);
        if (!parsed.is_done) {
            thread_local simdjson::dom::parser parser;
            simdjson::padded_string padded(parsed.data);
            simdjson::dom::element doc;
            if (parser.parse(padded).get(doc) == simdjson::SUCCESS) {
                if (event_type.empty()) {
                    std::string_view type_from_payload;
                    if (doc["type"].get(type_from_payload) == simdjson::SUCCESS) {
                        event_type = std::string(type_from_payload);
                    }
                }

                if (event_type == "response.output_item.done") {
                    simdjson::dom::object item;
                    if (doc["item"].get(item) == simdjson::SUCCESS) {
                        serialized_output_item = serialize_assistant_output_item(item);
                    }
                }
            }
        }
    }

    ParseResult result = OpenAIResponsesProtocol::parse_event(raw_event);

    if (!serialized_output_item.empty()) {
        active_response_items_.push_back(std::move(serialized_output_item));
    }

    if (result.done
        && event_type == "response.completed"
        && !active_transport_turn_id_.empty()
        && !last_response_id().empty()) {
        std::lock_guard lock(transport_state_->mutex);
        if (auto pending = transport_state_->pending_ws_requests.find(active_transport_turn_id_);
            pending != transport_state_->pending_ws_requests.end()) {
            transport_state_->last_ws_requests[active_transport_turn_id_] =
                TransportState::LastWebSocketRequest{
                    .response_id = last_response_id(),
                    .request_signature = std::move(pending->second.request_signature),
                    .request_input_items = std::move(pending->second.request_input_items),
                    .response_items_added = std::move(active_response_items_),
                };
            transport_state_->pending_ws_requests.erase(pending);
            while (transport_state_->last_ws_requests.size() > 32) {
                transport_state_->last_ws_requests.erase(
                    transport_state_->last_ws_requests.begin());
            }
        }
        active_transport_turn_id_.clear();
        active_response_items_.clear();
    }

    if (result.done && !active_transport_turn_id_.empty()) {
        std::lock_guard lock(transport_state_->mutex);
        transport_state_->pending_ws_requests.erase(active_transport_turn_id_);
        active_transport_turn_id_.clear();
        active_response_items_.clear();
    }

    return result;
}

std::string CodexResponsesProtocol::websocket_connection_key(
    std::string_view url,
    const cpr::Header& headers,
    const ChatRequest& request) const {
    std::string key(url);
    key += "\nsession=";
    key += request.session_id;
    key += "\nturn=";
    key += request.transport_turn_id;

    for (std::string_view name : {
             std::string_view{"Authorization"},
             std::string_view{"chatgpt-account-id"},
             std::string_view{"OpenAI-Beta"}}) {
        if (auto value = transport::find_header(headers, name);
            value.has_value() && !value->empty()) {
            key += '\n';
            key += name;
            key += ':';
            key += *value;
        }
    }
    return key;
}

void CodexResponsesProtocol::observe_response_headers(const cpr::Header& headers,
                                                      const ChatRequest& request) {
    if (request.transport_turn_id.empty()) {
        return;
    }

    auto turn_state = transport::find_header(headers, "x-codex-turn-state");
    if (!turn_state.has_value() || turn_state->empty()) {
        return;
    }

    std::lock_guard lock(transport_state_->mutex);
    transport_state_->turn_states[request.transport_turn_id] = *turn_state;
    while (transport_state_->turn_states.size() > 32) {
        transport_state_->turn_states.erase(transport_state_->turn_states.begin());
    }
}

void CodexResponsesProtocol::reset_state() {
    OpenAIResponsesProtocol::reset_state();
    std::lock_guard lock(transport_state_->mutex);
    transport_state_->turn_states.clear();
    transport_state_->last_ws_requests.clear();
    transport_state_->pending_ws_requests.clear();
    active_response_items_.clear();
    active_transport_turn_id_.clear();
}

std::unique_ptr<ApiProtocolBase> CodexResponsesProtocol::clone() const {
    auto cloned = std::make_unique<CodexResponsesProtocol>(
        include_reasoning_encrypted_,
        default_service_tier_,
        config_dir_);
    share_continuity_state_with(*cloned);
    cloned->transport_state_ = transport_state_;
    return cloned;
}

std::string CodexResponsesProtocol::installation_id() const {
    std::lock_guard lock(transport_state_->mutex);
    if (transport_state_->installation_id.empty()) {
        transport_state_->installation_id = load_or_create_codex_installation_id(config_dir_);
    }
    return transport_state_->installation_id;
}

ParseResult OpenAIResponsesProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};

    const std::string_view payload_sv = parsed.data;
    if (parsed.is_done) {
        ParseResult result;
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string padded(payload_sv);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        core::logging::debug("[responses] Failed to parse SSE payload JSON");
        return {};
    }

    std::string_view event_type = parsed.event;
    if (event_type.empty()) {
        std::string_view type_from_payload;
        if (doc["type"].get(type_from_payload) == simdjson::SUCCESS) {
            event_type = type_from_payload;
        }
    }

    ParseResult result;

    if (event_type == "response.created") {
        in_progress_response_id_ = parse_response_id(doc);
        return result;
    }

    if (event_type == "response.output_text.delta") {
        std::string_view delta;
        if (doc["delta"].get(delta) == simdjson::SUCCESS && !delta.empty()) {
            saw_text_delta_ = true;
            result.chunks.push_back(StreamChunk::make_content(std::string(delta)));
        }
        return result;
    }

    if (event_type == "response.output_text.done") {
        std::string_view text;
        if (doc["text"].get(text) == simdjson::SUCCESS && !text.empty()) {
            saw_text_delta_ = true;
            result.chunks.push_back(StreamChunk::make_content(std::string(text)));
        }
        return result;
    }

    if (event_type == "response.reasoning_text.delta"
        || event_type == "response.reasoning_summary_text.delta") {
        std::string_view delta;
        if (doc["delta"].get(delta) == simdjson::SUCCESS && !delta.empty()) {
            result.chunks.push_back(StreamChunk::make_reasoning(std::string(delta)));
        }
        return result;
    }

    if (event_type == "response.output_item.done" || event_type == "response.output_item.added") {
        simdjson::dom::object item;
        if (doc["item"].get(item) != simdjson::SUCCESS) return result;

        std::string_view item_type;
        if (item["type"].get(item_type) != simdjson::SUCCESS) return result;

        if (item_type == "function_call") {
            ToolCall call;
            call.type = "function";

            int64_t index = -1;
            if (item["index"].get(index) == simdjson::SUCCESS && index >= 0 && index <= INT32_MAX) {
                call.index = static_cast<int>(index);
            }

            std::string_view call_id;
            if (item["call_id"].get(call_id) == simdjson::SUCCESS) {
                call.id = std::string(call_id);
            } else {
                std::string_view id;
                if (item["id"].get(id) == simdjson::SUCCESS) {
                    call.id = std::string(id);
                }
            }

            std::string_view name;
            if (item["name"].get(name) == simdjson::SUCCESS) {
                call.function.name = std::string(name);
            }

            std::string_view arguments;
            if (item["arguments"].get(arguments) == simdjson::SUCCESS) {
                call.function.arguments = std::string(arguments);
            }
            if (call.function.arguments.empty()) {
                call.function.arguments = "{}";
            }

            if (!call.function.name.empty()) {
                StreamChunk chunk;
                chunk.tools.push_back(std::move(call));
                result.chunks.push_back(std::move(chunk));
            }
        } else if (item_type == "message" && !saw_text_delta_) {
            std::string_view role;
            if (item["role"].get(role) != simdjson::SUCCESS || role != "assistant") {
                return result;
            }

            std::string text = extract_assistant_output_text(item);
            if (!text.empty()) {
                result.chunks.push_back(StreamChunk::make_content(std::move(text)));
            }
        }

        return result;
    }

    if (event_type == "response.completed") {
        extract_usage_from_completed(doc, result);
        std::string response_id = parse_response_id(doc);
        if (response_id.empty() && !in_progress_response_id_.empty()) {
            response_id = in_progress_response_id_;
        }
        if (!response_id.empty()) {
            last_response_id_ = std::move(response_id);
        }
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    if (event_type == "response.incomplete") {
        std::string reason = parse_incomplete_reason(doc);
        if (reason.empty()) reason = "unknown";
        result.chunks.push_back(StreamChunk::make_error(
            "\n[Responses API incomplete response: " + reason + "]"));
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    if (event_type == "response.failed" || event_type == "error") {
        const std::string message = parse_failed_message(doc);
        result.chunks.push_back(StreamChunk::make_error("\n[Responses API error: " + message + "]"));
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    return result;
}

void OpenAIResponsesProtocol::on_response(const HttpResponse& response) {
    RateLimitInfo info;

    const auto parse_int = [&](std::string_view key) -> int32_t {
        if (const auto it = response.headers.find(std::string(key)); it != response.headers.end()) {
            try {
                return static_cast<int32_t>(std::stoi(it->second));
            } catch (...) {}
        }
        return 0;
    };

    info.requests_limit     = parse_int("x-ratelimit-limit-requests");
    info.requests_remaining = parse_int("x-ratelimit-remaining-requests");
    info.tokens_limit       = parse_int("x-ratelimit-limit-tokens");
    info.tokens_remaining   = parse_int("x-ratelimit-remaining-tokens");
    info.retry_after        = parse_int("retry-after");
    info.is_rate_limited    = (response.status_code == 429 || info.retry_after > 0);

    if (info.requests_limit > 0 && info.requests_remaining == 0 && !info.is_rate_limited) {
        info.requests_remaining = info.requests_limit;
    }
    if (info.tokens_limit > 0 && info.tokens_remaining == 0 && !info.is_rate_limited) {
        info.tokens_remaining = info.tokens_limit;
    }

    last_rate_limit_ = info;

    if (response.status_code == 200 && !last_response_id_.empty()) {
        std::scoped_lock lock(shared_state_->mutex);
        shared_state_->sessions[active_session_id_].previous_response_id = last_response_id_;
    }
}

} // namespace core::llm::protocols
