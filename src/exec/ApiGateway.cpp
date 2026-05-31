#include "ApiGateway.hpp"
#include "ApiGatewayModelList.hpp"

#include "http/SseStream.hpp"
#include "openai/OpenAIChatCompletionStreamEncoder.hpp"

#include "../core/config/ConfigManager.hpp"
#include "../core/budget/BudgetTracker.hpp"
#include "../core/llm/HttpLLMProvider.hpp"
#include "../core/llm/ModelCatalogDiscovery.hpp"
#include "../core/llm/Models.hpp"
#include "../core/llm/ProviderManager.hpp"
#include "../core/llm/providers/RouterProvider.hpp"
#include "../core/llm/routing/RouterEngine.hpp"
#include "../core/logging/Logger.hpp"
#include "../core/utils/JsonUtils.hpp"
#include "../core/utils/JsonWriter.hpp"
#include "../core/utils/UriUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace exec::gateway {

namespace {

using core::utils::JsonWriter;
namespace uri = core::utils::uri;

struct GatewayRuntime {
    GatewayRuntime(const core::config::AppConfig& config_ref, core::llm::ProviderManager& provider_manager_ref, ProviderCatalog provider_catalog) : config(config_ref),
          provider_manager(provider_manager_ref),
          providers(std::move(provider_catalog.providers)),
          provider_default_models(std::move(provider_catalog.provider_default_models)) {}

    const core::config::AppConfig& config;
    core::llm::ProviderManager& provider_manager;
    core::llm::ProviderDescriptorSet providers;
    std::unordered_map<std::string, std::string> provider_default_models;
    std::shared_ptr<core::llm::routing::RouterEngine> router_engine;
    std::shared_ptr<core::llm::providers::RouterProvider> router_provider;
    bool router_available = false;
};

[[nodiscard]] std::string to_lower_ascii(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

[[nodiscard]] std::filesystem::path normalize_path(const std::filesystem::path& path) {
    std::error_code ec;
    auto normalized = std::filesystem::weakly_canonical(path, ec);
    if (!ec) {
        return normalized.lexically_normal();
    }

    ec.clear();
    normalized = std::filesystem::absolute(path, ec);
    if (!ec) {
        return normalized.lexically_normal();
    }

    return path.lexically_normal();
}

[[nodiscard]] std::optional<std::filesystem::path>
parse_local_file_uri(std::string_view uri_value, std::string& error_out) {
    if (!to_lower_ascii(uri_value).starts_with("file://")) {
        error_out = "Only file:// root URIs are supported";
        return std::nullopt;
    }

    std::string_view rest = uri_value.substr(7);
    std::string_view encoded_path;
    if (rest.starts_with('/')) {
        encoded_path = rest;
    } else {
        const auto slash = rest.find('/');
        const std::string_view authority =
            slash == std::string_view::npos ? rest : rest.substr(0, slash);
        const auto lowered_authority = to_lower_ascii(authority);
        if (!authority.empty() && lowered_authority != "localhost") {
            error_out = "Unsupported file URI authority";
            return std::nullopt;
        }
        encoded_path = slash == std::string_view::npos ? std::string_view{"/"} : rest.substr(slash);
    }

    if (encoded_path.find('?') != std::string_view::npos
        || encoded_path.find('#') != std::string_view::npos) {
        error_out = "File URI must not include query or fragment";
        return std::nullopt;
    }

    std::string decoded_path;
    if (!uri::percent_decode(encoded_path, decoded_path)) {
        error_out = "Malformed percent-encoding in file URI";
        return std::nullopt;
    }

    if (decoded_path.empty() || decoded_path.find('\0') != std::string::npos) {
        error_out = "Invalid file URI path";
        return std::nullopt;
    }

    return normalize_path(std::filesystem::path(decoded_path));
}

struct GatewayRequest {
    core::llm::ChatRequest request;
    std::string requested_model;
    bool stream = false;
    bool stream_include_usage = false;
};

struct GatewaySelection {
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::shared_ptr<core::llm::providers::RouterProvider> ephemeral_router;
    std::string provider_name;
    std::string model_override;
    bool uses_router = false;
};

struct CompletionResult {
    bool has_error = false;
    std::string error_message;
    std::string content;
    std::vector<core::llm::ToolCall> tool_calls;
    core::llm::TokenUsage usage;
    std::string model;
};

[[nodiscard]] std::string trim_copy(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size()
           && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin
           && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

[[nodiscard]] std::string normalize_model_selection_mode(std::string_view value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (const unsigned char ch : value) {
        if (std::isspace(ch) || ch == '-' || ch == '_') continue;
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

[[nodiscard]] bool prefer_router_by_default(std::string_view mode) {
    const std::string normalized = normalize_model_selection_mode(mode);
    return normalized == "router" || normalized == "auto";
}

[[nodiscard]] int64_t unix_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(
               now.time_since_epoch()).count();
}

[[nodiscard]] std::string make_gateway_id(std::string_view prefix) {
    static std::atomic<uint64_t> counter{1};
    const uint64_t sequence = counter.fetch_add(1, std::memory_order_relaxed);
    return std::format("{}-{:x}", prefix, sequence);
}

[[nodiscard]] std::string sanitize_gateway_scope_component(std::string_view value) {
    std::string out;
    out.reserve(std::min<std::size_t>(value.size(), 128));
    for (const unsigned char ch : value) {
        if (out.size() >= 128) break;
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == ':') {
            out.push_back(static_cast<char>(ch));
        } else if (!std::iscntrl(ch)) {
            out.push_back('_');
        }
    }
    return trim_copy(out);
}

[[nodiscard]] std::string hashed_gateway_scope(std::string_view prefix,
                                               std::string_view value) {
    return std::format(
        "gateway:{}:{:016x}",
        prefix,
        static_cast<unsigned long long>(std::hash<std::string_view>{}(value)));
}

[[nodiscard]] std::string gateway_session_id_for_request(const httplib::Request& request) {
    std::string explicit_session;
    for (const std::string_view header_name : {
             "X-Filo-Session-Id",
             "X-Gateway-Session-Id",
             "OpenAI-Session-Id",
             "MCP-Session-Id",
         }) {
        const std::string header_value =
            trim_copy(request.get_header_value(std::string(header_name)));
        if (header_value.empty()) continue;
        const std::string sanitized = sanitize_gateway_scope_component(header_value);
        if (!sanitized.empty()) {
            explicit_session = sanitized;
            break;
        }
    }

    const std::string authorization = trim_copy(request.get_header_value("Authorization"));
    if (!authorization.empty()) {
        std::string auth_scope = hashed_gateway_scope("auth", authorization);
        if (!explicit_session.empty()) {
            auth_scope += ":session:";
            auth_scope += explicit_session;
        }
        return auth_scope;
    }

    if (!explicit_session.empty()) {
        return "gateway:session:" + explicit_session;
    }

    if (!request.remote_addr.empty()) {
        return hashed_gateway_scope("remote", request.remote_addr);
    }

    return "gateway:anonymous";
}

void assign_gateway_session_scope(const httplib::Request& http_request,
                                  GatewayRequest& gateway_request) {
    gateway_request.request.session_id = gateway_session_id_for_request(http_request);
}

void append_message_text(core::llm::Message& message, std::string_view text) {
    if (text.empty()) return;
    message.content_parts.push_back(core::llm::ContentPart::make_text(std::string(text)));
    message.content += std::string(text);
}

void append_message_marker(core::llm::Message& message, std::string_view marker) {
    if (marker.empty()) return;
    if (!message.content.empty()) message.content += '\n';
    append_message_text(message, marker);
}

void parse_schema_parameters(simdjson::dom::object schema_obj,
                             std::vector<core::tools::ToolParameter>& out) {
    std::set<std::string> required_names;
    simdjson::dom::array required_arr;
    if (schema_obj["required"].get(required_arr) == simdjson::SUCCESS) {
        for (simdjson::dom::element required_el : required_arr) {
            std::string_view required_name;
            if (required_el.get(required_name) == simdjson::SUCCESS) {
                required_names.insert(std::string(required_name));
            }
        }
    }

    simdjson::dom::object properties_obj;
    if (schema_obj["properties"].get(properties_obj) != simdjson::SUCCESS) {
        return;
    }

    for (auto field : properties_obj) {
        core::tools::ToolParameter parameter;
        parameter.name = std::string(field.key);

        simdjson::dom::object property_obj;
        if (field.value.get(property_obj) == simdjson::SUCCESS) {
            std::string_view type_value;
            if (property_obj["type"].get(type_value) == simdjson::SUCCESS) {
                parameter.type = std::string(type_value);
            }

            std::string_view description;
            if (property_obj["description"].get(description) == simdjson::SUCCESS) {
                parameter.description = std::string(description);
            }

            simdjson::dom::element items_el;
            if (property_obj["items"].get(items_el) == simdjson::SUCCESS) {
                parameter.items_schema = simdjson::minify(items_el);
            }
        }

        if (parameter.type.empty()) {
            parameter.type = "string";
        }
        parameter.required = required_names.contains(parameter.name);
        out.push_back(std::move(parameter));
    }
}

[[nodiscard]] std::string extract_raw_json(simdjson::dom::element element) {
    return simdjson::minify(element);
}

[[nodiscard]] std::string extract_raw_json(simdjson::dom::object object) {
    return simdjson::minify(object);
}

bool parse_openai_tools(simdjson::dom::array tools_arr,
                        std::vector<core::llm::Tool>& out_tools,
                        std::string& error_out) {
    for (simdjson::dom::element tool_el : tools_arr) {
        simdjson::dom::object tool_obj;
        if (tool_el.get(tool_obj) != simdjson::SUCCESS) continue;

        std::string_view tool_type = "function";
        [[maybe_unused]] const auto ignored_type = tool_obj["type"].get(tool_type);
        if (tool_type != "function") continue;

        simdjson::dom::object function_obj;
        if (tool_obj["function"].get(function_obj) != simdjson::SUCCESS) {
            continue;
        }

        std::string_view function_name;
        if (function_obj["name"].get(function_name) != simdjson::SUCCESS
            || function_name.empty()) {
            error_out = "tools[].function.name must be a non-empty string";
            return false;
        }

        core::llm::Tool tool;
        tool.type = "function";
        tool.function.name = std::string(function_name);

        std::string_view description;
        if (function_obj["description"].get(description) == simdjson::SUCCESS) {
            tool.function.description = std::string(description);
        }

        simdjson::dom::object parameters_obj;
        if (function_obj["parameters"].get(parameters_obj) == simdjson::SUCCESS) {
            parse_schema_parameters(parameters_obj, tool.function.parameters);
        }

        out_tools.push_back(std::move(tool));
    }
    return true;
}

bool parse_anthropic_tools(simdjson::dom::array tools_arr,
                           std::vector<core::llm::Tool>& out_tools,
                           std::string& error_out) {
    for (simdjson::dom::element tool_el : tools_arr) {
        simdjson::dom::object tool_obj;
        if (tool_el.get(tool_obj) != simdjson::SUCCESS) continue;

        std::string_view function_name;
        if (tool_obj["name"].get(function_name) != simdjson::SUCCESS
            || function_name.empty()) {
            error_out = "tools[].name must be a non-empty string";
            return false;
        }

        core::llm::Tool tool;
        tool.type = "function";
        tool.function.name = std::string(function_name);

        std::string_view description;
        if (tool_obj["description"].get(description) == simdjson::SUCCESS) {
            tool.function.description = std::string(description);
        }

        simdjson::dom::object input_schema_obj;
        if (tool_obj["input_schema"].get(input_schema_obj) == simdjson::SUCCESS) {
            parse_schema_parameters(input_schema_obj, tool.function.parameters);
        }

        out_tools.push_back(std::move(tool));
    }
    return true;
}

void merge_tool_call_fragment(std::vector<core::llm::ToolCall>& accumulated,
                              const core::llm::ToolCall& incoming) {
    bool found = false;
    for (auto& existing : accumulated) {
        const bool same = (incoming.index != -1 && existing.index == incoming.index)
            || (incoming.index == -1 && !incoming.id.empty() && existing.id == incoming.id)
            || (incoming.index == -1
                && incoming.id.empty()
                && accumulated.size() == 1);
        if (!same) continue;

        if (!incoming.id.empty()) existing.id = incoming.id;
        if (!incoming.type.empty()) existing.type = incoming.type;
        if (!incoming.function.name.empty()) {
            existing.function.name = incoming.function.name;
        }
        existing.function.arguments += incoming.function.arguments;
        found = true;
        break;
    }
    if (!found) {
        accumulated.push_back(incoming);
    }
}

bool parse_openai_message(simdjson::dom::object message_obj,
                          std::vector<core::llm::Message>& out_messages,
                          std::string& error_out) {
    core::llm::Message message;

    std::string_view role;
    if (message_obj["role"].get(role) != simdjson::SUCCESS || role.empty()) {
        error_out = "messages[].role must be a non-empty string";
        return false;
    }
    message.role = std::string(role);

    std::string_view name;
    if (message_obj["name"].get(name) == simdjson::SUCCESS) {
        message.name = std::string(name);
    }

    std::string_view tool_call_id;
    if (message_obj["tool_call_id"].get(tool_call_id) == simdjson::SUCCESS) {
        message.tool_call_id = std::string(tool_call_id);
    }

    simdjson::dom::element content_el;
    if (message_obj["content"].get(content_el) == simdjson::SUCCESS) {
        const auto content_type = content_el.type();
        if (content_type == simdjson::dom::element_type::STRING) {
            std::string_view content_text;
            if (content_el.get(content_text) == simdjson::SUCCESS) {
                append_message_text(message, content_text);
            }
        } else if (content_type == simdjson::dom::element_type::ARRAY) {
            simdjson::dom::array content_arr;
            if (content_el.get(content_arr) == simdjson::SUCCESS) {
                for (simdjson::dom::element part_el : content_arr) {
                    simdjson::dom::object part_obj;
                    if (part_el.get(part_obj) != simdjson::SUCCESS) continue;

                    std::string_view part_type;
                    if (part_obj["type"].get(part_type) != simdjson::SUCCESS) continue;

                    if (part_type == "text") {
                        std::string_view text;
                        if (part_obj["text"].get(text) == simdjson::SUCCESS) {
                            append_message_text(message, text);
                        }
                        continue;
                    }

                    if (part_type == "image_url") {
                        simdjson::dom::object image_obj;
                        if (part_obj["image_url"].get(image_obj) != simdjson::SUCCESS) {
                            append_message_marker(message, "[Image input omitted]");
                            continue;
                        }

                        std::string_view image_url;
                        if (image_obj["url"].get(image_url) != simdjson::SUCCESS) {
                            append_message_marker(message, "[Image input omitted]");
                            continue;
                        }

                        std::string_view detail = "auto";
                        [[maybe_unused]] const auto ignored_detail = image_obj["detail"].get(detail);

                        std::string local_path;
                        if (image_url.starts_with("file://")) {
                            std::string parse_error;
                            if (auto resolved = parse_local_file_uri(image_url, parse_error);
                                resolved.has_value()) {
                                local_path = resolved->string();
                            }
                        } else if (!image_url.empty() && image_url.front() == '/') {
                            local_path = std::string(image_url);
                        }

                        if (!local_path.empty()) {
                            message.content_parts.push_back(core::llm::ContentPart::make_image(
                                std::move(local_path),
                                {},
                                std::string(detail)));
                            continue;
                        }

                        append_message_marker(
                            message,
                            "[Image input omitted: only local file:// or absolute-path image URLs are supported]");
                    }
                }
            }
        }
    }

    simdjson::dom::array tool_calls_arr;
    if (message_obj["tool_calls"].get(tool_calls_arr) == simdjson::SUCCESS) {
        for (simdjson::dom::element tool_call_el : tool_calls_arr) {
            simdjson::dom::object tool_call_obj;
            if (tool_call_el.get(tool_call_obj) != simdjson::SUCCESS) continue;

            core::llm::ToolCall call;
            int64_t index_value = -1;
            if (tool_call_obj["index"].get(index_value) == simdjson::SUCCESS) {
                call.index = static_cast<int>(index_value);
            }

            std::string_view call_id;
            if (tool_call_obj["id"].get(call_id) == simdjson::SUCCESS) {
                call.id = std::string(call_id);
            }

            std::string_view call_type;
            if (tool_call_obj["type"].get(call_type) == simdjson::SUCCESS) {
                call.type = std::string(call_type);
            }

            simdjson::dom::object function_obj;
            if (tool_call_obj["function"].get(function_obj) == simdjson::SUCCESS) {
                std::string_view function_name;
                if (function_obj["name"].get(function_name) == simdjson::SUCCESS) {
                    call.function.name = std::string(function_name);
                }

                std::string_view arguments_text;
                if (function_obj["arguments"].get(arguments_text) == simdjson::SUCCESS) {
                    call.function.arguments = std::string(arguments_text);
                } else {
                    simdjson::dom::element arguments_el;
                    if (function_obj["arguments"].get(arguments_el) == simdjson::SUCCESS) {
                        call.function.arguments = extract_raw_json(arguments_el);
                    }
                }
            }

            message.tool_calls.push_back(std::move(call));
        }
    }

    out_messages.push_back(std::move(message));
    return true;
}

bool parse_openai_messages(simdjson::dom::array messages_arr, std::vector<core::llm::Message>& out_messages, std::string& error_out) {
    for (simdjson::dom::element message_el : messages_arr) {
        simdjson::dom::object message_obj;
        if (message_el.get(message_obj) != simdjson::SUCCESS) {
            error_out = "messages[] entries must be JSON objects";
            return false;
        }
        if (!parse_openai_message(message_obj, out_messages, error_out)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string anthropic_content_to_text(simdjson::dom::element content_el) {
    const auto content_type = content_el.type();
    if (content_type == simdjson::dom::element_type::STRING) {
        std::string_view text;
        if (content_el.get(text) == simdjson::SUCCESS) {
            return std::string(text);
        }
        return {};
    }
    if (content_type == simdjson::dom::element_type::ARRAY) {
        std::string merged;
        simdjson::dom::array blocks;
        if (content_el.get(blocks) != simdjson::SUCCESS) return merged;
        for (simdjson::dom::element block_el : blocks) {
            simdjson::dom::object block_obj;
            if (block_el.get(block_obj) != simdjson::SUCCESS) continue;
            std::string_view block_type;
            if (block_obj["type"].get(block_type) != simdjson::SUCCESS) continue;
            if (block_type != "text") continue;
            std::string_view text;
            if (block_obj["text"].get(text) != simdjson::SUCCESS) continue;
            if (!merged.empty()) merged.push_back('\n');
            merged += std::string(text);
        }
        return merged;
    }
    return extract_raw_json(content_el);
}

bool parse_anthropic_message(simdjson::dom::object message_obj, std::vector<core::llm::Message>& out_messages, std::string& error_out) {
    std::string_view role;
    if (message_obj["role"].get(role) != simdjson::SUCCESS || role.empty()) {
        error_out = "messages[].role must be a non-empty string";
        return false;
    }

    simdjson::dom::element content_el;
    if (message_obj["content"].get(content_el) != simdjson::SUCCESS) {
        error_out = "messages[].content is required";
        return false;
    }

    core::llm::Message base_message;
    base_message.role = std::string(role);

    std::vector<core::llm::Message> trailing_messages;
    const auto content_type = content_el.type();
    if (content_type == simdjson::dom::element_type::STRING) {
        std::string_view text;
        if (content_el.get(text) == simdjson::SUCCESS) {
            append_message_text(base_message, text);
        }
    } else if (content_type == simdjson::dom::element_type::ARRAY) {
        simdjson::dom::array blocks;
        if (content_el.get(blocks) == simdjson::SUCCESS) {
            for (simdjson::dom::element block_el : blocks) {
                simdjson::dom::object block_obj;
                if (block_el.get(block_obj) != simdjson::SUCCESS) continue;

                std::string_view block_type;
                if (block_obj["type"].get(block_type) != simdjson::SUCCESS) continue;

                if (block_type == "text") {
                    std::string_view text;
                    if (block_obj["text"].get(text) == simdjson::SUCCESS) {
                        append_message_text(base_message, text);
                    }
                    continue;
                }

                if (block_type == "tool_use" && role == "assistant") {
                    core::llm::ToolCall call;
                    call.type = "function";

                    std::string_view call_id;
                    if (block_obj["id"].get(call_id) == simdjson::SUCCESS) {
                        call.id = std::string(call_id);
                    }

                    std::string_view name;
                    if (block_obj["name"].get(name) == simdjson::SUCCESS) {
                        call.function.name = std::string(name);
                    }

                    simdjson::dom::element input_el;
                    if (block_obj["input"].get(input_el) == simdjson::SUCCESS) {
                        call.function.arguments = extract_raw_json(input_el);
                    } else {
                        call.function.arguments = "{}";
                    }

                    base_message.tool_calls.push_back(std::move(call));
                    continue;
                }

                if (block_type == "tool_result") {
                    core::llm::Message tool_message;
                    tool_message.role = "tool";

                    std::string_view tool_use_id;
                    if (block_obj["tool_use_id"].get(tool_use_id) == simdjson::SUCCESS) {
                        tool_message.tool_call_id = std::string(tool_use_id);
                    }

                    simdjson::dom::element tool_content_el;
                    if (block_obj["content"].get(tool_content_el) == simdjson::SUCCESS) {
                        tool_message.content = anthropic_content_to_text(tool_content_el);
                    }

                    trailing_messages.push_back(std::move(tool_message));
                    continue;
                }

                if (block_type == "image") {
                    append_message_marker(
                        base_message,
                        "[Image input omitted: Anthropic base64 images are not yet translated by server mode]");
                }
            }
        }
    }

    const bool include_base = !base_message.content.empty()
        || !base_message.content_parts.empty()
        || !base_message.tool_calls.empty();

    if (include_base || base_message.role == "system") {
        out_messages.push_back(std::move(base_message));
    }
    for (auto& message : trailing_messages) {
        out_messages.push_back(std::move(message));
    }
    return true;
}

bool parse_anthropic_messages(simdjson::dom::array messages_arr, std::vector<core::llm::Message>& out_messages, std::string& error_out) {
    for (simdjson::dom::element message_el : messages_arr) {
        simdjson::dom::object message_obj;
        if (message_el.get(message_obj) != simdjson::SUCCESS) {
            error_out = "messages[] entries must be JSON objects";
            return false;
        }
        if (!parse_anthropic_message(message_obj, out_messages, error_out)) {
            return false;
        }
    }
    return true;
}

void parse_optional_generation_settings(simdjson::dom::object root_obj, core::llm::ChatRequest& request, bool& stream_out) {
    bool stream_value = false;
    [[maybe_unused]] const auto ignored_stream = root_obj["stream"].get(stream_value);
    stream_out = stream_value;
    request.stream = stream_value;

    double temperature = 0.0;
    if (root_obj["temperature"].get(temperature) == simdjson::SUCCESS) {
        request.temperature = static_cast<float>(temperature);
    }

    int64_t max_tokens = 0;
    if (root_obj["max_tokens"].get(max_tokens) == simdjson::SUCCESS) {
        request.max_tokens = static_cast<int>(max_tokens);
    } else if (root_obj["max_completion_tokens"].get(max_tokens) == simdjson::SUCCESS) {
        request.max_tokens = static_cast<int>(max_tokens);
    }
}

void parse_openai_stream_options(simdjson::dom::object root_obj, GatewayRequest& out_request) {
    simdjson::dom::object stream_options_obj;
    if (root_obj["stream_options"].get(stream_options_obj) != simdjson::SUCCESS) {
        return;
    }

    bool include_usage = false;
    if (stream_options_obj["include_usage"].get(include_usage) == simdjson::SUCCESS) {
        out_request.stream_include_usage = include_usage;
        out_request.request.stream_include_usage = include_usage;
    }
}

void parse_openai_response_format(simdjson::dom::object root_obj,
                                  core::llm::ChatRequest& request) {
    simdjson::dom::object response_format_obj;
    if (root_obj["response_format"].get(response_format_obj) != simdjson::SUCCESS) {
        return;
    }

    std::string_view type;
    if (response_format_obj["type"].get(type) != simdjson::SUCCESS) return;

    const std::string lowered = to_lower_ascii(type);
    if (lowered == "json_object") {
        request.response_format.type = core::llm::ResponseFormat::Type::JsonObject;
        return;
    }
    if (lowered != "json_schema") {
        return;
    }

    request.response_format.type = core::llm::ResponseFormat::Type::JsonSchema;

    simdjson::dom::element schema_el;
    if (response_format_obj["schema"].get(schema_el) == simdjson::SUCCESS) {
        request.response_format.schema = extract_raw_json(schema_el);
        return;
    }

    simdjson::dom::object json_schema_obj;
    if (response_format_obj["json_schema"].get(json_schema_obj) != simdjson::SUCCESS) {
        return;
    }

    simdjson::dom::element inner_schema_el;
    if (json_schema_obj["schema"].get(inner_schema_el) == simdjson::SUCCESS) {
        request.response_format.schema = extract_raw_json(inner_schema_el);
    } else {
        request.response_format.schema = extract_raw_json(json_schema_obj);
    }
}

bool parse_openai_gateway_request(const httplib::Request& http_request,
                                  GatewayRequest& out_request,
                                  std::string& error_out) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded_body(http_request.body);
    simdjson::dom::element doc;
    if (parser.parse(padded_body).get(doc) != simdjson::SUCCESS) {
        error_out = "Invalid JSON";
        return false;
    }

    simdjson::dom::object root_obj;
    if (doc.get(root_obj) != simdjson::SUCCESS) {
        error_out = "Request body must be a JSON object";
        return false;
    }

    std::string_view model;
    if (root_obj["model"].get(model) == simdjson::SUCCESS) {
        out_request.requested_model = std::string(model);
    }
    out_request.request.model = out_request.requested_model;

    parse_optional_generation_settings(root_obj, out_request.request, out_request.stream);
    parse_openai_stream_options(root_obj, out_request);
    parse_openai_response_format(root_obj, out_request.request);

    std::string_view previous_response_id;
    if (root_obj["previous_response_id"].get(previous_response_id) == simdjson::SUCCESS) {
        out_request.request.previous_response_id = std::string(previous_response_id);
    }

    std::string_view prompt_cache_key;
    if (root_obj["prompt_cache_key"].get(prompt_cache_key) == simdjson::SUCCESS) {
        out_request.request.prompt_cache_key = std::string(prompt_cache_key);
    }

    std::string_view service_tier;
    if (root_obj["service_tier"].get(service_tier) == simdjson::SUCCESS) {
        out_request.request.service_tier = std::string(service_tier);
    }

    simdjson::dom::array tools_arr;
    if (root_obj["tools"].get(tools_arr) == simdjson::SUCCESS) {
        if (!parse_openai_tools(tools_arr, out_request.request.tools, error_out)) {
            return false;
        }
    }

    simdjson::dom::array messages_arr;
    if (root_obj["messages"].get(messages_arr) != simdjson::SUCCESS) {
        error_out = "Missing required field: messages";
        return false;
    }
    if (!parse_openai_messages(messages_arr, out_request.request.messages, error_out)) {
        return false;
    }
    if (out_request.request.messages.empty()) {
        error_out = "messages must contain at least one entry";
        return false;
    }

    return true;
}

bool parse_anthropic_gateway_request(const httplib::Request& http_request,
                                     GatewayRequest& out_request,
                                     std::string& error_out) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded_body(http_request.body);
    simdjson::dom::element doc;
    if (parser.parse(padded_body).get(doc) != simdjson::SUCCESS) {
        error_out = "Invalid JSON";
        return false;
    }

    simdjson::dom::object root_obj;
    if (doc.get(root_obj) != simdjson::SUCCESS) {
        error_out = "Request body must be a JSON object";
        return false;
    }

    std::string_view model;
    if (root_obj["model"].get(model) == simdjson::SUCCESS) {
        out_request.requested_model = std::string(model);
    }
    out_request.request.model = out_request.requested_model;

    parse_optional_generation_settings(root_obj, out_request.request, out_request.stream);

    simdjson::dom::element system_el;
    if (root_obj["system"].get(system_el) == simdjson::SUCCESS) {
        const auto system_type = system_el.type();
        if (system_type == simdjson::dom::element_type::STRING) {
            std::string_view system_prompt;
            if (system_el.get(system_prompt) == simdjson::SUCCESS) {
                out_request.request.messages.push_back(core::llm::Message{
                    .role = "system",
                    .content = std::string(system_prompt),
                });
            }
        } else if (system_type == simdjson::dom::element_type::ARRAY) {
            std::string merged_system;
            simdjson::dom::array system_arr;
            if (system_el.get(system_arr) == simdjson::SUCCESS) {
                for (simdjson::dom::element block_el : system_arr) {
                    simdjson::dom::object block_obj;
                    if (block_el.get(block_obj) != simdjson::SUCCESS) continue;
                    std::string_view block_type;
                    if (block_obj["type"].get(block_type) != simdjson::SUCCESS) continue;
                    if (block_type != "text") continue;
                    std::string_view text;
                    if (block_obj["text"].get(text) != simdjson::SUCCESS) continue;
                    if (!merged_system.empty()) merged_system.push_back('\n');
                    merged_system += std::string(text);
                }
            }
            if (!merged_system.empty()) {
                out_request.request.messages.push_back(core::llm::Message{
                    .role = "system",
                    .content = std::move(merged_system),
                });
            }
        }
    }

    simdjson::dom::array tools_arr;
    if (root_obj["tools"].get(tools_arr) == simdjson::SUCCESS) {
        if (!parse_anthropic_tools(tools_arr, out_request.request.tools, error_out)) {
            return false;
        }
    }

    simdjson::dom::array messages_arr;
    if (root_obj["messages"].get(messages_arr) != simdjson::SUCCESS) {
        error_out = "Missing required field: messages";
        return false;
    }
    if (!parse_anthropic_messages(messages_arr, out_request.request.messages, error_out)) {
        return false;
    }

    if (out_request.request.messages.empty()) {
        error_out = "messages must contain at least one entry";
        return false;
    }

    return true;
}

[[nodiscard]] std::optional<GatewaySelection>
resolve_gateway_selection(const GatewayRuntime& runtime,
                          const GatewayRequest& gateway_request,
                          std::string& error_out) {
    const auto& config = runtime.config;
    auto& provider_manager = runtime.provider_manager;

    auto resolve_provider = [&](std::string_view provider_name,
                                GatewaySelection& selection) -> bool {
        try {
            selection.provider = provider_manager.get_provider(std::string(provider_name));
            selection.provider_name = std::string(provider_name);
            return true;
        } catch (const std::exception& e) {
            error_out = std::format("Provider '{}' is not available: {}",
                                    provider_name, e.what());
            return false;
        }
    };

    const std::string requested_model = trim_copy(gateway_request.requested_model);
    const bool has_requested_model = !requested_model.empty();

    GatewaySelection selection;

    if (requested_model.starts_with("policy/")) {
        if (!runtime.router_available) {
            error_out = "Router policies are unavailable. Configure config.router first.";
            return std::nullopt;
        }

        const std::string policy_name = requested_model.substr(7);
        if (policy_name.empty()) {
            error_out = "policy/<name> requires a non-empty policy name";
            return std::nullopt;
        }
        if (!runtime.router_engine || !runtime.router_engine->has_policy(policy_name)) {
            error_out = std::format("Unknown router policy '{}'", policy_name);
            return std::nullopt;
        }

        auto policy_router_config = config.router;
        policy_router_config.enabled = true;
        policy_router_config.default_policy = policy_name;

        auto policy_engine = std::make_shared<core::llm::routing::RouterEngine>(
            policy_router_config,
            runtime.providers);
        auto policy_provider = std::make_shared<core::llm::providers::RouterProvider>(
            provider_manager,
            std::move(policy_engine),
            runtime.provider_default_models);

        selection.provider = policy_provider;
        selection.ephemeral_router = policy_provider;
        selection.provider_name = "router";
        selection.model_override.clear();
        selection.uses_router = true;
        return selection;
    }

    if (requested_model == "router" || requested_model == "auto") {
        if (!runtime.router_available || !runtime.router_provider) {
            error_out = "Router is unavailable. Configure config.router with at least one policy.";
            return std::nullopt;
        }
        selection.provider = runtime.router_provider;
        selection.provider_name = "router";
        selection.model_override.clear();
        selection.uses_router = true;
        return selection;
    }

    if (has_requested_model) {
        const std::size_t slash = requested_model.find('/');
        if (slash != std::string::npos) {
            const std::string provider_name = requested_model.substr(0, slash);
            const std::string routed_model = requested_model.substr(slash + 1);
            if (core::llm::contains_provider(runtime.providers, provider_name)) {
                if (!resolve_provider(provider_name, selection)) return std::nullopt;
                if (!routed_model.empty()) {
                    selection.model_override = routed_model;
                } else if (const auto it = runtime.provider_default_models.find(provider_name);
                           it != runtime.provider_default_models.end()) {
                    selection.model_override = it->second;
                }
                selection.uses_router = false;
                return selection;
            }
        }

        if (core::llm::contains_provider(runtime.providers, requested_model)) {
            if (!resolve_provider(requested_model, selection)) return std::nullopt;
            if (const auto it = runtime.provider_default_models.find(requested_model);
                it != runtime.provider_default_models.end()) {
                selection.model_override = it->second;
            }
            selection.uses_router = false;
            return selection;
        }

        // Plain model id: send to default provider with requested model override.
        if (!resolve_provider(config.default_provider, selection)) {
            return std::nullopt;
        }
        selection.model_override = requested_model;
        selection.uses_router = false;
        return selection;
    }

    if (runtime.router_available
        && runtime.router_provider
        && prefer_router_by_default(config.default_model_selection)) {
        selection.provider = runtime.router_provider;
        selection.provider_name = "router";
        selection.model_override.clear();
        selection.uses_router = true;
        return selection;
    }

    if (!resolve_provider(config.default_provider, selection)) {
        if (runtime.router_available && runtime.router_provider) {
            selection.provider = runtime.router_provider;
            selection.provider_name = "router";
            selection.model_override.clear();
            selection.uses_router = true;
            return selection;
        }
        return std::nullopt;
    }

    if (const auto it = runtime.provider_default_models.find(config.default_provider);
        it != runtime.provider_default_models.end()) {
        selection.model_override = it->second;
    }
    selection.uses_router = false;
    return selection;
}

[[nodiscard]] CompletionResult run_completion_sync(
    const GatewaySelection& selection,
    const core::llm::ChatRequest& request,
    std::string_view response_id) {
    struct SyncState {
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
        bool had_error = false;
        std::string error_message;
        std::string content;
        std::vector<core::llm::ToolCall> tool_calls;
    };

    CompletionResult result;
    auto state = std::make_shared<SyncState>();

    try {
        selection.provider->stream_response(
            request,
            [state](const core::llm::StreamChunk& chunk) {
                std::lock_guard<std::mutex> lock(state->mutex);

                if (!chunk.content.empty()) {
                    state->content += chunk.content;
                }
                for (const auto& tool_call : chunk.tools) {
                    merge_tool_call_fragment(state->tool_calls, tool_call);
                }
                if (chunk.is_error) {
                    state->had_error = true;
                    if (!chunk.content.empty()) {
                        if (!state->error_message.empty()) state->error_message += '\n';
                        state->error_message += chunk.content;
                    }
                }
                if (chunk.is_final) {
                    state->done = true;
                    state->cv.notify_all();
                }
            });
    } catch (const std::exception& e) {
        result.has_error = true;
        result.error_message = std::format("Provider execution failed: {}", e.what());
        return result;
    } catch (...) {
        result.has_error = true;
        result.error_message = "Provider execution failed with an unknown error";
        return result;
    }

    {
        std::unique_lock<std::mutex> lock(state->mutex);
        constexpr auto kWaitTimeout = std::chrono::minutes{10};
        if (!state->cv.wait_for(lock, kWaitTimeout, [&]() { return state->done; })) {
            result.has_error = true;
            result.error_message = "Timed out while waiting for provider response";
            return result;
        }
        result.has_error = state->had_error;
        result.error_message = state->error_message;
        result.content = std::move(state->content);
        result.tool_calls = std::move(state->tool_calls);
    }

    result.usage = selection.provider->get_last_usage();
    result.model = selection.provider->get_last_model();
    if (result.model.empty()) {
        result.model = request.model;
    }
    if (result.usage.has_data()) {
        const bool should_estimate_cost = selection.provider->should_estimate_cost();
        core::budget::BudgetTracker::get_instance().record_event({
            .kind = core::budget::TokenLedgerEventKind::Actual,
            .source = core::budget::TokenLedgerSource::Gateway,
            .session_id = request.session_id,
            .request_id = std::string(response_id),
            .actor = "api_gateway",
            .provider = selection.provider_name,
            .model = result.model,
            .usage = result.usage,
            .should_estimate_cost = should_estimate_cost,
            .billable = should_estimate_cost,
        });
    }
    return result;
}

[[nodiscard]] std::string build_openai_error_body(std::string_view message,
                                                  std::string_view type = "invalid_request_error") {
    JsonWriter writer(256 + message.size());
    {
        auto _root = writer.object();
        writer.key("error");
        {
            auto _error = writer.object();
            writer.kv_str("message", message).comma()
                .kv_str("type", type);
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string build_anthropic_error_body(
    std::string_view message,
    std::string_view type = "invalid_request_error") {
    JsonWriter writer(256 + message.size());
    {
        auto _root = writer.object();
        writer.kv_str("type", "error").comma().key("error");
        {
            auto _error = writer.object();
            writer.kv_str("type", type).comma()
                .kv_str("message", message);
        }
    }
    return std::move(writer).take();
}

void write_openai_usage(JsonWriter& writer, const core::llm::TokenUsage& usage) {
    auto _usage = writer.object();
    writer.kv_num("prompt_tokens", usage.prompt_tokens).comma()
        .kv_num("completion_tokens", usage.completion_tokens).comma()
        .kv_num("total_tokens", usage.total_tokens);
}

[[nodiscard]] std::string normalize_tool_input_json(std::string_view arguments_json) {
    if (arguments_json.empty()) return "{}";

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(arguments_json).get(doc) != simdjson::SUCCESS) {
        return "{}";
    }
    if (doc.type() != simdjson::dom::element_type::OBJECT) {
        return "{}";
    }
    return simdjson::minify(doc);
}

[[nodiscard]] std::string build_openai_chat_completion_body(
    const CompletionResult& completion,
    std::string_view response_id) {
    const int64_t created = unix_timestamp_now();
    const std::string model = completion.model.empty()
        ? std::string("unknown")
        : completion.model;

    JsonWriter writer(1024 + completion.content.size());
    {
        auto _root = writer.object();
        writer.kv_str("id", response_id).comma()
            .kv_str("object", "chat.completion").comma()
            .kv_num("created", created).comma()
            .kv_str("model", model).comma()
            .key("choices");
        {
            auto _choices = writer.array();
            {
                auto _choice = writer.object();
                writer.kv_num("index", 0).comma().key("message");
                {
                    auto _message = writer.object();
                    writer.kv_str("role", "assistant").comma();
                    if (completion.content.empty()) {
                        writer.kv_null("content");
                    } else {
                        writer.kv_str("content", completion.content);
                    }

                    if (!completion.tool_calls.empty()) {
                        writer.comma().key("tool_calls");
                        {
                            auto _tool_calls = writer.array();
                            for (std::size_t i = 0; i < completion.tool_calls.size(); ++i) {
                                const auto& tool_call = completion.tool_calls[i];
                                if (i > 0) writer.comma();
                                auto _tool = writer.object();
                                const std::string fallback_id = std::format("call_{}", i + 1);
                                writer.kv_str("id", tool_call.id.empty() ? fallback_id : tool_call.id).comma()
                                    .kv_str("type", tool_call.type.empty() ? "function" : tool_call.type).comma()
                                    .key("function");
                                {
                                    auto _function = writer.object();
                                    writer.kv_str("name",
                                                  tool_call.function.name.empty()
                                                      ? "unknown_tool"
                                                      : tool_call.function.name).comma()
                                        .kv_str("arguments", tool_call.function.arguments);
                                }
                            }
                        }
                    }
                }
                writer.comma().kv_str(
                    "finish_reason",
                    completion.tool_calls.empty() ? "stop" : "tool_calls");
            }
        }

        writer.comma().key("usage");
        write_openai_usage(writer, completion.usage);
    }

    return std::move(writer).take();
}

[[nodiscard]] std::string build_anthropic_message_body(
    const CompletionResult& completion,
    std::string_view response_id) {
    const std::string model = completion.model.empty()
        ? std::string("unknown")
        : completion.model;

    JsonWriter writer(1024 + completion.content.size());
    {
        auto _root = writer.object();
        writer.kv_str("id", response_id).comma()
            .kv_str("type", "message").comma()
            .kv_str("role", "assistant").comma()
            .kv_str("model", model).comma()
            .key("content");
        {
            auto _content = writer.array();
            bool first_block = true;
            if (!completion.content.empty()) {
                auto _text = writer.object();
                writer.kv_str("type", "text").comma()
                    .kv_str("text", completion.content);
                first_block = false;
            }

            for (std::size_t i = 0; i < completion.tool_calls.size(); ++i) {
                const auto& tool_call = completion.tool_calls[i];
                if (!first_block) writer.comma();
                first_block = false;

                auto _tool_use = writer.object();
                const std::string fallback_id = std::format("toolu_{}", i + 1);
                writer.kv_str("type", "tool_use").comma()
                    .kv_str("id", tool_call.id.empty() ? fallback_id : tool_call.id).comma()
                    .kv_str("name",
                            tool_call.function.name.empty()
                                ? "unknown_tool"
                                : tool_call.function.name)
                    .comma()
                    .kv_raw("input", normalize_tool_input_json(tool_call.function.arguments));
            }

            if (first_block) {
                auto _text = writer.object();
                writer.kv_str("type", "text").comma().kv_str("text", "");
            }
        }

        writer.comma().kv_str(
            "stop_reason",
            completion.tool_calls.empty() ? "end_turn" : "tool_use").comma()
            .kv_null("stop_sequence").comma()
            .key("usage");
        {
            auto _usage = writer.object();
            writer.kv_num("input_tokens", completion.usage.prompt_tokens).comma()
                .kv_num("output_tokens", completion.usage.completion_tokens);
        }
    }

    return std::move(writer).take();
}

void request_model_discovery_if_needed(const GatewayRuntime& runtime,
                                       const core::llm::ProviderDescriptor& provider) {
    if (provider.is_local) {
        return;
    }

    const auto snapshot =
        core::llm::ModelCatalogAvailability::instance().snapshot(provider.name);
    if (!snapshot.refresh_due()) {
        return;
    }

    try {
        auto llm_provider = runtime.provider_manager.get_provider(provider.name);
        if (auto http_provider =
                std::dynamic_pointer_cast<core::llm::HttpLLMProvider>(llm_provider)) {
            http_provider->discover_models({.timeout_ms = 1000});
        }
    } catch (const std::exception& ex) {
        core::logging::debug(
            "Model discovery unavailable for provider '{}': {}",
            provider.name,
            ex.what());
    }
}

[[nodiscard]] std::string build_models_list_body(const GatewayRuntime& runtime,
                                                 std::string_view provider_filter = {}) {
    GatewayModelList models;
    const bool filtered = !provider_filter.empty();
    auto include_provider = [&](std::string_view provider_name) {
        return !filtered || provider_name == provider_filter;
    };
    auto add_model = [&](std::string id, std::string provider, bool discovered) {
        if (id.empty() || !include_provider(provider)) return;
        upsert_gateway_model_list_entry(
            models,
            std::move(id),
            std::move(provider),
            discovered);
    };

    for (const auto& provider : runtime.providers) {
        const auto& provider_name = provider.name;
        if (!include_provider(provider_name)) continue;

        request_model_discovery_if_needed(runtime, provider);

        add_model(provider_name, provider_name, false);
        if (const auto it = runtime.provider_default_models.find(provider_name);
            it != runtime.provider_default_models.end() && !it->second.empty()) {
            add_model(it->second, provider_name, false);
            add_model(provider_name + "/" + it->second, provider_name, false);
        }

        const auto snapshot =
            core::llm::ModelCatalogAvailability::instance().snapshot(provider_name);
        for (const auto& model : snapshot.models) {
            add_model(model.canonical_id, provider_name, true);
            add_model(provider_name + "/" + model.canonical_id, provider_name, true);
        }
    }

    if (!filtered && runtime.router_available && runtime.router_engine) {
        add_model("router", "filo", false);
        add_model("auto", "filo", false);
        for (const auto& policy_name : runtime.router_engine->list_policies()) {
            add_model("policy/" + policy_name, "filo", false);
        }
    }

    JsonWriter writer(512 + models.size() * 96);
    {
        auto _root = writer.object();
        writer.kv_str("object", "list").comma().key("data");
        {
            auto _data = writer.array();
            bool first = true;
            for (const auto& [_, model] : models) {
                if (!first) writer.comma();
                first = false;

                auto _entry = writer.object();
                writer.kv_str("id", model.id).comma()
                    .kv_str("object", "model").comma()
                    .kv_num("created", 0).comma()
                    .kv_str("owned_by", model.provider).comma()
                    .kv_str("filo_provider", model.provider).comma()
                    .kv_bool("filo_discovered", model.discovered);
            }
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string choose_stream_model_label(const GatewaySelection& selection,
                                                    const GatewayRequest& gateway_request) {
    if (!gateway_request.requested_model.empty()) {
        return gateway_request.requested_model;
    }
    if (!gateway_request.request.model.empty()) {
        return gateway_request.request.model;
    }
    if (!selection.model_override.empty()) {
        return selection.model_override;
    }
    if (!selection.provider_name.empty()) {
        return selection.provider_name;
    }
    return "unknown";
}

void stream_openai_chat_completion_response(const GatewaySelection& selection,
                                            const GatewayRequest& gateway_request,
                                            httplib::Response& res) {
    struct StreamState {
        exec::http::SseFrameQueue queue;
        std::atomic<bool> saw_tool_calls{false};
    };

    auto state = std::make_shared<StreamState>();
    const std::string response_id = make_gateway_id("chatcmpl");
    const int64_t created = unix_timestamp_now();
    const std::string model = choose_stream_model_label(selection, gateway_request);
    const bool include_usage = gateway_request.stream_include_usage;
    const auto provider = selection.provider;
    const std::string provider_name = selection.provider_name;
    const auto request = gateway_request.request;

    res.status = 200;
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [state,
         provider,
         request,
         response_id,
         created,
         model,
         provider_name,
         include_usage,
         started = false](std::size_t, httplib::DataSink& sink) mutable -> bool {
            if (started) {
                return true;
            }
            started = true;

            auto finish_with_error = [&](std::string_view message) {
                std::string trimmed_message = trim_copy(message);
                if (trimmed_message.empty()) {
                    trimmed_message = "Upstream provider error";
                }
                state->queue.push_data(exec::gateway::openai::error_body(trimmed_message));
                state->queue.push_data("[DONE]");
                state->queue.close();
            };

            const exec::gateway::openai::StreamChunkContext context{
                .response_id = response_id,
                .created = created,
                .model = model,
                .include_usage = include_usage,
            };
            state->queue.push_data(exec::gateway::openai::role_chunk(context));

            try {
                provider->stream_response(
                    request,
                    [state,
                     provider,
                     provider_name,
                     gateway_session_id = request.session_id,
                     response_id,
                     created,
                     model,
                     include_usage](const core::llm::StreamChunk& chunk) {
                        if (state->queue.is_closed()) {
                            return;
                        }

                        const exec::gateway::openai::StreamChunkContext chunk_context{
                            .response_id = response_id,
                            .created = created,
                            .model = model,
                            .include_usage = include_usage,
                        };

                        if (chunk.is_error) {
                            std::string message = trim_copy(chunk.content);
                            if (message.empty()) {
                                message = "Upstream provider error";
                            }
                            state->queue.push_data(exec::gateway::openai::error_body(message));
                            state->queue.push_data("[DONE]");
                            state->queue.close();
                            return;
                        }

                        if (!chunk.content.empty()) {
                            state->queue.push_data(
                                exec::gateway::openai::content_chunk(
                                    chunk_context,
                                    chunk.content));
                        }

                        if (!chunk.tools.empty()) {
                            state->saw_tool_calls.store(true, std::memory_order_relaxed);
                            state->queue.push_data(
                                exec::gateway::openai::tool_calls_chunk(
                                    chunk_context,
                                    chunk.tools));
                        }

                        if (chunk.is_final) {
                            const auto usage = provider->get_last_usage();
                            std::string actual_model = provider->get_last_model();
                            if (actual_model.empty()) {
                                actual_model = model;
                            }
                            if (usage.has_data()) {
                                const bool should_estimate_cost = provider->should_estimate_cost();
                                core::budget::BudgetTracker::get_instance().record_event({
                                    .kind = core::budget::TokenLedgerEventKind::Actual,
                                    .source = core::budget::TokenLedgerSource::Gateway,
                                    .session_id = gateway_session_id,
                                    .request_id = response_id,
                                    .actor = "api_gateway",
                                    .provider = provider_name,
                                    .model = actual_model,
                                    .usage = usage,
                                    .should_estimate_cost = should_estimate_cost,
                                    .billable = should_estimate_cost,
                                });
                            }
                            const std::string_view finish_reason =
                                state->saw_tool_calls.load(std::memory_order_relaxed)
                                    ? "tool_calls"
                                    : "stop";
                            state->queue.push_data(
                                exec::gateway::openai::finish_chunk(
                                    chunk_context,
                                    finish_reason));
                            if (include_usage) {
                                state->queue.push_data(
                                    exec::gateway::openai::usage_chunk(
                                        chunk_context,
                                        usage));
                            }
                            state->queue.push_data("[DONE]");
                            state->queue.close();
                        }
                    });
            } catch (const std::exception& e) {
                finish_with_error(std::format("Provider execution failed: {}", e.what()));
            } catch (...) {
                finish_with_error("Provider execution failed with an unknown error");
            }

            constexpr auto kWaitTimeout = std::chrono::minutes{10};
            const auto deadline = std::chrono::steady_clock::now() + kWaitTimeout;

            while (true) {
                std::deque<std::string> frames;
                const auto wait_result = state->queue.wait_pop_until(deadline, frames);
                if (wait_result == exec::http::SseFrameQueue::WaitResult::timeout) {
                    const auto error_frame = exec::http::sse_data_frame(
                        exec::gateway::openai::error_body(
                            "Timed out while waiting for provider response"));
                    if (!sink.write(error_frame.data(), error_frame.size())) {
                        return false;
                    }
                    const auto done_frame = exec::http::sse_data_frame("[DONE]");
                    if (!sink.write(done_frame.data(), done_frame.size())) {
                        return false;
                    }
                    sink.done();
                    return true;
                }

                for (const auto& frame : frames) {
                    if (!sink.write(frame.data(), frame.size())) {
                        return false;
                    }
                }

                if (wait_result == exec::http::SseFrameQueue::WaitResult::closed) {
                    sink.done();
                    return true;
                }
            }
        });
}

} // namespace

struct ApiGateway::Impl {
    Impl(const core::config::AppConfig& config,
         core::llm::ProviderManager& provider_manager,
         ProviderCatalog provider_catalog)
        : runtime(config, provider_manager, std::move(provider_catalog)) {
        runtime.router_engine = std::make_shared<core::llm::routing::RouterEngine>(
            config.router,
            runtime.providers);
        runtime.router_provider = std::make_shared<core::llm::providers::RouterProvider>(
            provider_manager,
            runtime.router_engine,
            runtime.provider_default_models);
        runtime.router_available =
            config.router.enabled && !runtime.router_engine->list_policies().empty();
    }

    void handle_openai_models(const httplib::Request& req, httplib::Response& res) const {
        std::string provider_filter;
        if (req.has_param("provider")) {
            provider_filter = req.get_param_value("provider");
        }
        res.status = 200;
        res.set_content(build_models_list_body(runtime, provider_filter), "application/json");
    }

    void handle_openai_chat_completions(const httplib::Request& req, httplib::Response& res) const {
        GatewayRequest gateway_request;
        std::string parse_error;
        if (!parse_openai_gateway_request(req, gateway_request, parse_error)) {
            res.status = 400;
            res.set_content(build_openai_error_body(parse_error), "application/json");
            return;
        }
        assign_gateway_session_scope(req, gateway_request);

        std::string selection_error;
        auto selection = resolve_gateway_selection(runtime, gateway_request, selection_error);
        if (!selection.has_value()) {
            res.status = 400;
            res.set_content(build_openai_error_body(selection_error), "application/json");
            return;
        }

        if (selection->uses_router) {
            gateway_request.request.model.clear();
        } else if (!selection->model_override.empty()) {
            gateway_request.request.model = selection->model_override;
        }

        if (gateway_request.stream) {
            stream_openai_chat_completion_response(*selection, gateway_request, res);
            return;
        }

        const std::string response_id = make_gateway_id("chatcmpl");
        CompletionResult completion =
            run_completion_sync(*selection, gateway_request.request, response_id);
        if (completion.has_error) {
            const std::string message = trim_copy(
                completion.error_message.empty()
                    ? std::string_view{"Upstream provider error"}
                    : std::string_view{completion.error_message});
            res.status = 502;
            res.set_content(build_openai_error_body(message, "api_error"), "application/json");
            return;
        }

        if (completion.model.empty()) {
            completion.model = gateway_request.request.model;
        }

        res.status = 200;
        res.set_content(
            build_openai_chat_completion_body(completion, response_id),
            "application/json");
    }

    void handle_anthropic_messages(const httplib::Request& req, httplib::Response& res) const {
        GatewayRequest gateway_request;
        std::string parse_error;
        if (!parse_anthropic_gateway_request(req, gateway_request, parse_error)) {
            res.status = 400;
            res.set_content(build_anthropic_error_body(parse_error), "application/json");
            return;
        }
        assign_gateway_session_scope(req, gateway_request);

        if (gateway_request.stream) {
            res.status = 400;
            res.set_content(
                build_anthropic_error_body(
                    "stream=true is not supported yet on /v1/messages"),
                "application/json");
            return;
        }

        std::string selection_error;
        auto selection = resolve_gateway_selection(runtime, gateway_request, selection_error);
        if (!selection.has_value()) {
            res.status = 400;
            res.set_content(build_anthropic_error_body(selection_error), "application/json");
            return;
        }

        if (selection->uses_router) {
            gateway_request.request.model.clear();
        } else if (!selection->model_override.empty()) {
            gateway_request.request.model = selection->model_override;
        }

        const std::string response_id = make_gateway_id("msg");
        CompletionResult completion =
            run_completion_sync(*selection, gateway_request.request, response_id);
        if (completion.has_error) {
            const std::string message = trim_copy(
                completion.error_message.empty()
                    ? std::string_view{"Upstream provider error"}
                    : std::string_view{completion.error_message});
            res.status = 502;
            res.set_content(build_anthropic_error_body(message, "api_error"), "application/json");
            return;
        }

        if (completion.model.empty()) {
            completion.model = gateway_request.request.model;
        }

        res.status = 200;
        res.set_content(
            build_anthropic_message_body(completion, response_id),
            "application/json");
    }

    GatewayRuntime runtime;
};

ApiGateway::ApiGateway(const core::config::AppConfig& config,
                       core::llm::ProviderManager& provider_manager,
                       ProviderCatalog provider_catalog)
    : impl_(std::make_shared<Impl>(config, provider_manager, std::move(provider_catalog))) {}

void ApiGateway::register_routes(httplib::Server& server) const
{
    auto impl = impl_;
    server.Get("/v1/models",
               [impl](const httplib::Request& req, httplib::Response& res) {
                   impl->handle_openai_models(req, res);
               });
    server.Post("/v1/chat/completions",
                [impl](const httplib::Request& req, httplib::Response& res) {
                    impl->handle_openai_chat_completions(req, res);
                });
    server.Post("/v1/messages",
                [impl](const httplib::Request& req, httplib::Response& res) {
                    impl->handle_anthropic_messages(req, res);
                });
}

} // namespace exec::gateway
