#include "McpSamplingBridge.hpp"

#include "../llm/Models.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include <simdjson.h>

#include <atomic>
#include <format>
#include <ranges>

namespace core::mcp {

namespace {

using core::utils::JsonWriter;

[[nodiscard]] std::string tool_content_to_text(simdjson::dom::element block);

[[nodiscard]] std::string sampling_content_to_text(simdjson::dom::element content) {
    std::string out;

    auto append_piece = [&out](std::string_view piece) {
        if (piece.empty()) return;
        if (!out.empty()) out.push_back('\n');
        out.append(piece);
    };

    auto parse_block = [&append_piece](simdjson::dom::element block) {
        std::string_view type;
        if (block["type"].get(type) != simdjson::SUCCESS) return;

        if (type == "text") {
            std::string_view text;
            if (block["text"].get(text) == simdjson::SUCCESS) {
                append_piece(text);
            }
            return;
        }

        if (type == "tool_result" || type == "tool_use") {
            append_piece(tool_content_to_text(block));
            return;
        }

        if (type == "resource") {
            std::string_view text;
            if (block["resource"]["text"].get(text) == simdjson::SUCCESS) {
                append_piece(text);
            } else {
                append_piece("[resource content omitted]");
            }
            return;
        }

        if (type == "image") {
            append_piece("[image content omitted]");
            return;
        }
        if (type == "audio") {
            append_piece("[audio content omitted]");
            return;
        }
    };

    simdjson::dom::array content_arr;
    if (content.get(content_arr) == simdjson::SUCCESS) {
        for (simdjson::dom::element block : content_arr) {
            parse_block(block);
        }
        return out;
    }

    parse_block(content);
    return out;
}

[[nodiscard]] std::string tool_content_to_text(simdjson::dom::element block) {
    std::string_view type;
    if (block["type"].get(type) != simdjson::SUCCESS) return {};

    if (type == "tool_use") {
        std::string_view name;
        if (block["name"].get(name) != simdjson::SUCCESS) name = "tool";
        simdjson::dom::element input;
        if (block["input"].get(input) == simdjson::SUCCESS) {
            return std::format("Tool request {} with input {}", name, simdjson::to_string(input));
        }
        return std::format("Tool request {}", name);
    }

    if (type == "tool_result") {
        std::string_view tool_use_id;
        [[maybe_unused]] const auto id_result = block["toolUseId"].get(tool_use_id);
        simdjson::dom::element result_content;
        if (block["content"].get(result_content) == simdjson::SUCCESS) {
            const std::string result_text = sampling_content_to_text(result_content);
            if (!tool_use_id.empty()) {
                return std::format("Tool result {}: {}", tool_use_id, result_text);
            }
            return std::format("Tool result: {}", result_text);
        }
        return "Tool result";
    }

    return {};
}

[[nodiscard]] std::vector<core::llm::Tool> parse_sampling_tools(
    simdjson::dom::element params_doc)
{
    std::vector<core::llm::Tool> tools;
    simdjson::dom::array tools_arr;
    if (params_doc["tools"].get(tools_arr) != simdjson::SUCCESS) return tools;

    for (simdjson::dom::element tool_elem : tools_arr) {
        core::llm::Tool tool;
        auto& def = tool.function;

        std::string_view name_v;
        if (tool_elem["name"].get(name_v) != simdjson::SUCCESS || name_v.empty()) {
            continue;
        }
        def.name = std::string(name_v);
        def.title = def.name;

        std::string_view desc_v;
        if (tool_elem["description"].get(desc_v) == simdjson::SUCCESS) {
            def.description = std::string(desc_v);
        }

        simdjson::dom::object input_schema;
        if (tool_elem["inputSchema"].get(input_schema) == simdjson::SUCCESS) {
            def.input_schema = simdjson::to_string(input_schema);

            std::vector<std::string> required_names;
            simdjson::dom::array required;
            if (input_schema["required"].get(required) == simdjson::SUCCESS) {
                for (simdjson::dom::element required_item : required) {
                    std::string_view name;
                    if (required_item.get(name) == simdjson::SUCCESS) {
                        required_names.emplace_back(name);
                    }
                }
            }

            simdjson::dom::object properties;
            if (input_schema["properties"].get(properties) == simdjson::SUCCESS) {
                for (auto [key, value] : properties) {
                    core::tools::ToolParameter parameter;
                    parameter.name = std::string(key);
                    parameter.schema = simdjson::to_string(value);
                    parameter.required = std::ranges::find(required_names, parameter.name)
                        != required_names.end();

                    std::string_view type_v;
                    if (value["type"].get(type_v) == simdjson::SUCCESS) {
                        parameter.type = std::string(type_v);
                    } else {
                        parameter.type = "string";
                    }

                    std::string_view param_desc_v;
                    if (value["description"].get(param_desc_v) == simdjson::SUCCESS) {
                        parameter.description = std::string(param_desc_v);
                    }

                    simdjson::dom::element items_elem;
                    if (value["items"].get(items_elem) == simdjson::SUCCESS) {
                        parameter.items_schema = simdjson::to_string(items_elem);
                    }

                    def.parameters.push_back(std::move(parameter));
                }
            }
        }

        tools.push_back(std::move(tool));
    }

    return tools;
}

struct StreamingSamplingResult {
    std::string text;
    std::vector<core::llm::ToolCall> tool_calls;
    std::string model;
    bool is_error = false;
    std::string error_message;
};

void merge_streamed_tool_call(std::vector<core::llm::ToolCall>& accumulator,
                              const core::llm::ToolCall& incoming) {
    for (auto& existing : accumulator) {
        const bool same_by_index = incoming.index != -1 && incoming.index == existing.index;
        const bool same_by_id = incoming.index == -1 && !incoming.id.empty() && incoming.id == existing.id;
        if (!same_by_index && !same_by_id) continue;

        if (!incoming.id.empty()) existing.id = incoming.id;
        if (!incoming.type.empty()) existing.type = incoming.type;
        if (!incoming.function.name.empty()) existing.function.name = incoming.function.name;
        existing.function.arguments += incoming.function.arguments;
        return;
    }

    accumulator.push_back(incoming);
}

[[nodiscard]] std::string tool_arguments_or_empty_object(std::string_view arguments_json) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(arguments_json);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return "{}";
    simdjson::dom::object object;
    if (doc.get(object) != simdjson::SUCCESS) return "{}";
    return std::string(simdjson::to_string(object));
}

} // namespace

McpSamplingBridge::McpSamplingBridge(std::shared_ptr<core::llm::LLMProvider> provider,
                                     std::string default_model)
    : provider_(std::move(provider))
    , default_model_(std::move(default_model)) {}

void McpSamplingBridge::set_backend(std::shared_ptr<core::llm::LLMProvider> provider,
                                    std::string default_model) {
    std::lock_guard lock(mutex_);
    provider_ = std::move(provider);
    default_model_ = std::move(default_model);
}

std::string McpSamplingBridge::create_message(std::string_view server_name,
                                              std::string_view params_json) const {
    std::lock_guard lock(mutex_);
    if (!provider_) {
        throw std::runtime_error(
            std::format("sampling bridge for '{}' has no provider configured", server_name));
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded_params(params_json);
    simdjson::dom::element params_doc;
    if (parser.parse(padded_params).get(params_doc) != simdjson::SUCCESS) {
        throw std::runtime_error(
            std::format("sampling/createMessage params are not valid JSON for '{}'",
                        server_name));
    }

    core::llm::ChatRequest request;
    request.model = default_model_;
    request.stream = true;

    int64_t max_tokens = 0;
    if (params_doc["maxTokens"].get(max_tokens) == simdjson::SUCCESS && max_tokens > 0) {
        request.max_tokens = static_cast<int>(max_tokens);
    }

    double temperature = 0.0;
    if (params_doc["temperature"].get(temperature) == simdjson::SUCCESS) {
        request.temperature = static_cast<float>(temperature);
    }

    std::string_view system_prompt;
    if (params_doc["systemPrompt"].get(system_prompt) == simdjson::SUCCESS
        && !system_prompt.empty()) {
        request.messages.push_back(
            core::llm::Message{
                .role = "system",
                .content = std::string(system_prompt),
            });
    }

    simdjson::dom::array messages_arr;
    if (params_doc["messages"].get(messages_arr) != simdjson::SUCCESS) {
        throw std::runtime_error("sampling/createMessage missing 'messages'");
    }

    for (simdjson::dom::element message_elem : messages_arr) {
        std::string_view role;
        if (message_elem["role"].get(role) != simdjson::SUCCESS || role.empty()) {
            continue;
        }

        simdjson::dom::element content_elem;
        if (message_elem["content"].get(content_elem) != simdjson::SUCCESS) {
            continue;
        }

        request.messages.push_back(core::llm::Message{
            .role = std::string(role),
            .content = sampling_content_to_text(content_elem),
        });
    }

    request.tools = parse_sampling_tools(params_doc);

    StreamingSamplingResult sampling_result;
    std::atomic_bool is_final_seen{false};
    provider_->stream_response(
        request,
        [&sampling_result, &is_final_seen](const core::llm::StreamChunk& chunk) {
            sampling_result.text += chunk.content;
            for (const auto& tool_call : chunk.tools) {
                merge_streamed_tool_call(sampling_result.tool_calls, tool_call);
            }

            if (!chunk.is_final) return;

            is_final_seen.store(true, std::memory_order_release);
            sampling_result.is_error = chunk.is_error;
            if (chunk.is_error) sampling_result.error_message = chunk.content;
        });

    if (!is_final_seen.load(std::memory_order_acquire)) {
        throw std::runtime_error("sampling/createMessage did not receive a final chunk");
    }

    if (sampling_result.is_error) {
        throw std::runtime_error(
            std::format("provider error while handling sampling: {}", sampling_result.error_message));
    }

    sampling_result.model = provider_->get_last_model();
    if (sampling_result.model.empty()) {
        sampling_result.model = request.model;
    }

    JsonWriter response_writer(1024 + sampling_result.text.size());
    {
        auto _root = response_writer.object();
        response_writer.kv_str("role", "assistant").comma();
        response_writer.key("content");
        if (!sampling_result.tool_calls.empty()) {
            auto _content = response_writer.array();
            bool first = true;
            if (!sampling_result.text.empty()) {
                auto _text = response_writer.object();
                response_writer.kv_str("type", "text").comma()
                    .kv_str("text", sampling_result.text);
                first = false;
            }
            for (const auto& tool_call : sampling_result.tool_calls) {
                if (!first) response_writer.comma();
                first = false;

                auto _tool_use = response_writer.object();
                response_writer.kv_str("type", "tool_use").comma()
                    .kv_str("id", tool_call.id).comma()
                    .kv_str("name", tool_call.function.name).comma()
                    .key("input")
                    .raw(tool_arguments_or_empty_object(tool_call.function.arguments));
            }
        } else {
            auto _content = response_writer.object();
            response_writer.kv_str("type", "text").comma()
                .kv_str("text", sampling_result.text);
        }

        if (!sampling_result.model.empty()) {
            response_writer.comma().kv_str("model", sampling_result.model);
        }
        response_writer.comma().kv_str(
            "stopReason",
            sampling_result.tool_calls.empty() ? "endTurn" : "toolUse");
    }

    return std::move(response_writer).take();
}

} // namespace core::mcp
