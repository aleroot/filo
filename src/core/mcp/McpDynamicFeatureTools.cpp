#include "McpDynamicFeatureTools.hpp"

#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include <simdjson.h>

#include <cctype>
#include <format>
#include <string_view>

namespace core::mcp {

namespace {

using core::utils::JsonWriter;

[[nodiscard]] core::tools::ToolAnnotations read_only_remote_annotations() {
    return core::tools::ToolAnnotations{
        .read_only_hint = true,
        .destructive_hint = false,
        .idempotent_hint = true,
        .open_world_hint = true,
    };
}

[[nodiscard]] std::string prefixed_description(std::string_view server_name,
                                               std::string_view description) {
    return std::format("[MCP:{}] {}", server_name, description);
}

[[nodiscard]] std::string require_string_argument(std::string_view json_args,
                                                  std::string_view key,
                                                  std::string_view tool_name) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json_args);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        throw std::runtime_error(
            std::format("{}: arguments must be valid JSON", tool_name));
    }

    simdjson::dom::object args;
    if (doc.get(args) != simdjson::SUCCESS) {
        throw std::runtime_error(
            std::format("{}: arguments must be a JSON object", tool_name));
    }

    std::string_view value;
    if (args[key].get(value) != simdjson::SUCCESS || value.empty()) {
        throw std::runtime_error(
            std::format("{}: missing required '{}' argument", tool_name, key));
    }

    return std::string(value);
}

struct ParsedPromptToolArgs {
    std::string name;
    std::string arguments_json{"{}"};
};

[[nodiscard]] ParsedPromptToolArgs parse_prompt_tool_args(std::string_view json_args,
                                                          std::string_view tool_name) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json_args);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        throw std::runtime_error(
            std::format("{}: arguments must be valid JSON", tool_name));
    }

    simdjson::dom::object args;
    if (doc.get(args) != simdjson::SUCCESS) {
        throw std::runtime_error(
            std::format("{}: arguments must be a JSON object", tool_name));
    }

    ParsedPromptToolArgs parsed;
    std::string_view name_sv;
    if (args["name"].get(name_sv) != simdjson::SUCCESS || name_sv.empty()) {
        throw std::runtime_error(
            std::format("{}: missing required 'name' argument", tool_name));
    }
    parsed.name = std::string(name_sv);

    simdjson::dom::element arguments_elem;
    if (args["arguments"].get(arguments_elem) == simdjson::SUCCESS) {
        simdjson::dom::object arguments_object;
        if (arguments_elem.get(arguments_object) != simdjson::SUCCESS) {
            throw std::runtime_error(
                std::format("{}: 'arguments' must be an object", tool_name));
        }
        parsed.arguments_json = simdjson::to_string(arguments_object);
    }

    return parsed;
}

} // namespace

std::string sanitize_mcp_tool_token(std::string_view value) {
    std::string out;
    out.reserve(value.size());

    bool previous_was_underscore = false;
    for (const unsigned char ch : value) {
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(std::tolower(ch)));
            previous_was_underscore = false;
            continue;
        }

        if (!previous_was_underscore) {
            out.push_back('_');
            previous_was_underscore = true;
        }
    }

    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back() == '_') out.pop_back();
    if (out.empty()) return "server";
    return out;
}

std::string make_mcp_feature_tool_name(std::string_view server_name,
                                       std::string_view suffix) {
    return "mcp_" + sanitize_mcp_tool_token(server_name) + "_" + std::string(suffix);
}

McpSessionBackedTool::McpSessionBackedTool(std::shared_ptr<IMcpClientSession> session,
                                           std::string server_name,
                                           core::tools::ToolDefinition def)
    : session_(std::move(session))
    , server_name_(std::move(server_name))
    , def_(std::move(def)) {}

std::string McpSessionBackedTool::make_error(std::string_view message) const {
    return std::format(
        R"({{"error":"{}"}})",
        core::utils::escape_json_string(message));
}

McpListResourcesTool::McpListResourcesTool(std::shared_ptr<IMcpClientSession> session,
                                           std::string server_name)
    : McpSessionBackedTool(
        std::move(session),
        server_name,
        core::tools::ToolDefinition{
            .name = make_mcp_feature_tool_name(server_name, "list_resources"),
            .title = std::format("List Resources ({})", server_name),
            .description = prefixed_description(
                server_name,
                "List resources exposed by this MCP server."),
            .parameters = {},
            .annotations = read_only_remote_annotations(),
        }) {}

std::string McpListResourcesTool::execute(
    const std::string&,
    const core::context::SessionContext&) {
    try {
        const auto resources = session_->list_resources();
        JsonWriter w(512 + resources.size() * 192);
        {
            auto _root = w.object();
            w.key("resources");
            {
                auto _arr = w.array();
                bool first = true;
                for (const auto& resource : resources) {
                    if (!first) w.comma();
                    first = false;
                    auto _item = w.object();
                    w.kv_str("uri", resource.uri);
                    if (!resource.name.empty()) {
                        w.comma().kv_str("name", resource.name);
                    }
                    if (!resource.title.empty()) {
                        w.comma().kv_str("title", resource.title);
                    }
                    if (!resource.description.empty()) {
                        w.comma().kv_str("description", resource.description);
                    }
                    if (!resource.mime_type.empty()) {
                        w.comma().kv_str("mimeType", resource.mime_type);
                    }
                }
            }
        }
        return std::move(w).take();
    } catch (const std::exception& e) {
        return make_error(std::format("list_resources failed: {}", e.what()));
    }
}

McpListResourceTemplatesTool::McpListResourceTemplatesTool(
    std::shared_ptr<IMcpClientSession> session,
    std::string server_name)
    : McpSessionBackedTool(
        std::move(session),
        server_name,
        core::tools::ToolDefinition{
            .name = make_mcp_feature_tool_name(server_name, "list_resource_templates"),
            .title = std::format("List Resource Templates ({})", server_name),
            .description = prefixed_description(
                server_name,
                "List parameterized resource templates exposed by this MCP server."),
            .parameters = {},
            .annotations = read_only_remote_annotations(),
        }) {}

std::string McpListResourceTemplatesTool::execute(
    const std::string&,
    const core::context::SessionContext&) {
    try {
        const auto templates = session_->list_resource_templates();
        JsonWriter w(512 + templates.size() * 192);
        {
            auto _root = w.object();
            w.key("resourceTemplates");
            {
                auto _arr = w.array();
                bool first = true;
                for (const auto& resource_template : templates) {
                    if (!first) w.comma();
                    first = false;
                    auto _item = w.object();
                    w.kv_str("uriTemplate", resource_template.uri_template);
                    if (!resource_template.name.empty()) {
                        w.comma().kv_str("name", resource_template.name);
                    }
                    if (!resource_template.title.empty()) {
                        w.comma().kv_str("title", resource_template.title);
                    }
                    if (!resource_template.description.empty()) {
                        w.comma().kv_str("description", resource_template.description);
                    }
                    if (!resource_template.mime_type.empty()) {
                        w.comma().kv_str("mimeType", resource_template.mime_type);
                    }
                }
            }
        }
        return std::move(w).take();
    } catch (const std::exception& e) {
        return make_error(std::format("list_resource_templates failed: {}", e.what()));
    }
}

McpReadResourceTool::McpReadResourceTool(std::shared_ptr<IMcpClientSession> session,
                                         std::string server_name)
    : McpSessionBackedTool(
        std::move(session),
        server_name,
        core::tools::ToolDefinition{
            .name = make_mcp_feature_tool_name(server_name, "read_resource"),
            .title = std::format("Read Resource ({})", server_name),
            .description = prefixed_description(
                server_name,
                "Read one MCP resource by URI."),
            .parameters = {
                core::tools::ToolParameter{
                    .name = "uri",
                    .type = "string",
                    .description = "The resource URI to read.",
                    .required = true,
                },
            },
            .annotations = read_only_remote_annotations(),
        }) {}

std::string McpReadResourceTool::execute(
    const std::string& json_args,
    const core::context::SessionContext&) {
    try {
        const std::string uri = require_string_argument(json_args, "uri", def_.name);
        return session_->read_resource(uri);
    } catch (const std::exception& e) {
        return make_error(std::format("read_resource failed: {}", e.what()));
    }
}

McpListPromptsTool::McpListPromptsTool(std::shared_ptr<IMcpClientSession> session,
                                       std::string server_name)
    : McpSessionBackedTool(
        std::move(session),
        server_name,
        core::tools::ToolDefinition{
            .name = make_mcp_feature_tool_name(server_name, "list_prompts"),
            .title = std::format("List Prompts ({})", server_name),
            .description = prefixed_description(
                server_name,
                "List prompts exposed by this MCP server."),
            .parameters = {},
            .annotations = read_only_remote_annotations(),
        }) {}

std::string McpListPromptsTool::execute(
    const std::string&,
    const core::context::SessionContext&) {
    try {
        const auto prompts = session_->list_prompts();
        JsonWriter w(512 + prompts.size() * 224);
        {
            auto _root = w.object();
            w.key("prompts");
            {
                auto _arr = w.array();
                bool first_prompt = true;
                for (const auto& prompt : prompts) {
                    if (!first_prompt) w.comma();
                    first_prompt = false;

                    auto _item = w.object();
                    w.kv_str("name", prompt.name);
                    if (!prompt.title.empty()) {
                        w.comma().kv_str("title", prompt.title);
                    }
                    if (!prompt.description.empty()) {
                        w.comma().kv_str("description", prompt.description);
                    }
                    if (!prompt.arguments.empty()) {
                        w.comma().key("arguments");
                        {
                            auto _args = w.array();
                            bool first_arg = true;
                            for (const auto& argument : prompt.arguments) {
                                if (!first_arg) w.comma();
                                first_arg = false;

                                auto _arg = w.object();
                                w.kv_str("name", argument.name);
                                if (!argument.title.empty()) {
                                    w.comma().kv_str("title", argument.title);
                                }
                                if (!argument.description.empty()) {
                                    w.comma().kv_str("description", argument.description);
                                }
                                w.comma().kv_bool("required", argument.required);
                            }
                        }
                    }
                }
            }
        }
        return std::move(w).take();
    } catch (const std::exception& e) {
        return make_error(std::format("list_prompts failed: {}", e.what()));
    }
}

McpGetPromptTool::McpGetPromptTool(std::shared_ptr<IMcpClientSession> session,
                                   std::string server_name)
    : McpSessionBackedTool(
        std::move(session),
        server_name,
        core::tools::ToolDefinition{
            .name = make_mcp_feature_tool_name(server_name, "get_prompt"),
            .title = std::format("Get Prompt ({})", server_name),
            .description = prefixed_description(
                server_name,
                "Resolve an MCP prompt by name with optional arguments."),
            .parameters = {
                core::tools::ToolParameter{
                    .name = "name",
                    .type = "string",
                    .description = "Prompt name to resolve.",
                    .required = true,
                },
                core::tools::ToolParameter{
                    .name = "arguments",
                    .type = "object",
                    .description = "Optional prompt arguments as a JSON object.",
                    .required = false,
                },
            },
            .annotations = read_only_remote_annotations(),
        }) {}

std::string McpGetPromptTool::execute(
    const std::string& json_args,
    const core::context::SessionContext&) {
    try {
        const auto parsed = parse_prompt_tool_args(json_args, def_.name);
        return session_->get_prompt(parsed.name, parsed.arguments_json);
    } catch (const std::exception& e) {
        return make_error(std::format("get_prompt failed: {}", e.what()));
    }
}

} // namespace core::mcp
