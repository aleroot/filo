#include "AcpProtocol.hpp"

#include "core/tools/ToolNames.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/utils/Base64.hpp"
#include "core/version/Version.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <initializer_list>
#include <utility>

namespace exec::acp {
namespace {

constexpr std::size_t kMaxDiffChars = 200 * 1024;
constexpr std::size_t kMaxImageBytes = 15 * 1024 * 1024;

[[nodiscard]] std::string compact(const core::utils::JsonValue& value) {
    return value.dump(0);
}

[[nodiscard]] std::string truncate(std::string_view text, std::size_t limit) {
    if (text.size() <= limit) return std::string(text);
    std::string out(text.substr(0, limit));
    out += "\n… truncated";
    return out;
}

[[nodiscard]] int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

[[nodiscard]] std::string percent_decode(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '%' && i + 2 < text.size()) {
            const int hi = hex_value(text[i + 1]);
            const int lo = hex_value(text[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>(hi * 16 + lo));
                i += 2;
                continue;
            }
        }
        out.push_back(text[i]);
    }
    return out;
}

[[nodiscard]] std::optional<std::filesystem::path> file_uri_to_path(std::string_view uri) {
    if (!uri.starts_with("file:")) return std::nullopt;
    std::string rest(uri.substr(5));
    if (rest.starts_with("//")) {
        rest.erase(0, 2);
        const auto slash = rest.find('/');
        if (slash == std::string::npos) return std::nullopt;
        const auto host = std::string_view(rest).substr(0, slash);
        if (!host.empty() && host != "localhost") return std::nullopt;
        rest.erase(0, slash);
    }
    if (rest.empty()) return std::nullopt;
    return std::filesystem::path(percent_decode(rest));
}

[[nodiscard]] const core::utils::JsonValue* field(
    const core::utils::JsonValue& object,
    std::string_view key) {
    return object.is_object() ? object.find(key) : nullptr;
}

[[nodiscard]] std::string field_string(const core::utils::JsonValue& object,
                                       std::string_view key) {
    const auto* value = field(object, key);
    if (!value || !value->is_string()) return {};
    return std::string(value->as_string());
}

void write_id(core::utils::JsonValue& message, const RpcId& id) {
    if (id.kind == RpcId::Kind::number) {
        message.set("id", core::utils::JsonValue::integer(id.number));
    } else if (id.kind == RpcId::Kind::string) {
        message.set("id", id.text);
    } else {
        message.set("id", core::utils::JsonValue{});
    }
}

[[nodiscard]] core::utils::JsonValue envelope(const RpcId* id, std::string_view method) {
    auto message = core::utils::JsonValue::object();
    message.set("jsonrpc", "2.0");
    if (id) write_id(message, *id);
    if (!method.empty()) message.set("method", std::string(method));
    return message;
}

[[nodiscard]] std::string first_argument(std::string_view args,
                                         std::initializer_list<std::string_view> keys) {
    const auto parsed = core::utils::JsonValue::parse(args);
    if (!parsed || !parsed->is_object()) return {};
    for (const auto key : keys) {
        if (auto value = field_string(*parsed, key); !value.empty()) return value;
    }
    return {};
}

[[nodiscard]] std::filesystem::path absolutize(std::string_view path,
                                               const std::filesystem::path& cwd) {
    std::filesystem::path file(path);
    if (file.empty() || file.is_absolute()) return file;
    return cwd / file;
}

[[nodiscard]] std::string image_extension(std::string_view mime) {
    if (mime.find("png") != std::string_view::npos) return ".png";
    if (mime.find("jpeg") != std::string_view::npos
        || mime.find("jpg") != std::string_view::npos) return ".jpg";
    if (mime.find("gif") != std::string_view::npos) return ".gif";
    if (mime.find("webp") != std::string_view::npos) return ".webp";
    return ".img";
}

[[nodiscard]] bool write_bytes(const std::filesystem::path& path, std::string_view bytes) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

void append_text(std::string& text, std::string_view extra) {
    if (extra.empty()) return;
    if (!text.empty() && text.back() != '\n') text.push_back('\n');
    text.append(extra);
}

[[nodiscard]] core::utils::JsonValue session_params(std::string_view session_id) {
    auto params = core::utils::JsonValue::object();
    params.set("sessionId", std::string(session_id));
    return params;
}

void read_env(const core::utils::JsonValue* node, std::vector<std::string>& env) {
    if (!node) return;
    // Spec: [{name, value}, ...]. Some editors still send a string map.
    if (node->is_object() && !node->is_array()) {
        for (const auto& [name, value] : node->members()) {
            if (!name.empty() && value.is_string()) {
                env.push_back(name + "=" + std::string(value.as_string()));
            }
        }
        return;
    }
    if (!node->is_array()) return;
    for (const auto& variable : node->items()) {
        if (!variable.is_object()) continue;
        auto name = field_string(variable, "name");
        if (name.empty()) continue;
        env.push_back(name + "=" + field_string(variable, "value"));
    }
}

void read_headers(const core::utils::JsonValue* node,
                  std::vector<std::pair<std::string, std::string>>& headers) {
    if (!node) return;
    if (node->is_object() && !node->is_array()) {
        for (const auto& [name, value] : node->members()) {
            if (!name.empty() && value.is_string()) {
                headers.emplace_back(name, std::string(value.as_string()));
            }
        }
        return;
    }
    if (!node->is_array()) return;
    for (const auto& header : node->items()) {
        if (!header.is_object()) continue;
        auto name = field_string(header, "name");
        if (name.empty()) continue;
        headers.emplace_back(std::move(name), field_string(header, "value"));
    }
}

} // namespace

std::string RpcId::key() const {
    if (kind == Kind::number) return "n:" + std::to_string(number);
    if (kind == Kind::string) return "s:" + text;
    return {};
}

RpcId rpc_id(const core::utils::JsonValue* value) {
    RpcId id;
    if (!value || value->is_null()) return id;
    if (value->is_string()) {
        id.kind = RpcId::Kind::string;
        id.text = std::string(value->as_string());
        return id;
    }
    if (const auto number = value->as_integer()) {
        id.kind = RpcId::Kind::number;
        id.number = *number;
    }
    return id;
}

RpcMessage parse_message(std::string_view line) {
    RpcMessage message;
    std::string error;
    auto root = core::utils::JsonValue::parse(line, &error);
    if (!root || !root->is_object()) {
        message.parse_error = error.empty() ? "expected a JSON object" : error;
        return message;
    }
    message.id = rpc_id(root->find("id"));
    if (const auto* method = root->find("method"); method && method->is_string()) {
        message.method = std::string(method->as_string());
        message.kind = message.id.kind == RpcId::Kind::none
            ? RpcMessage::Kind::notification
            : RpcMessage::Kind::request;
    } else if (root->find("result") || root->find("error")) {
        message.kind = RpcMessage::Kind::response;
    } else {
        message.parse_error = "missing method";
        return message;
    }
    if (const auto* params = root->find("params")) {
        message.params = *params;
    } else {
        message.params = core::utils::JsonValue::object();
    }
    if (const auto* result = root->find("result")) message.result = *result;
    if (const auto* err = root->find("error"); err && err->is_object()) {
        message.has_error = true;
        if (const auto code = err->find("code"); code) {
            if (const auto number = code->as_integer()) message.error_code = static_cast<int>(*number);
        }
        message.error_message = field_string(*err, "message");
    }
    return message;
}

std::string encode_result(const RpcId& id, const core::utils::JsonValue& result) {
    auto message = envelope(&id, {});
    message.set("result", result);
    return compact(message);
}

std::string encode_error(const RpcId& id,
                         int code,
                         std::string_view message,
                         const core::utils::JsonValue* data) {
    auto envelope_value = envelope(&id, {});
    auto error = core::utils::JsonValue::object();
    error.set("code", core::utils::JsonValue::integer(code));
    error.set("message", std::string(message));
    if (data && !data->is_null()) error.set("data", *data);
    envelope_value.set("error", std::move(error));
    return compact(envelope_value);
}

std::string encode_notification(std::string_view method, const core::utils::JsonValue& params) {
    auto message = envelope(nullptr, method);
    message.set("params", params);
    return compact(message);
}

std::string encode_request(const RpcId& id,
                           std::string_view method,
                           const core::utils::JsonValue& params) {
    auto message = envelope(&id, method);
    message.set("params", params);
    return compact(message);
}

std::optional<int> negotiated_protocol_version(const core::utils::JsonValue& params,
                                             std::string* error) {
    const auto* version = params.is_object() ? params.find("protocolVersion") : nullptr;
    const auto requested = version ? version->as_integer() : std::nullopt;
    if (!requested || *requested < 0 || *requested > 65535) {
        if (error) *error = "protocolVersion must be an integer from 0 to 65535";
        return std::nullopt;
    }
    // Echo the client's version only when it is one we speak. Otherwise the
    // client is told our latest and disconnects if it cannot use that.
    return kProtocolVersion;
}

ClientCapabilities parse_client_capabilities(const core::utils::JsonValue& params) {
    // A capability is present only as a non-null value; `null` means absent.
    const auto present = [](const core::utils::JsonValue* value) {
        return value != nullptr && !value->is_null();
    };
    ClientCapabilities caps;
    const auto* client = field(params, "clientCapabilities");
    if (!client) return caps;
    if (const auto* session = field(*client, "session"); present(session)) {
        caps.notices = present(field(*session, "notices"));
    }
    if (const auto* elicitation = field(*client, "elicitation"); present(elicitation)) {
        caps.elicitation_form = present(field(*elicitation, "form"));
    }
    if (const auto* auth = field(*client, "auth"); present(auth)) {
        const auto* terminal = field(*auth, "terminal");
        caps.terminal_auth = terminal && terminal->is_bool() && terminal->as_bool();
    }
    return caps;
}

core::utils::JsonValue initialize_result(int protocol_version, std::string_view login_provider) {
    auto result = core::utils::JsonValue::object();
    result.set("protocolVersion", core::utils::JsonValue::integer(protocol_version));

    auto capabilities = core::utils::JsonValue::object();
    capabilities.set("loadSession", true);
    auto prompt = core::utils::JsonValue::object();
    prompt.set("image", true);
    prompt.set("audio", false);
    prompt.set("embeddedContext", true);
    capabilities.set("promptCapabilities", std::move(prompt));
    auto mcp = core::utils::JsonValue::object();
    mcp.set("http", true);
    mcp.set("sse", false);
    capabilities.set("mcpCapabilities", std::move(mcp));
    // Advertised only because session/new and session/load honor the field.
    // Editors omit it unless the agent says it is supported.
    auto session = core::utils::JsonValue::object();
    session.set("additionalDirectories", core::utils::JsonValue::object());
    session.set("resume", core::utils::JsonValue::object());
    session.set("close", core::utils::JsonValue::object());
    session.set("list", core::utils::JsonValue::object());
    session.set("delete", core::utils::JsonValue::object());
    capabilities.set("sessionCapabilities", std::move(session));
    result.set("agentCapabilities", std::move(capabilities));

    auto info = core::utils::JsonValue::object();
    info.set("name", std::string(core::version::product_name));
    info.set("title", "Filo");
    info.set("version", std::string(core::version::value));
    result.set("agentInfo", std::move(info));
    auto auth_methods = core::utils::JsonValue::array();
    if (!login_provider.empty()) {
        // The editor reruns this program with these arguments appended; --auth
        // signs in and exits before --acp is considered.
        auto login = core::utils::JsonValue::object();
        login.set("id", "filo-login");
        login.set("name", "Sign in to " + std::string(login_provider));
        login.set("description", "Runs Filo's sign-in for this provider in a terminal");
        login.set("type", "terminal");
        auto args = core::utils::JsonValue::array();
        args.items().emplace_back("--auth");
        args.items().emplace_back(std::string(login_provider));
        login.set("args", std::move(args));
        auth_methods.items().push_back(std::move(login));
    }
    result.set("authMethods", std::move(auth_methods));
    return result;
}

std::optional<PromptBlocks> parse_prompt(const core::utils::JsonValue& prompt,
                                         const std::filesystem::path& media_dir,
                                         std::string* error) {
    if (!prompt.is_array()) {
        if (error) *error = "prompt must be an array of content blocks";
        return std::nullopt;
    }
    PromptBlocks blocks;
    int image_index = 0;
    for (const auto& block : prompt.items()) {
        if (!block.is_object()) continue;
        const auto type = field_string(block, "type");
        if (type == "text") {
            blocks.text += field_string(block, "text");
            continue;
        }
        if (type == "resource") {
            const auto* resource = field(block, "resource");
            if (!resource || !resource->is_object()) continue;
            const auto uri = field_string(*resource, "uri");
            const auto text = field_string(*resource, "text");
            if (!text.empty()) {
                append_text(blocks.text, uri.empty() ? text : "[" + uri + "]\n" + text);
            } else if (!uri.empty()) {
                append_text(blocks.text, "[resource omitted: " + uri + "]");
            }
            continue;
        }
        if (type == "resource_link") {
            const auto uri = field_string(block, "uri");
            const auto name = field_string(block, "name");
            if (auto path = file_uri_to_path(uri)) {
                blocks.grant_paths.push_back(*path);
                append_text(blocks.text, "[file] " + path->string());
            } else if (!name.empty() || !uri.empty()) {
                append_text(blocks.text, "[link] " + (name.empty() ? uri : name));
            }
            continue;
        }
        if (type == "image") {
            const auto mime = field_string(block, "mimeType");
            const auto data = field_string(block, "data");
            auto decoded = core::utils::Base64::decode(data);
            if (!decoded || decoded->size() > kMaxImageBytes) {
                append_text(blocks.text, "[image omitted]");
                continue;
            }
            const auto path = media_dir / ("image-" + std::to_string(++image_index) + image_extension(mime));
            if (!write_bytes(path, *decoded)) {
                append_text(blocks.text, "[image omitted]");
                continue;
            }
            blocks.parts.push_back(core::llm::ContentPart::make_image(path.string(), mime));
            continue;
        }
        if (type == "audio") {
            append_text(blocks.text, "[audio omitted]");
            continue;
        }
        if (!type.empty()) append_text(blocks.text, "[unsupported content: " + type + "]");
    }
    return blocks;
}

std::optional<std::vector<core::config::McpServerConfig>> parse_mcp_servers(
    const core::utils::JsonValue* servers,
    std::string* error) {
    std::vector<core::config::McpServerConfig> out;
    if (!servers || servers->is_null()) return out;
    if (!servers->is_array()) {
        if (error) *error = "mcpServers must be an array";
        return std::nullopt;
    }
    for (const auto& entry : servers->items()) {
        if (!entry.is_object()) {
            if (error) *error = "mcpServers entries must be objects";
            return std::nullopt;
        }
        core::config::McpServerConfig server;
        server.name = field_string(entry, "name");
        if (server.name.empty()) {
            if (error) *error = "an MCP server is missing a name";
            return std::nullopt;
        }
        const auto type = field_string(entry, "type");
        if (type == "sse") {
            if (error) *error = "MCP server '" + server.name + "' uses unsupported SSE transport";
            return std::nullopt;
        }
        if (type == "http") {
            server.transport = "http";
            server.url = field_string(entry, "url");
            if (server.url.empty()) {
                if (error) *error = "MCP server '" + server.name + "' is missing a url";
                return std::nullopt;
            }
            read_headers(field(entry, "headers"), server.headers);
        } else if (type.empty() || type == "stdio") {
            server.transport = "stdio";
            server.command = field_string(entry, "command");
            if (server.command.empty()) {
                if (error) *error = "MCP server '" + server.name + "' is missing a command";
                return std::nullopt;
            }
            if (const auto* args = field(entry, "args"); args && args->is_array()) {
                for (const auto& arg : args->items()) {
                    if (arg.is_string()) server.args.emplace_back(arg.as_string());
                }
            }
            read_env(field(entry, "env"), server.env);
        } else {
            if (error) *error = "MCP server '" + server.name + "' has an unknown transport";
            return std::nullopt;
        }
        out.push_back(std::move(server));
    }
    return out;
}

std::string tool_kind(std::string_view tool_name) {
    using namespace core::tools::names;
    const auto name = canonical_alias(tool_name);
    if (is_read_tool(name)) return "read";
    if (name == kDeleteFile) return "delete";
    if (name == kMoveFile) return "move";
    if (is_file_modification_tool(name)) return "edit";
    if (name == kFileSearch || name == kGrepSearch || name == kListDirectory) return "search";
    if (is_terminal_tool(name) || is_verification_tool(name) || name == kPython) return "execute";
    if (is_web_access_tool(name)) return "fetch";
    if (name == kWriteTodos) return "think";
    return "other";
}

std::string tool_title(std::string_view tool_name, std::string_view args) {
    if (core::tools::names::canonical_alias(tool_name) == core::tools::names::kAskUserQuestion) {
        return "Ask the user";
    }
    const auto kind = tool_kind(tool_name);
    const auto target = first_argument(args, {"command", "file_path", "path", "dir_path", "url", "query", "pattern"});
    if (target.empty()) return std::string(tool_name);
    std::string shown = target;
    if (kind == "execute") {
        const auto newline = shown.find('\n');
        if (newline != std::string::npos) shown.resize(newline);
        if (shown.size() > 120) shown.resize(120);
        return "Run " + shown;
    }
    const auto slash = shown.find_last_of("/\\");
    if (slash != std::string::npos && slash + 1 < shown.size()) shown.erase(0, slash + 1);
    if (kind == "read") return "Read " + shown;
    if (kind == "edit") return "Edit " + shown;
    if (kind == "delete") return "Delete " + shown;
    if (kind == "move") return "Move " + shown;
    if (kind == "search") return "Search " + shown;
    if (kind == "fetch") return "Fetch " + shown;
    return std::string(tool_name) + " " + shown;
}

std::vector<std::filesystem::path> tool_locations(std::string_view args,
                                                  const std::filesystem::path& cwd) {
    std::vector<std::filesystem::path> locations;
    for (const auto key : {"file_path", "path", "dir_path", "source", "destination"}) {
        auto value = first_argument(args, {key});
        if (value.empty()) continue;
        auto path = absolutize(value, cwd);
        if (std::ranges::find(locations, path) == locations.end()) {
            locations.push_back(std::move(path));
        }
    }
    return locations;
}

FileSide read_file_side(const std::filesystem::path& path) {
    std::error_code ec;
    const auto status = std::filesystem::status(path, ec);
    if (ec || !std::filesystem::exists(status)) {
        // Nothing there: the tool created it, or deleted it.
        return FileSide{.readable = true};
    }
    if (!std::filesystem::is_regular_file(status)) return {};
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size > kMaxDiffChars) return {};
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::string text(static_cast<std::size_t>(size), '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(in.gcount()));
    // A NUL byte means binary content an editor cannot render as text.
    if (text.find('\0') != std::string::npos) return {};
    return FileSide{.readable = true, .text = std::move(text)};
}

std::optional<core::utils::JsonValue> file_diff(const std::filesystem::path& path,
                                                const FileSide& before,
                                                const FileSide& after) {
    if (!path.is_absolute() || !before.readable || !after.readable) return std::nullopt;
    if (!before.text && !after.text) return std::nullopt;
    if (before.text == after.text) return std::nullopt;
    auto diff = core::utils::JsonValue::object();
    diff.set("type", "diff");
    diff.set("path", path.lexically_normal().string());
    if (before.text) diff.set("oldText", *before.text);
    else diff.set("oldText", core::utils::JsonValue{});
    diff.set("newText", after.text.value_or(std::string{}));
    return diff;
}

bool tool_result_failed(std::string_view result) {
    const auto parsed = core::utils::JsonValue::parse(result);
    if (!parsed || !parsed->is_object()) return false;
    if (const auto* error = parsed->find("error");
        error && error->is_string() && !error->as_string().empty()) {
        return true;
    }
    if (const auto* ok = parsed->find("ok"); ok && ok->is_bool() && !ok->as_bool()) return true;
    if (const auto* success = parsed->find("success");
        success && success->is_bool() && !success->as_bool()) {
        return true;
    }
    return false;
}

std::string tool_display_text(std::string_view tool_name, std::string_view result) {
    const auto parsed = core::utils::JsonValue::parse(result);
    if (!parsed || !parsed->is_object()) return std::string(result);
    if (const auto* error = parsed->find("error");
        error && error->is_string() && !error->as_string().empty()) {
        return std::string(error->as_string());
    }
    if (tool_kind(tool_name) != "execute") return std::string(result);
    const auto* output = parsed->find("output");
    if (!output || !output->is_string()) return std::string(result);
    std::string text(output->as_string());
    if (const auto* code = parsed->find("exit_code")) {
        if (const auto value = code->as_integer(); value && *value != 0) {
            append_text(text, std::format("[exit code {}]", *value));
        }
    }
    return text;
}

PermissionDecision interpret_permission(const core::utils::JsonValue& result,
                                        std::string_view allow_always_id) {
    PermissionDecision decision;
    const auto* outcome = field(result, "outcome");
    if (!outcome || !outcome->is_object()) return decision;
    const auto kind = field_string(*outcome, "outcome");
    if (kind == "cancelled") {
        decision.cancelled = true;
        return decision;
    }
    if (kind != "selected") return decision;
    const auto option = field_string(*outcome, "optionId");
    if (option == "allow-once") {
        decision.allowed = true;
        return decision;
    }
    if (!allow_always_id.empty() && option == allow_always_id) {
        decision.allowed = true;
        decision.remember = true;
        return decision;
    }
    return decision;
}

core::utils::JsonValue text_update(std::string_view session_id,
                                   std::string_view kind,
                                   std::string_view text) {
    auto params = session_params(session_id);
    auto update = core::utils::JsonValue::object();
    update.set("sessionUpdate", std::string(kind));
    auto content = core::utils::JsonValue::object();
    content.set("type", "text");
    content.set("text", std::string(text));
    update.set("content", std::move(content));
    params.set("update", std::move(update));
    return params;
}

core::utils::JsonValue mode_state(std::string_view current,
                                  const std::vector<ModeChoice>& modes) {
    auto state = core::utils::JsonValue::object();
    state.set("currentModeId", std::string(current));
    auto available = core::utils::JsonValue::array();
    for (const auto& mode : modes) {
        auto entry = core::utils::JsonValue::object();
        entry.set("id", mode.id);
        entry.set("name", mode.name);
        if (!mode.description.empty()) entry.set("description", mode.description);
        available.items().push_back(std::move(entry));
    }
    state.set("availableModes", std::move(available));
    return state;
}

core::utils::JsonValue config_options(const std::vector<ConfigSelect>& options) {
    auto list = core::utils::JsonValue::array();
    for (const auto& option : options) {
        auto entry = core::utils::JsonValue::object();
        entry.set("id", option.id);
        entry.set("name", option.name);
        if (!option.description.empty()) entry.set("description", option.description);
        if (!option.category.empty()) entry.set("category", option.category);
        entry.set("type", "select");
        entry.set("currentValue", option.current);
        auto choices = core::utils::JsonValue::array();
        for (const auto& choice : option.options) {
            auto item = core::utils::JsonValue::object();
            item.set("value", choice.value);
            item.set("name", choice.name);
            if (!choice.description.empty()) item.set("description", choice.description);
            choices.items().push_back(std::move(item));
        }
        entry.set("options", std::move(choices));
        list.items().push_back(std::move(entry));
    }
    return list;
}

core::utils::JsonValue commands_update(std::string_view session_id,
                                       const std::vector<CommandSpec>& commands) {
    auto params = session_params(session_id);
    auto update = core::utils::JsonValue::object();
    update.set("sessionUpdate", "available_commands_update");
    auto list = core::utils::JsonValue::array();
    for (const auto& command : commands) {
        auto entry = core::utils::JsonValue::object();
        entry.set("name", command.name);
        entry.set("description", command.description);
        list.items().push_back(std::move(entry));
    }
    update.set("availableCommands", std::move(list));
    params.set("update", std::move(update));
    return params;
}

core::utils::JsonValue plan_update(std::string_view session_id,
                                   const std::vector<PlanItem>& entries) {
    auto params = session_params(session_id);
    auto update = core::utils::JsonValue::object();
    update.set("sessionUpdate", "plan");
    auto list = core::utils::JsonValue::array();
    for (const auto& entry : entries) {
        auto item = core::utils::JsonValue::object();
        item.set("content", entry.content);
        item.set("priority", entry.priority);
        item.set("status", entry.status);
        list.items().push_back(std::move(item));
    }
    update.set("entries", std::move(list));
    params.set("update", std::move(update));
    return params;
}

core::utils::JsonValue model_state(const ConfigSelect& model) {
    auto state = core::utils::JsonValue::object();
    state.set("currentModelId", model.current);
    auto available = core::utils::JsonValue::array();
    for (const auto& choice : model.options) {
        auto entry = core::utils::JsonValue::object();
        entry.set("modelId", choice.value);
        entry.set("name", choice.name);
        if (!choice.description.empty()) entry.set("description", choice.description);
        available.items().push_back(std::move(entry));
    }
    state.set("availableModels", std::move(available));
    return state;
}

core::utils::JsonValue question_form(std::string_view session_id,
                                     std::string_view tool_call_id,
                                     const std::vector<core::tools::QuestionItem>& questions) {
    auto params = session_params(session_id);
    if (!tool_call_id.empty()) params.set("toolCallId", std::string(tool_call_id));
    params.set("mode", "form");
    params.set("message", questions.size() == 1 ? questions.front().question
                                                : std::format("Filo has {} questions.", questions.size()));
    auto properties = core::utils::JsonValue::object();
    for (std::size_t i = 0; i < questions.size(); ++i) {
        const auto& question = questions[i];
        const auto key = std::format("q{}", i + 1);
        auto labels = core::utils::JsonValue::array();
        std::string description = question.question;
        if (!question.body.empty()) description += "\n\n" + question.body;
        for (const auto& option : question.options) {
            if (option.accepts_free_text || option.label.empty()) continue;
            labels.items().emplace_back(option.label);
            if (!option.description.empty()) description += "\n• " + option.label + ": " + option.description;
        }
        auto choice = core::utils::JsonValue::object();
        choice.set("title", question.header.empty() ? question.question : question.header);
        choice.set("description", description);
        if (question.multi_select) {
            choice.set("type", "array");
            auto items = core::utils::JsonValue::object();
            items.set("type", "string");
            items.set("enum", std::move(labels));
            choice.set("items", std::move(items));
        } else {
            choice.set("type", "string");
            choice.set("enum", std::move(labels));
        }
        properties.set(key, std::move(choice));
        auto other = core::utils::JsonValue::object();
        other.set("type", "string");
        other.set("title", "Other");
        other.set("description", "Optional: your own answer instead of a choice.");
        properties.set(key + "_other", std::move(other));
    }
    auto schema = core::utils::JsonValue::object();
    schema.set("type", "object");
    schema.set("properties", std::move(properties));
    params.set("requestedSchema", std::move(schema));
    return params;
}

std::optional<std::vector<std::pair<std::string, std::string>>> question_answers(
    const core::utils::JsonValue& result,
    const std::vector<core::tools::QuestionItem>& questions) {
    if (field_string(result, "action") != "accept") return std::nullopt;
    const auto* content = field(result, "content");
    std::vector<std::pair<std::string, std::string>> answers;
    for (std::size_t i = 0; i < questions.size(); ++i) {
        const auto key = std::format("q{}", i + 1);
        std::string answer;
        if (content) {
            answer = core::utils::str::trim_ascii_copy(field_string(*content, key + "_other"));
            if (answer.empty()) {
                if (const auto* picked = field(*content, key); picked && picked->is_array()) {
                    for (const auto& item : picked->items()) {
                        if (!item.is_string()) continue;
                        if (!answer.empty()) answer += ", ";
                        answer += item.as_string();
                    }
                } else {
                    answer = field_string(*content, key);
                }
            }
        }
        answers.emplace_back(questions[i].question, std::move(answer));
    }
    return answers;
}

core::utils::JsonValue notice_update(std::string_view session_id,
                                     std::string_view severity,
                                     std::string_view title,
                                     std::string_view description) {
    auto params = session_params(session_id);
    auto update = core::utils::JsonValue::object();
    update.set("sessionUpdate", "notice");
    update.set("severity", std::string(severity));
    update.set("title", std::string(title));
    if (!description.empty()) update.set("description", std::string(description));
    params.set("update", std::move(update));
    return params;
}

core::utils::JsonValue session_info_update(std::string_view session_id,
                                           std::optional<std::string> title,
                                           std::optional<std::string> updated_at) {
    auto params = session_params(session_id);
    auto update = core::utils::JsonValue::object();
    update.set("sessionUpdate", "session_info_update");
    if (title) update.set("title", std::move(*title));
    if (updated_at) update.set("updatedAt", std::move(*updated_at));
    params.set("update", std::move(update));
    return params;
}

core::utils::JsonValue session_list_result(const SessionPage& page) {
    auto result = core::utils::JsonValue::object();
    auto sessions = core::utils::JsonValue::array();
    for (const auto& session : page.sessions) {
        auto entry = core::utils::JsonValue::object();
        entry.set("sessionId", session.session_id);
        entry.set("cwd", session.cwd);
        if (!session.title.empty()) entry.set("title", session.title);
        if (!session.updated_at.empty()) entry.set("updatedAt", session.updated_at);
        sessions.items().push_back(std::move(entry));
    }
    result.set("sessions", std::move(sessions));
    if (page.next_cursor) result.set("nextCursor", *page.next_cursor);
    return result;
}

std::string session_title(std::string_view prompt_text, std::size_t limit) {
    std::string title;
    bool space = false;
    for (const char c : prompt_text) {
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            space = !title.empty();
            continue;
        }
        if (space) title.push_back(' ');
        space = false;
        title.push_back(c);
        if (title.size() > limit) break;
    }
    if (title.size() <= limit) return title;
    // Cut on a UTF-8 boundary, then mark the cut.
    std::size_t cut = limit;
    while (cut > 0 && (static_cast<unsigned char>(title[cut]) & 0xC0) == 0x80) --cut;
    title.resize(cut);
    while (!title.empty() && title.back() == ' ') title.pop_back();
    return title + "…";
}

core::utils::JsonValue usage_update(std::string_view session_id,
                                    long long used,
                                    long long size,
                                    std::optional<double> cost_usd) {
    auto params = session_params(session_id);
    auto update = core::utils::JsonValue::object();
    update.set("sessionUpdate", "usage_update");
    update.set("used", core::utils::JsonValue::integer(used));
    update.set("size", core::utils::JsonValue::integer(size));
    if (cost_usd.has_value()) {
        auto cost = core::utils::JsonValue::object();
        cost.set("amount", core::utils::JsonValue::Number{std::format("{:.6g}", *cost_usd)});
        cost.set("currency", "USD");
        update.set("cost", std::move(cost));
    }
    params.set("update", std::move(update));
    return params;
}

core::utils::JsonValue permission_params(std::string_view session_id,
                                         std::string_view tool_call_id,
                                         std::string_view title,
                                         std::string_view allow_always_id,
                                         std::string_view allow_always_name) {
    auto params = session_params(session_id);
    auto call = core::utils::JsonValue::object();
    call.set("toolCallId", std::string(tool_call_id));
    if (!title.empty()) call.set("title", std::string(title));
    params.set("toolCall", std::move(call));
    auto options = core::utils::JsonValue::array();
    auto allow = core::utils::JsonValue::object();
    allow.set("optionId", "allow-once");
    allow.set("name", "Allow once");
    allow.set("kind", "allow_once");
    options.items().push_back(std::move(allow));
    if (!allow_always_id.empty()) {
        auto always = core::utils::JsonValue::object();
        always.set("optionId", std::string(allow_always_id));
        always.set("name", std::string(allow_always_name));
        always.set("kind", "allow_always");
        options.items().push_back(std::move(always));
    }
    auto reject = core::utils::JsonValue::object();
    reject.set("optionId", "reject-once");
    reject.set("name", "Reject");
    reject.set("kind", "reject_once");
    options.items().push_back(std::move(reject));
    params.set("options", std::move(options));
    return params;
}

core::utils::JsonValue tool_update(std::string_view session_id,
                                   const ToolCallView& call,
                                   bool initial) {
    auto params = session_params(session_id);
    auto update = core::utils::JsonValue::object();
    update.set("sessionUpdate", initial ? "tool_call" : "tool_call_update");
    update.set("toolCallId", call.tool_call_id);
    if (!call.status.empty()) update.set("status", call.status);
    if (initial || !call.title.empty()) update.set("title", call.title);
    if (initial || !call.name.empty()) update.set("name", call.name);
    if (!call.name.empty()) {
        // `name` is not in the schema; editors that read extensions find it here.
        auto filo = core::utils::JsonValue::object();
        filo.set("toolName", call.name);
        auto meta = core::utils::JsonValue::object();
        meta.set("filo", std::move(filo));
        update.set("_meta", std::move(meta));
    }
    if (initial || !call.kind.empty()) update.set("kind", call.kind);
    if (!call.locations.empty()) {
        auto locations = core::utils::JsonValue::array();
        for (const auto& path : call.locations) {
            auto location = core::utils::JsonValue::object();
            location.set("path", path.string());
            locations.items().push_back(std::move(location));
        }
        update.set("locations", std::move(locations));
    }
    if (!call.raw_input.empty()) {
        if (auto parsed = core::utils::JsonValue::parse(call.raw_input)) {
            update.set("rawInput", std::move(*parsed));
        } else {
            update.set("rawInput", call.raw_input);
        }
    }
    if (call.include_raw_output && !call.raw_output.empty()) {
        if (auto parsed = core::utils::JsonValue::parse(call.raw_output)) {
            update.set("rawOutput", std::move(*parsed));
        } else {
            update.set("rawOutput", truncate(call.raw_output, kMaxToolTextChars));
        }
    }
    if (!call.content_text.empty() || !call.diffs.empty()) {
        auto content = core::utils::JsonValue::array();
        if (!call.content_text.empty()) {
            auto block = core::utils::JsonValue::object();
            block.set("type", "content");
            auto text = core::utils::JsonValue::object();
            text.set("type", "text");
            text.set("text", truncate(call.content_text, kMaxToolTextChars));
            block.set("content", std::move(text));
            content.items().push_back(std::move(block));
        }
        for (const auto& diff : call.diffs) content.items().push_back(diff);
        update.set("content", std::move(content));
    }
    params.set("update", std::move(update));
    return params;
}

} // namespace exec::acp
