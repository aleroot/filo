#pragma once

#include "core/config/ConfigManager.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/AskUserQuestionTool.hpp"
#include "core/utils/JsonValue.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace exec::acp {

inline constexpr int kProtocolVersion = 1;

struct RpcId {
    enum class Kind { none, number, string };

    Kind kind = Kind::none;
    long long number = 0;
    std::string text;

    /// Map key that keeps a numeric id distinct from the same digits as text.
    [[nodiscard]] std::string key() const;
};

[[nodiscard]] RpcId rpc_id(const core::utils::JsonValue* value);

inline constexpr std::size_t kMaxToolTextChars = 8 * 1024;

struct RpcMessage {
    enum class Kind { invalid, request, notification, response };

    Kind kind = Kind::invalid;
    RpcId id;
    std::string method;
    core::utils::JsonValue params;
    core::utils::JsonValue result;
    bool has_error = false;
    int error_code = 0;
    std::string error_message;
    std::string parse_error;
};

[[nodiscard]] RpcMessage parse_message(std::string_view line);

[[nodiscard]] std::string encode_result(const RpcId& id, const core::utils::JsonValue& result);
[[nodiscard]] std::string encode_error(const RpcId& id,
                                       int code,
                                       std::string_view message,
                                       const core::utils::JsonValue* data = nullptr);
[[nodiscard]] std::string encode_notification(std::string_view method,
                                              const core::utils::JsonValue& params);
[[nodiscard]] std::string encode_request(const RpcId& id,
                                         std::string_view method,
                                         const core::utils::JsonValue& params);

/// The version to advertise. Null when `protocolVersion` is missing or not an
/// integer in range: the caller answers with -32602 and stays uninitialized.
/// A newer client still gets version 1, which is the latest this agent speaks.
[[nodiscard]] std::optional<int> negotiated_protocol_version(
    const core::utils::JsonValue& params,
    std::string* error);

/// What the editor said it can do in `initialize`. Anything not advertised is
/// off: the protocol forbids sending a client what it did not ask for.
struct ClientCapabilities {
    /// `session.notices`: live notices outside the conversation.
    bool notices = false;
    /// `elicitation.form`: structured questions rendered as a form.
    bool elicitation_form = false;
    /// `auth.terminal`: the editor can rerun the agent in a terminal to sign in.
    bool terminal_auth = false;
};

[[nodiscard]] ClientCapabilities parse_client_capabilities(const core::utils::JsonValue& params);

/// The `initialize` answer. A non-empty `login_provider` adds a terminal
/// sign-in method (`--auth <provider>`); offer it only to a client with
/// `terminal_auth`.
[[nodiscard]] core::utils::JsonValue initialize_result(int protocol_version = kProtocolVersion,
                                                       std::string_view login_provider = {});

/// JSON-RPC code the protocol reserves for "sign in first".
inline constexpr int kAuthRequiredCode = -32000;

struct PromptBlocks {
    std::string text;
    std::vector<core::llm::ContentPart> parts;
    /// Absolute paths the client named. The session grants the ones that exist.
    std::vector<std::filesystem::path> grant_paths;
};

/// `media_dir` receives decoded image attachments. Audio blocks become a note
/// in `text`; the protocol advertises no audio capability.
[[nodiscard]] std::optional<PromptBlocks> parse_prompt(
    const core::utils::JsonValue& prompt,
    const std::filesystem::path& media_dir,
    std::string* error);

/// Missing or null `servers` is an empty list. Credentials are copied into the
/// config and must not be logged by the caller.
[[nodiscard]] std::optional<std::vector<core::config::McpServerConfig>> parse_mcp_servers(
    const core::utils::JsonValue* servers,
    std::string* error);

[[nodiscard]] std::string tool_kind(std::string_view tool_name);
[[nodiscard]] std::string tool_title(std::string_view tool_name, std::string_view args);
[[nodiscard]] std::vector<std::filesystem::path> tool_locations(
    std::string_view args,
    const std::filesystem::path& cwd);

/// One side of a file diff, read straight from disk. `readable` is false when
/// the content cannot be shown: binary, too large, or unreadable. A missing
/// file is readable with no text.
struct FileSide {
    bool readable = false;
    std::optional<std::string> text;
};

[[nodiscard]] FileSide read_file_side(const std::filesystem::path& path);

/// The protocol's diff content for one file a tool changed. Null when the path
/// is not absolute, either side cannot be shown, or nothing changed. A file the
/// tool created has a null `oldText`; a file it deleted has an empty `newText`.
[[nodiscard]] std::optional<core::utils::JsonValue> file_diff(
    const std::filesystem::path& path,
    const FileSide& before,
    const FileSide& after);

[[nodiscard]] bool tool_result_failed(std::string_view result);

/// What a tool card shows as text: a command's output instead of its JSON
/// envelope, an error's message, otherwise the result as returned.
[[nodiscard]] std::string tool_display_text(std::string_view tool_name, std::string_view result);

struct PermissionDecision {
    bool allowed = false;
    bool remember = false;
    bool cancelled = false;
};

/// Anything other than a recognised selected option is a denial. `cancelled`
/// is reported separately so the turn can stop instead of continuing denied.
[[nodiscard]] PermissionDecision interpret_permission(
    const core::utils::JsonValue& result,
    std::string_view allow_always_id);

struct ModeChoice {
    std::string id;
    std::string name;
    std::string description;
};

struct ConfigChoice {
    std::string value;
    std::string name;
    std::string description;
};

struct ConfigSelect {
    std::string id;
    std::string name;
    std::string description;
    std::string category;
    std::string current;
    std::vector<ConfigChoice> options;
};

/// The editor's model picker (`models` with `session/set_model`), offered next
/// to the `model` config option for clients that still use it.
[[nodiscard]] core::utils::JsonValue model_state(const ConfigSelect& model);

/// A form asking the questions of one ask_user_question call. Each question
/// is a choice field `qN`, plus an optional free-text field `qN_other` that
/// stands for the tool's "Other" row.
[[nodiscard]] core::utils::JsonValue question_form(std::string_view session_id,
                                                   std::string_view tool_call_id,
                                                   const std::vector<core::tools::QuestionItem>& questions);

/// The answers in an `elicitation/create` result, in question order. Null
/// when the user declined or dismissed the form.
[[nodiscard]] std::optional<std::vector<std::pair<std::string, std::string>>> question_answers(
    const core::utils::JsonValue& result,
    const std::vector<core::tools::QuestionItem>& questions);

struct CommandSpec {
    std::string name;
    std::string description;
};

[[nodiscard]] core::utils::JsonValue text_update(std::string_view session_id,
                                                 std::string_view kind,
                                                 std::string_view text);
[[nodiscard]] core::utils::JsonValue mode_state(std::string_view current,
                                                const std::vector<ModeChoice>& modes);
[[nodiscard]] core::utils::JsonValue config_options(const std::vector<ConfigSelect>& options);
[[nodiscard]] core::utils::JsonValue commands_update(std::string_view session_id,
                                                     const std::vector<CommandSpec>& commands);
struct PlanItem {
    std::string content;
    std::string priority;
    std::string status;
};

[[nodiscard]] core::utils::JsonValue plan_update(std::string_view session_id,
                                                 const std::vector<PlanItem>& entries);
/// A live notice. Only sent to clients that advertised `session.notices`.
[[nodiscard]] core::utils::JsonValue notice_update(std::string_view session_id,
                                                   std::string_view severity,
                                                   std::string_view title,
                                                   std::string_view description = {});
[[nodiscard]] core::utils::JsonValue session_info_update(std::string_view session_id,
                                                         std::optional<std::string> title,
                                                         std::optional<std::string> updated_at);

struct ListedSession {
    std::string session_id;
    std::string cwd;
    std::string title;
    std::string updated_at;
};

struct SessionPage {
    std::vector<ListedSession> sessions;
    std::optional<std::string> next_cursor;
};

[[nodiscard]] core::utils::JsonValue session_list_result(const SessionPage& page);

/// A one-line title from the first prompt, at most `limit` characters.
[[nodiscard]] std::string session_title(std::string_view prompt_text, std::size_t limit = 80);

[[nodiscard]] core::utils::JsonValue usage_update(std::string_view session_id,
                                                  long long used,
                                                  long long size,
                                                  std::optional<double> cost_usd);
[[nodiscard]] core::utils::JsonValue permission_params(std::string_view session_id,
                                                       std::string_view tool_call_id,
                                                       std::string_view title,
                                                       std::string_view allow_always_id,
                                                       std::string_view allow_always_name);

struct ToolCallView {
    std::string tool_call_id;
    std::string name;
    std::string title;
    std::string kind;
    std::string status;
    std::string raw_input;
    std::string raw_output;
    std::string content_text;
    std::vector<core::utils::JsonValue> diffs;
    std::vector<std::filesystem::path> locations;
    bool include_raw_output = false;
};

[[nodiscard]] core::utils::JsonValue tool_update(std::string_view session_id,
                                                 const ToolCallView& call,
                                                 bool initial);

} // namespace exec::acp
