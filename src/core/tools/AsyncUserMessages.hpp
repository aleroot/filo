#pragma once

#include "Tool.hpp"

#include <expected>
#include <string>
#include <string_view>

namespace core::tools {

// Codex's asynchronous user messaging (codex-rs core/src/tools/handlers/
// request_user_input_async.rs, send_message_to_user_async.rs): the call shows
// the user a message and returns at once. The turn keeps going, and a reply
// arrives later as a user message that steers it.

/// `request_user_input_async`: questions, each with optional suggested answers.
[[nodiscard]] ToolDefinition request_user_input_async_definition();
/// `send_message_to_user_async`: one message that needs the user's attention.
[[nodiscard]] ToolDefinition send_message_to_user_async_definition();

/// Whether @p tool_name is one of the two.
[[nodiscard]] bool is_async_user_message_tool(std::string_view tool_name) noexcept;

/// What the user is shown for a call (questions as a title followed by its
/// options), or the error the model reads when the arguments say nothing.
[[nodiscard]] std::expected<std::string, std::string> async_user_message(
    std::string_view tool_name, std::string_view arguments);

/// The result the model reads once the message was shown.
inline constexpr std::string_view kAsyncUserMessageAccepted = R"({"accepted":true})";

} // namespace core::tools
