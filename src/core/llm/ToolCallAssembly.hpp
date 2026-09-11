#pragma once

/**
 * @file
 * @brief Assembly of streamed tool-call fragments into whole tool calls.
 *
 * Every streaming protocol delivers a tool call as a sequence of deltas that
 * must be stitched together by index (or id). The rule lived in three places —
 * the agent turn loop, the API gateway, and the MCP sampling bridge — which is
 * how one provider quirk could be fixed in one of them and stay broken in the
 * other two. It lives here now.
 */

#include "Models.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace core::llm {

/**
 * Appends one streamed arguments fragment to an accumulating tool call.
 *
 * Some OpenAI-compatible providers announce a tool call with a placeholder
 * `{}` argument payload and then stream the real object, which naive
 * concatenation turns into `{}{"path":"..."}` — not JSON, and rejected as an
 * unusable tool call. A complete empty object can never be the prefix or the
 * suffix of a larger JSON value, so it is safe to drop in either position.
 */
void append_arguments_fragment(std::string& accumulated,
                               std::string_view fragment);

/**
 * Merges one streamed tool-call fragment into @p accumulated.
 *
 * Fragments are matched on the streaming index, then on the call id, and
 * finally — for protocols that supply neither — on the single call in flight.
 * Identity and name fields are overwritten when present; arguments accumulate.
 */
void merge_tool_call_fragment(std::vector<ToolCall>& accumulated,
                              const ToolCall& incoming);

} // namespace core::llm
