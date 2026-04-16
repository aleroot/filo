#pragma once

#include "core/llm/Models.hpp"
#include <string>
#include <vector>

namespace core::session {

struct SessionTodoItem {
    std::string id;
    std::string text;
    bool completed = false;
    std::string created_at;
    std::string completed_at;
};

// ---------------------------------------------------------------------------
// SessionData — complete serializable session state.
// JSON encoding/decoding lives in SessionStore so this struct stays a plain
// data type with no external dependencies.
// ---------------------------------------------------------------------------
struct SessionData {
    static constexpr int kVersion = 2;

    int         version         = kVersion;
    std::string session_id;
    std::string created_at;     ///< ISO 8601, e.g. "2026-03-22T10:15:30Z"
    std::string last_active_at; ///< ISO 8601
    std::string working_dir;    ///< filesystem::current_path() at session start
    std::string provider;       ///< active provider name
    std::string model;          ///< active model name
    std::string mode = "BUILD"; ///< agent mode
    std::string context_summary;
    std::string handoff_summary;

    /// Conversation messages — system message is excluded (regenerated on load).
    std::vector<core::llm::Message> messages;
    std::vector<SessionTodoItem> todos;

    struct Stats {
        int32_t prompt_tokens      = 0;
        int32_t completion_tokens  = 0;
        double  cost_usd           = 0.0;
        int32_t turn_count         = 0;
        int32_t tool_calls_total   = 0;
        int32_t tool_calls_success = 0;
    } stats;
};

} // namespace core::session
