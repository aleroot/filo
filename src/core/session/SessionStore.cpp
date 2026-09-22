#include "SessionStore.hpp"

#include "ActiveSessionLease.hpp"
#include "ThreadCatalog.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/JsonUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <system_error>

namespace core::session {

namespace {

// SessionStore::to_json emits all catalogue fields before `messages`. Read
// only through that top-level key when selecting a session to resume. This is
// deliberately a small JSON lexer rather than a substring search: summaries
// and goal snapshots are strings that may themselves contain the word
// "messages" or JSON-looking text.
[[nodiscard]] bool read_session_header_prefix(
    std::istream& input,
    std::string& header) {
    enum class KeySuffix { None, Colon, Array };

    std::string prefix;
    bool in_string = false;
    bool escaped = false;
    std::size_t object_depth = 0;
    std::size_t array_depth = 0;
    std::size_t string_start = std::string::npos;
    std::size_t last_top_level_comma = std::string::npos;
    KeySuffix key_suffix = KeySuffix::None;

    char ch = '\0';
    while (input.get(ch)) {
        const std::size_t position = prefix.size();
        prefix.push_back(ch);

        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
                if (object_depth == 1 && array_depth == 0
                    && string_start != std::string::npos
                    && std::string_view(prefix).substr(
                           string_start, position - string_start) == "messages") {
                    key_suffix = KeySuffix::Colon;
                }
            }
            continue;
        }

        if (key_suffix == KeySuffix::Colon) {
            if (core::utils::json::is_whitespace(ch)) continue;
            if (ch == ':') {
                key_suffix = KeySuffix::Array;
                continue;
            }
            key_suffix = KeySuffix::None;
        } else if (key_suffix == KeySuffix::Array) {
            if (core::utils::json::is_whitespace(ch)) continue;
            if (ch == '[' && last_top_level_comma != std::string::npos) {
                prefix.resize(last_top_level_comma);
                prefix.push_back('}');
                header = std::move(prefix);
                return true;
            }
            key_suffix = KeySuffix::None;
        }

        if (ch == '"') {
            string_start = object_depth == 1 && array_depth == 0
                ? prefix.size()
                : std::string::npos;
            in_string = true;
            escaped = false;
            continue;
        }

        if (ch == ',' && object_depth == 1 && array_depth == 0) {
            last_top_level_comma = position;
        } else if (ch == '{') {
            ++object_depth;
        } else if (ch == '}') {
            if (object_depth > 0) --object_depth;
        } else if (ch == '[') {
            ++array_depth;
        } else if (ch == ']') {
            if (array_depth > 0) --array_depth;
        }
    }

    // A legacy or externally generated session may order fields differently.
    // The caller falls back to the full decoder for that uncommon case.
    header = std::move(prefix);
    return false;
}

[[nodiscard]] std::optional<SessionInfo> parse_session_header(
    std::string_view json,
    const std::filesystem::path& path) {
    const auto parse = [&](std::string_view input) -> std::optional<SessionInfo> {
        simdjson::padded_string padded(input);
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document document;
        if (parser.iterate(padded).get(document) != simdjson::SUCCESS) {
            return std::nullopt;
        }

        simdjson::ondemand::object fields;
        if (document.get_object().get(fields) != simdjson::SUCCESS) {
            return std::nullopt;
        }

        SessionInfo info;
        info.path = path;
        int64_t version = 1;
        for (auto field : fields) {
            std::string_view key;
            if (field.unescaped_key().get(key) != simdjson::SUCCESS) {
                return std::nullopt;
            }

            if (key == "version") {
                int64_t parsed_version = 1;
                if (field.value().get_int64().get(parsed_version) == simdjson::SUCCESS) {
                    version = parsed_version;
                }
                continue;
            }

            std::string* target = nullptr;
            if (key == "session_id") target = &info.session_id;
            else if (key == "name") target = &info.name;
            else if (key == "created_at") target = &info.created_at;
            else if (key == "last_active_at") target = &info.last_active_at;
            else if (key == "working_dir") target = &info.working_dir;
            else if (key == "provider") target = &info.provider;
            else if (key == "model") target = &info.model;
            else if (key == "mode") target = &info.mode;
            if (target == nullptr) continue;

            std::string_view value;
            if (field.value().get_string().get(value) == simdjson::SUCCESS) {
                *target = value;
            }
        }

        if (version > SessionData::kVersion) return std::nullopt;
        return info;
    };

    if (auto info = parse(json); info.has_value()) return info;
    if (simdjson::validate_utf8(json)) return std::nullopt;
    return parse(core::utils::repair_utf8(json));
}

[[nodiscard]] bool more_recent(const SessionInfo& lhs, const SessionInfo& rhs) {
    const auto& lhs_time = lhs.last_active_at.empty() ? lhs.created_at : lhs.last_active_at;
    const auto& rhs_time = rhs.last_active_at.empty() ? rhs.created_at : rhs.last_active_at;
    return lhs_time > rhs_time;
}

[[nodiscard]] SessionInfo session_info_from_data(
    const SessionData& data,
    const std::filesystem::path& path) {
    SessionInfo info;
    info.session_id = data.session_id;
    info.name = data.name;
    info.created_at = data.created_at;
    info.last_active_at = data.last_active_at;
    info.working_dir = data.working_dir;
    info.provider = data.provider;
    info.model = data.model;
    info.mode = data.mode;
    info.preview = first_user_message_preview(data.messages);
    info.turn_count = data.stats.turn_count;
    info.path = path;
    return info;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SessionStore::SessionStore(std::filesystem::path sessions_dir)
    : sessions_dir_(std::move(sessions_dir)) {}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

std::string SessionStore::generate_id() {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<int> dis{0, 15};
    static constexpr std::string_view kHex = "0123456789abcdef";
    std::string id(8, '0');
    for (char& c : id) {
        c = kHex[static_cast<std::size_t>(dis(gen))];
    }
    return id;
}

std::string SessionStore::to_iso8601(std::chrono::system_clock::time_point when) {
    const auto tt = std::chrono::system_clock::to_time_t(when);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    return std::format("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec);
}

std::string SessionStore::now_iso8601() {
    return to_iso8601(std::chrono::system_clock::now());
}

std::filesystem::path SessionStore::default_sessions_dir() {
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && xdg[0] != '\0') {
        return std::filesystem::path{xdg} / "filo" / "sessions";
    }
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::filesystem::path{home} / ".local" / "share" / "filo" / "sessions";
    }
    return std::filesystem::temp_directory_path() / "filo" / "sessions";
}

std::filesystem::path SessionStore::canonicalize_working_dir(std::string_view dir) {
    if (dir.empty()) {
        return {};
    }
    std::error_code ec;
    std::filesystem::path resolved{dir};
    if (!resolved.is_absolute()) {
        resolved = std::filesystem::absolute(resolved, ec);
    }
    resolved = std::filesystem::weakly_canonical(resolved, ec);
    if (ec) {
        // The path may no longer exist (e.g. a deleted project). Fall back to a
        // purely lexical normalization so such sessions are still comparable.
        resolved = std::filesystem::path{dir}.lexically_normal();
    }
    return resolved;
}

bool SessionStore::working_dirs_match(std::string_view a, std::string_view b) {
    // An unknown directory on either side can't be compared — treat it as a
    // match so legacy sessions (pre-dating working_dir) never trigger a warning.
    if (a.empty() || b.empty()) {
        return true;
    }
    return canonicalize_working_dir(a) == canonicalize_working_dir(b);
}

std::optional<std::string>
SessionStore::working_dir_mismatch_notice(std::string_view session_dir,
                                          std::string_view current_dir) {
    if (working_dirs_match(session_dir, current_dir)) {
        return std::nullopt;
    }
    return std::format(
        "Resuming a session from a different project directory:\n"
        "  session dir : {}\n"
        "  current dir : {}\n"
        "Tools will operate in the current directory. To resume within the "
        "original project, run filo from there; for an auto-scoped resume use "
        "`filo -c`.",
        session_dir, current_dir);
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

std::filesystem::path SessionStore::compute_path(const SessionData& data) const {
    // Derive a sortable date segment from created_at.
    // "2026-03-22T10:15:30Z" → "20260322-101530"
    std::string date_part;
    if (data.created_at.size() >= 19) {
        date_part =  data.created_at.substr(0, 4)    // year
                   + data.created_at.substr(5, 2)    // month
                   + data.created_at.substr(8, 2)    // day
                   + "-"
                   + data.created_at.substr(11, 2)   // hour
                   + data.created_at.substr(14, 2)   // min
                   + data.created_at.substr(17, 2);  // sec
    } else {
        date_part = "00000000-000000";
    }
    const std::string filename = std::format("session-{}-{}.json", date_part, data.session_id);
    return sessions_dir_ / filename;
}

bool SessionStore::ensure_dir(std::string* error) const {
    std::error_code ec;
    std::filesystem::create_directories(sessions_dir_, ec);
    if (ec) {
        if (error) {
            *error = std::format("Cannot create session directory '{}': {}",
                                 sessions_dir_.string(), ec.message());
        }
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

namespace {

void append_message_json(std::string& out, const core::llm::Message& msg) {
    out += "{\"role\":\"";
    core::utils::append_escaped(out, msg.role);
    out += "\",\"content\":\"";
    core::utils::append_escaped(out, msg.content);
    out += '"';

    if (!msg.name.empty()) {
        out += ",\"name\":\"";
        core::utils::append_escaped(out, msg.name);
        out += '"';
    }
    if (!msg.tool_call_id.empty()) {
        out += ",\"tool_call_id\":\"";
        core::utils::append_escaped(out, msg.tool_call_id);
        out += '"';
    }
    if (msg.synthetic) {
        out += ",\"synthetic\":true";
    }
    if (!msg.input_text.empty()) {
        out += ",\"input_text\":\"";
        core::utils::append_escaped(out, msg.input_text);
        out += '"';
    }
    if (!msg.reasoning_content.empty()) {
        out += ",\"reasoning_content\":\"";
        core::utils::append_escaped(out, msg.reasoning_content);
        out += '"';
    }
    if (!msg.reasoning_protocol.empty()) {
        out += ",\"reasoning_protocol\":\"";
        core::utils::append_escaped(out, msg.reasoning_protocol);
        out += '"';
    }
    if (!msg.reasoning_elapsed.empty()) {
        out += ",\"reasoning_elapsed\":\"";
        core::utils::append_escaped(out, msg.reasoning_elapsed);
        out += '"';
    }
    if (!msg.continuation_items.empty()) {
        out += ",\"continuation_items\":[";
        for (std::size_t i = 0; i < msg.continuation_items.size(); ++i) {
            if (i > 0) out += ',';
            const auto& item = msg.continuation_items[i];
            out += "{\"provider\":\"";
            core::utils::append_escaped(out, item.provider);
            out += "\",\"kind\":\"";
            core::utils::append_escaped(out, item.kind);
            out += "\",\"payload\":\"";
            core::utils::append_escaped(out, item.payload);
            out += "\"}";
        }
        out += ']';
    }
    if (!msg.tool_calls.empty()) {
        out += ",\"tool_calls\":[";
        for (std::size_t i = 0; i < msg.tool_calls.size(); ++i) {
            const auto& tc = msg.tool_calls[i];
            if (i > 0) out += ',';
            out += "{\"id\":\"";
            core::utils::append_escaped(out, tc.id);
            out += "\",\"type\":\"";
            core::utils::append_escaped(out, tc.type);
            out += "\",\"function\":{\"name\":\"";
            core::utils::append_escaped(out, tc.function.name);
            out += "\",\"arguments\":\"";
            core::utils::append_escaped(out, tc.function.arguments);
            out += "\"}}";
        }
        out += ']';
    }
    if (!msg.content_parts.empty()) {
        out += ",\"content_parts\":[";
        for (std::size_t i = 0; i < msg.content_parts.size(); ++i) {
            const auto& part = msg.content_parts[i];
            if (i > 0) out += ',';
            out += "{\"type\":\"";
            out += core::llm::media_kind(part.type);
            out += '"';
            if (part.type == core::llm::ContentPartType::Text) {
                out += ",\"text\":\"";
                core::utils::append_escaped(out, part.text);
                out += '"';
            } else {
                if (!part.path.empty()) {
                    out += ",\"path\":\"";
                    core::utils::append_escaped(out, part.path);
                    out += '"';
                }
                if (!part.url.empty()) {
                    out += ",\"url\":\"";
                    core::utils::append_escaped(out, part.url);
                    out += '"';
                }
                if (!part.media_id.empty()) {
                    out += ",\"media_id\":\"";
                    core::utils::append_escaped(out, part.media_id);
                    out += '"';
                }
                if (!part.mime_type.empty()) {
                    out += ",\"mime_type\":\"";
                    core::utils::append_escaped(out, part.mime_type);
                    out += '"';
                }
                if (!part.detail.empty()) {
                    out += ",\"detail\":\"";
                    core::utils::append_escaped(out, part.detail);
                    out += '"';
                }
            }
            out += '}';
        }
        out += ']';
    }
    out += '}';
}

void append_todo_json(std::string& out, const core::session::SessionTodoItem& todo) {
    out += "{\"id\":\"";
    core::utils::append_escaped(out, todo.id);
    out += "\",\"text\":\"";
    core::utils::append_escaped(out, todo.text);
    out += "\",\"status\":\"";
    core::utils::append_escaped(out, core::session::to_string(todo.status));
    out += "\",\"created_at\":\"";
    core::utils::append_escaped(out, todo.created_at);
    out += "\",\"updated_at\":\"";
    core::utils::append_escaped(out, todo.updated_at);
    out += "\",\"completed_at\":\"";
    core::utils::append_escaped(out, todo.completed_at);
    out += "\"}";
}

void append_goal_json(std::string& out, const core::session::SessionGoal& goal) {
    out += "{\"objective\":\"";
    core::utils::append_escaped(out, goal.objective);
    out += "\",\"status\":\"";
    core::utils::append_escaped(out, core::session::to_string(goal.status));
    out += "\",\"note\":\"";
    core::utils::append_escaped(out, goal.note);
    out += "\",\"created_at\":\"";
    core::utils::append_escaped(out, goal.created_at);
    out += "\",\"updated_at\":\"";
    core::utils::append_escaped(out, goal.updated_at);
    out += "\",\"completed_at\":\"";
    core::utils::append_escaped(out, goal.completed_at);
    out += "\"}";
}

void append_goal_graph_json(std::string& out,
                            const core::session::SessionGoalGraph& graph) {
    out += "{\"plan_version\":";
    out += std::to_string(graph.plan_version);
    out += ",\"run_state\":\"";
    core::utils::append_escaped(out, graph.run_state);
    out += "\",\"snapshot\":\"";
    core::utils::append_escaped(out, graph.snapshot);
    out += "\",\"updated_at\":\"";
    core::utils::append_escaped(out, graph.updated_at);
    out += "\"}";
}

} // namespace

std::string SessionStore::to_json(const SessionData& data) {
    std::string out;
    out.reserve(4096 + data.messages.size() * 512);

    out += "{\"version\":";
    out += std::to_string(SessionData::kVersion);
    out += ",\"session_id\":\"";
    core::utils::append_escaped(out, data.session_id);
    out += "\",\"name\":\"";
    core::utils::append_escaped(out, data.name);
    out += "\",\"created_at\":\"";
    core::utils::append_escaped(out, data.created_at);
    out += "\",\"last_active_at\":\"";
    core::utils::append_escaped(out, data.last_active_at);
    out += "\",\"working_dir\":\"";
    core::utils::append_escaped(out, data.working_dir);
    out += "\",\"provider\":\"";
    core::utils::append_escaped(out, data.provider);
    out += "\",\"model\":\"";
    core::utils::append_escaped(out, data.model);
    out += "\",\"mode\":\"";
    core::utils::append_escaped(out, data.mode);
    // Keep catalogue data ahead of the conversation and opaque summaries so
    // resume scans can stop at `messages` without reading large payloads.
    out += "\",\"stats\":{";
    out += "\"prompt_tokens\":";
    out += std::to_string(data.stats.prompt_tokens);
    out += ",\"completion_tokens\":";
    out += std::to_string(data.stats.completion_tokens);
    out += ",\"cost_usd\":";
    out += std::format("{:.6f}", data.stats.cost_usd);
    out += ",\"turn_count\":";
    out += std::to_string(data.stats.turn_count);
    out += ",\"tool_calls_total\":";
    out += std::to_string(data.stats.tool_calls_total);
    out += ",\"tool_calls_success\":";
    out += std::to_string(data.stats.tool_calls_success);
    out += '}';
    out += ",\"messages\":[";

    for (std::size_t i = 0; i < data.messages.size(); ++i) {
        if (i > 0) out += ',';
        append_message_json(out, data.messages[i]);
    }

    out += "],\"context_summary\":\"";
    core::utils::append_escaped(out, data.context_summary);
    out += "\",\"handoff_summary\":\"";
    core::utils::append_escaped(out, data.handoff_summary);
    out += '"';
    if (data.goal.has_value()) {
        out += ",\"goal\":";
        append_goal_json(out, *data.goal);
    }
    if (data.goal_graph.has_value()) {
        out += ",\"goal_graph\":";
        append_goal_graph_json(out, *data.goal_graph);
    }

    out += ",\"todos\":[";
    for (std::size_t i = 0; i < data.todos.size(); ++i) {
        if (i > 0) out += ',';
        append_todo_json(out, data.todos[i]);
    }

    out += "]}";

    return out;
}

std::optional<SessionData> SessionStore::from_json(std::string_view json) {
    try {
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        std::string repaired_json;
        if (parser.parse(json.data(), json.size()).get(doc) != simdjson::SUCCESS) {
            if (simdjson::validate_utf8(json)) return std::nullopt;
            repaired_json = core::utils::repair_utf8(json);
            if (parser.parse(repaired_json).get(doc) != simdjson::SUCCESS) {
                return std::nullopt;
            }
        }

        int64_t stored_version = 1;
        if (doc["version"].get(stored_version) != simdjson::SUCCESS) {
            stored_version = 1;
        }
        if (stored_version > SessionData::kVersion) {
            return std::nullopt;
        }

        SessionData data;
        // Successful parsing normalizes the in-memory representation. Any
        // subsequent save must advertise the schema it actually emits, not
        // the legacy version found on disk.
        data.version = SessionData::kVersion;

        auto get_str = [&](const char* key, std::string& out) {
            std::string_view sv;
            if (doc[key].get(sv) == simdjson::SUCCESS) out = std::string(sv);
        };
        get_str("session_id",      data.session_id);
        get_str("name",            data.name);
        get_str("created_at",      data.created_at);
        get_str("last_active_at",  data.last_active_at);
        get_str("working_dir",     data.working_dir);
        get_str("provider",        data.provider);
        get_str("model",           data.model);
        get_str("mode",            data.mode);
        get_str("context_summary", data.context_summary);
        get_str("handoff_summary", data.handoff_summary);

        simdjson::dom::object goal_obj;
        if (doc["goal"].get(goal_obj) == simdjson::SUCCESS) {
            SessionGoal goal;
            std::string_view sv;
            if (goal_obj["objective"].get(sv) == simdjson::SUCCESS) {
                goal.objective = std::string(sv);
            }
            if (goal_obj["status"].get(sv) == simdjson::SUCCESS) {
                goal.status = goal_status_from_string(sv);
            }
            if (goal_obj["note"].get(sv) == simdjson::SUCCESS) {
                goal.note = std::string(sv);
            }
            if (goal_obj["created_at"].get(sv) == simdjson::SUCCESS) {
                goal.created_at = std::string(sv);
            }
            if (goal_obj["updated_at"].get(sv) == simdjson::SUCCESS) {
                goal.updated_at = std::string(sv);
            }
            if (goal_obj["completed_at"].get(sv) == simdjson::SUCCESS) {
                goal.completed_at = std::string(sv);
            }
            if (!goal.objective.empty()) {
                data.goal = std::move(goal);
            }
        }

        simdjson::dom::object goal_graph_obj;
        if (doc["goal_graph"].get(goal_graph_obj) == simdjson::SUCCESS) {
            SessionGoalGraph goal_graph;
            std::string_view sv;
            int64_t plan_version = 0;
            if (goal_graph_obj["plan_version"].get(plan_version) == simdjson::SUCCESS) {
                goal_graph.plan_version = static_cast<int>(plan_version);
            }
            if (goal_graph_obj["run_state"].get(sv) == simdjson::SUCCESS) {
                goal_graph.run_state = std::string(sv);
            }
            if (goal_graph_obj["snapshot"].get(sv) == simdjson::SUCCESS) {
                goal_graph.snapshot = std::string(sv);
            }
            if (goal_graph_obj["updated_at"].get(sv) == simdjson::SUCCESS) {
                goal_graph.updated_at = std::string(sv);
            }
            if (!goal_graph.snapshot.empty()) {
                data.goal_graph = std::move(goal_graph);
            }
        }

        simdjson::dom::array messages_arr;
        if (doc["messages"].get(messages_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element msg_el : messages_arr) {
                core::llm::Message msg;
                std::string_view sv;
                if (msg_el["role"].get(sv)         == simdjson::SUCCESS) msg.role         = std::string(sv);
                if (msg_el["content"].get(sv)      == simdjson::SUCCESS) msg.content      = std::string(sv);
                if (msg_el["name"].get(sv)         == simdjson::SUCCESS) msg.name         = std::string(sv);
                if (msg_el["tool_call_id"].get(sv) == simdjson::SUCCESS) msg.tool_call_id = std::string(sv);
                bool synthetic = false;
                if (msg_el["synthetic"].get(synthetic) == simdjson::SUCCESS) {
                    msg.synthetic = synthetic;
                }
                if (msg_el["input_text"].get(sv) == simdjson::SUCCESS) {
                    msg.input_text = std::string(sv);
                }
                if (msg_el["reasoning_content"].get(sv) == simdjson::SUCCESS) {
                    msg.reasoning_content = std::string(sv);
                }
                if (msg_el["reasoning_protocol"].get(sv) == simdjson::SUCCESS) {
                    msg.reasoning_protocol = std::string(sv);
                }
                if (msg_el["reasoning_elapsed"].get(sv) == simdjson::SUCCESS) {
                    msg.reasoning_elapsed = std::string(sv);
                }
                simdjson::dom::array continuation_arr;
                if (msg_el["continuation_items"].get(continuation_arr)
                    == simdjson::SUCCESS) {
                    for (simdjson::dom::element item_el : continuation_arr) {
                        core::llm::ContinuationItem item;
                        if (item_el["provider"].get(sv) == simdjson::SUCCESS) {
                            item.provider = std::string(sv);
                        }
                        if (item_el["kind"].get(sv) == simdjson::SUCCESS) {
                            item.kind = std::string(sv);
                        }
                        if (item_el["payload"].get(sv) == simdjson::SUCCESS) {
                            item.payload = std::string(sv);
                        }
                        if (!item.payload.empty()) {
                            msg.continuation_items.push_back(std::move(item));
                        }
                    }
                }

                simdjson::dom::array tc_arr;
                if (msg_el["tool_calls"].get(tc_arr) == simdjson::SUCCESS) {
                    for (simdjson::dom::element tc_el : tc_arr) {
                        core::llm::ToolCall tc;
                        if (tc_el["id"].get(sv)   == simdjson::SUCCESS) tc.id   = std::string(sv);
                        if (tc_el["type"].get(sv) == simdjson::SUCCESS) tc.type = std::string(sv);
                        simdjson::dom::object fn_obj;
                        if (tc_el["function"].get(fn_obj) == simdjson::SUCCESS) {
                            if (fn_obj["name"].get(sv)      == simdjson::SUCCESS)
                                tc.function.name      = std::string(sv);
                            if (fn_obj["arguments"].get(sv) == simdjson::SUCCESS)
                                tc.function.arguments = std::string(sv);
                        }
                        msg.tool_calls.push_back(std::move(tc));
                    }
                }

                simdjson::dom::array parts_arr;
                if (msg_el["content_parts"].get(parts_arr) == simdjson::SUCCESS) {
                    for (simdjson::dom::element part_el : parts_arr) {
                        core::llm::ContentPart part;
                        if (part_el["type"].get(sv) == simdjson::SUCCESS) {
                            if (sv == "image") {
                                part.type = core::llm::ContentPartType::Image;
                            } else if (sv == "video") {
                                part.type = core::llm::ContentPartType::Video;
                            }
                        }
                        if (part_el["text"].get(sv) == simdjson::SUCCESS) {
                            part.text = std::string(sv);
                        }
                        if (part_el["path"].get(sv) == simdjson::SUCCESS) {
                            part.path = std::string(sv);
                        }
                        if (part_el["url"].get(sv) == simdjson::SUCCESS) {
                            part.url = std::string(sv);
                        }
                        if (part_el["media_id"].get(sv) == simdjson::SUCCESS) {
                            part.media_id = std::string(sv);
                        }
                        if (part_el["mime_type"].get(sv) == simdjson::SUCCESS) {
                            part.mime_type = std::string(sv);
                        }
                        if (part_el["detail"].get(sv) == simdjson::SUCCESS) {
                            part.detail = std::string(sv);
                        }
                        msg.content_parts.push_back(std::move(part));
                    }
                }
                data.messages.push_back(std::move(msg));
            }
        }

        simdjson::dom::array todos_arr;
        if (doc["todos"].get(todos_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element todo_el : todos_arr) {
                core::session::SessionTodoItem todo;
                std::string_view sv;
                if (todo_el["id"].get(sv) == simdjson::SUCCESS) {
                    todo.id = std::string(sv);
                }
                if (todo_el["text"].get(sv) == simdjson::SUCCESS) {
                    todo.text = std::string(sv);
                }
                if (todo_el["status"].get(sv) == simdjson::SUCCESS) {
                    todo.status = core::session::todo_status_from_string(sv);
                } else {
                    bool completed = false;
                    if (todo_el["completed"].get(completed) == simdjson::SUCCESS && completed) {
                        todo.status = core::session::TodoStatus::Completed;
                    }
                }
                if (todo_el["created_at"].get(sv) == simdjson::SUCCESS) {
                    todo.created_at = std::string(sv);
                }
                if (todo_el["updated_at"].get(sv) == simdjson::SUCCESS) {
                    todo.updated_at = std::string(sv);
                }
                if (todo_el["completed_at"].get(sv) == simdjson::SUCCESS) {
                    todo.completed_at = std::string(sv);
                }
                if (!todo.id.empty() || !todo.text.empty()) {
                    data.todos.push_back(std::move(todo));
                }
            }
        }

        simdjson::dom::object stats_obj;
        if (doc["stats"].get(stats_obj) == simdjson::SUCCESS) {
            int64_t ival = 0;
            double  dval = 0.0;
            if (stats_obj["prompt_tokens"].get(ival)      == simdjson::SUCCESS)
                data.stats.prompt_tokens      = static_cast<int32_t>(ival);
            if (stats_obj["completion_tokens"].get(ival)  == simdjson::SUCCESS)
                data.stats.completion_tokens  = static_cast<int32_t>(ival);
            if (stats_obj["cost_usd"].get(dval)           == simdjson::SUCCESS)
                data.stats.cost_usd           = dval;
            if (stats_obj["turn_count"].get(ival)         == simdjson::SUCCESS)
                data.stats.turn_count         = static_cast<int32_t>(ival);
            if (stats_obj["tool_calls_total"].get(ival)   == simdjson::SUCCESS)
                data.stats.tool_calls_total   = static_cast<int32_t>(ival);
            if (stats_obj["tool_calls_success"].get(ival) == simdjson::SUCCESS)
                data.stats.tool_calls_success = static_cast<int32_t>(ival);
        }

        return data;
    } catch (...) {
        return std::nullopt;
    }
}

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

bool SessionStore::save(const SessionData& data, std::string* error) const {
    if (!ensure_dir(error)) return false;

    const auto target_path = compute_path(data);
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(target_path), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }

    const std::string json = to_json(data);
    return core::utils::atomic_write_file(target_path, json, error);
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

std::optional<SessionInfo> SessionStore::read_session_header(
    const std::filesystem::path& path) const {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;

    std::string header;
    const bool has_header_prefix = read_session_header_prefix(in, header);
    if (has_header_prefix) {
        if (auto info = parse_session_header(header, path); info.has_value()) return info;

        // Preserve the existing UTF-8 repair and schema compatibility behavior
        // if the lightweight header parser cannot handle this file.
        if (auto data = load_by_path(path); data.has_value()) {
            return session_info_from_data(*data, path);
        }
        return std::nullopt;
    }

    // Old or externally produced files may put `messages` somewhere other than
    // the serialized header position. Their full JSON is already in `header`,
    // so use the canonical decoder instead of reopening the file.
    if (auto data = from_json(header); data.has_value()) {
        return session_info_from_data(*data, path);
    }
    return std::nullopt;
}

std::vector<SessionInfo> SessionStore::list_session_headers() const {
    std::vector<SessionInfo> result;
    std::error_code ec;
    if (!std::filesystem::exists(sessions_dir_, ec)) return result;

    for (const auto& entry : std::filesystem::directory_iterator(sessions_dir_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const auto& path = entry.path();
        if (path.extension() != ".json"
            || !path.filename().string().starts_with("session-")) {
            continue;
        }

        if (auto info = read_session_header(path); info.has_value()) {
            result.push_back(std::move(*info));
        }
    }

    std::ranges::sort(result, more_recent);
    return result;
}

std::optional<SessionData> SessionStore::load_by_path(
    const std::filesystem::path& path) const {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    const std::string content{
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>()};
    return from_json(content);
}

std::vector<SessionInfo> SessionStore::list() const {
    std::vector<SessionInfo> result;
    std::error_code ec;
    if (!std::filesystem::exists(sessions_dir_, ec)) return result;

    for (const auto& entry : std::filesystem::directory_iterator(sessions_dir_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const auto& p  = entry.path();
        if (p.extension() != ".json") continue;
        if (!p.filename().string().starts_with("session-")) continue;

        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        const std::string content{
            std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
        auto data = from_json(content);
        if (!data.has_value()) continue;

        result.push_back(session_info_from_data(*data, p));
    }

    // Most recently active first (ISO 8601 sorts lexicographically correctly).
    std::ranges::sort(result, more_recent);
    return result;
}

// ---------------------------------------------------------------------------
// load helpers
// ---------------------------------------------------------------------------

std::optional<SessionData> SessionStore::load_by_id(std::string_view session_id) const {
    std::error_code ec;
    if (!std::filesystem::exists(sessions_dir_, ec)) return std::nullopt;

    for (const auto& entry : std::filesystem::directory_iterator(sessions_dir_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const auto& p = entry.path();
        if (p.extension() != ".json") continue;
        const std::string fname  = p.filename().string();
        const auto dot_pos  = fname.rfind('.');
        const auto dash_pos = fname.rfind('-', dot_pos);
        if (dash_pos == std::string::npos || dash_pos + 1 >= dot_pos) continue;
        const std::string_view file_id =
            std::string_view(fname).substr(dash_pos + 1, dot_pos - dash_pos - 1);
        if (file_id != session_id) continue;

        return load_by_path(p);
    }
    return std::nullopt;
}

std::optional<SessionData> SessionStore::load_by_index(int index) const {
    if (index < 1) return std::nullopt;

    int valid_session_index = 0;
    for (const auto& info : list_session_headers()) {
        auto data = load_by_path(info.path);
        if (!data.has_value()) continue;
        if (++valid_session_index == index) return data;
    }
    return std::nullopt;
}

std::optional<SessionData> SessionStore::load_most_recent() const {
    return load_by_index(1);
}

std::optional<SessionData> SessionStore::load_most_recent_for_project(
    std::string_view working_dir) const {
    if (working_dir.empty()) return std::nullopt;

    // Normalize the requested path once so that symlinks, trailing separators,
    // and relative paths don't cause spurious mismatches against stored sessions.
    const std::filesystem::path target = canonicalize_working_dir(working_dir);

    for (const auto& info : list_session_headers()) {
        if (canonicalize_working_dir(info.working_dir) != target) continue;
        if (auto data = load_by_path(info.path); data.has_value()) {
            return data;
        }
    }
    return std::nullopt;
}

std::optional<SessionData> SessionStore::load(std::string_view id_or_index) const {
    if (id_or_index.empty()) return load_most_recent();
    // Try as integer index first.
    int idx = 0;
    const auto [ptr, ec] = std::from_chars(
        id_or_index.data(), id_or_index.data() + id_or_index.size(), idx);
    if (ec == std::errc{} && ptr == id_or_index.data() + id_or_index.size()) {
        return load_by_index(idx);
    }
    if (auto by_id = load_by_id(id_or_index); by_id.has_value()) {
        return by_id;
    }
    return load_by_name(id_or_index);
}

std::optional<SessionData> SessionStore::load_by_name(std::string_view name) const {
    if (name.empty()) return std::nullopt;
    // list() is sorted most-recent first, so the first match wins.
    for (const auto& info : list_session_headers()) {
        if (info.name != name) continue;
        if (auto data = load_by_path(info.path); data.has_value()) {
            return data;
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------

bool SessionStore::remove(std::string_view session_id, std::string* error) const {
    std::error_code ec;
    if (!std::filesystem::exists(sessions_dir_, ec)) return true;

    for (const auto& entry : std::filesystem::directory_iterator(sessions_dir_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const auto& p = entry.path();
        if (p.extension() != ".json") continue;
        const std::string fname  = p.filename().string();
        const auto dot_pos  = fname.rfind('.');
        const auto dash_pos = fname.rfind('-', dot_pos);
        if (dash_pos == std::string::npos || dash_pos + 1 >= dot_pos) continue;
        const std::string_view file_id =
            std::string_view(fname).substr(dash_pos + 1, dot_pos - dash_pos - 1);
        if (file_id != session_id) continue;

        std::string lock_error;
        auto active_lock = core::utils::InterprocessFileLock::try_acquire(
            active_session_lease_path(p), &lock_error);
        if (!active_lock) {
            if (error) {
                *error = lock_error.empty()
                    ? std::format("Session '{}' is currently open in a Filo process",
                                  session_id)
                    : lock_error;
            }
            return false;
        }
        auto file_lock = core::utils::InterprocessFileLock::acquire(
            core::utils::lock_path_for(p), &lock_error);
        if (!file_lock) {
            if (error) *error = lock_error;
            return false;
        }
        std::filesystem::remove(p, ec);
        if (ec && error) {
            *error = std::format("Cannot remove session file '{}': {}",
                                 p.string(), ec.message());
            return false;
        }
        return true;
    }
    return true; // not found — treat as success
}

} // namespace core::session
