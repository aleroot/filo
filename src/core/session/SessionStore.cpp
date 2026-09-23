#include "SessionStore.hpp"

#include "ActiveSessionLease.hpp"
#include "ThreadCatalog.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/JsonUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstring>
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
#include <unordered_map>

namespace core::session {

namespace {

/// One bulk read. istreambuf_iterator costs several calls per byte, which
/// dominates loading a multi-megabyte session in unoptimized builds.
[[nodiscard]] std::optional<std::string> read_whole_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return std::nullopt;
    const std::streamoff size = in.tellg();
    if (size < 0) return std::nullopt;
    std::string content(static_cast<std::size_t>(size), '\0');
    in.seekg(0);
    in.read(content.data(), size);
    content.resize(static_cast<std::size_t>(std::max<std::streamsize>(in.gcount(), 0)));
    return content;
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

[[nodiscard]] core::llm::Message decode_message(simdjson::dom::element msg_el) {
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
    return msg;
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
                data.messages.push_back(decode_message(msg_el));
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

namespace {

// ── Bounded catalogue-row extraction ────────────────────────────────────────
//
// A catalogue row needs the header scalars, stats.turn_count and the first
// real user message. Current files store header and stats before `messages`;
// legacy files store stats as their final member. Either way the row comes
// from a bounded prefix (plus a small tail for legacy stats), so listing never
// decodes whole conversations.

constexpr std::size_t kInitialReadChunk = 16 * 1024;
constexpr std::size_t kLegacyStatsTailBytes = 4096;

/// Buffered prefix of a stream that grows on demand.
class FilePrefix {
public:
    explicit FilePrefix(std::istream& input) : input_(input) {}

    /// Makes bytes [0, size) available; false when the stream is shorter.
    [[nodiscard]] bool ensure(std::size_t size) {
        while (data_.size() < size && !eof_) {
            const std::size_t old_size = data_.size();
            const std::size_t chunk = std::max(kInitialReadChunk, old_size);
            data_.resize(old_size + chunk);
            input_.read(data_.data() + old_size, static_cast<std::streamsize>(chunk));
            const auto got = static_cast<std::size_t>(
                std::max<std::streamsize>(input_.gcount(), 0));
            data_.resize(old_size + got);
            if (got < chunk) eof_ = true;
        }
        return data_.size() >= size;
    }

    [[nodiscard]] const std::string& data() const noexcept { return data_; }

private:
    std::istream& input_;
    std::string data_;
    bool eof_ = false;
};

/// Minimal JSON cursor: it understands only enough structure to step over
/// values, and skipped payloads are scanned with memchr, never decoded.
class PrefixCursor {
public:
    explicit PrefixCursor(FilePrefix& prefix) : prefix_(prefix) {}

    [[nodiscard]] std::size_t position() const noexcept { return pos_; }
    [[nodiscard]] std::string_view slice(std::size_t begin, std::size_t end) const {
        return std::string_view(prefix_.data()).substr(begin, end - begin);
    }

    /// Next non-whitespace byte without consuming it; '\0' at end of input.
    [[nodiscard]] char peek() {
        while (prefix_.ensure(pos_ + 1)) {
            const char ch = prefix_.data()[pos_];
            if (!core::utils::json::is_whitespace(static_cast<unsigned char>(ch))) return ch;
            ++pos_;
        }
        return '\0';
    }

    [[nodiscard]] bool consume(char expected) {
        if (peek() != expected) return false;
        ++pos_;
        return true;
    }

    /// Reads `"key":` and returns the raw key text.
    [[nodiscard]] std::optional<std::string> key() {
        if (peek() != '"') return std::nullopt;
        const std::size_t begin = pos_ + 1;
        if (!skip_string()) return std::nullopt;
        std::string key{slice(begin, pos_ - 1)};
        if (!consume(':')) return std::nullopt;
        return key;
    }

    /// Steps over one value (string, scalar, object or array).
    [[nodiscard]] bool skip_value() {
        const char first = peek();
        if (first == '"') return skip_string();
        if (first != '{' && first != '[') return skip_scalar();

        std::size_t depth = 0;
        while (prefix_.ensure(pos_ + 1)) {
            const char ch = prefix_.data()[pos_];
            if (ch == '"') {
                if (!skip_string()) return false;
                continue;
            }
            ++pos_;
            if (ch == '{' || ch == '[') {
                ++depth;
            } else if (ch == '}' || ch == ']') {
                if (--depth == 0) return true;
            }
        }
        return false;
    }

private:
    [[nodiscard]] bool skip_string() {
        ++pos_;  // opening quote
        while (prefix_.ensure(pos_ + 1)) {
            const std::string& data = prefix_.data();
            const char* begin = data.data() + pos_;
            const std::size_t available = data.size() - pos_;
            const auto* quote = static_cast<const char*>(std::memchr(begin, '"', available));
            const std::size_t span = quote != nullptr
                ? static_cast<std::size_t>(quote - begin)
                : available;
            if (const auto* escape = static_cast<const char*>(std::memchr(begin, '\\', span))) {
                pos_ += static_cast<std::size_t>(escape - begin) + 2;
                continue;
            }
            pos_ += span;
            if (quote != nullptr) {
                ++pos_;
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool skip_scalar() {
        const std::size_t begin = pos_;
        while (prefix_.ensure(pos_ + 1)) {
            const char ch = prefix_.data()[pos_];
            if (ch == ',' || ch == '}' || ch == ']'
                || core::utils::json::is_whitespace(static_cast<unsigned char>(ch))) {
                break;
            }
            ++pos_;
        }
        return pos_ > begin;
    }

    FilePrefix& prefix_;
    std::size_t pos_ = 0;
};

[[nodiscard]] simdjson::error_code parse_repairing(simdjson::dom::parser& parser,
                                                   std::string_view json,
                                                   std::string& repaired,
                                                   simdjson::dom::element& out) {
    const auto error = parser.parse(json.data(), json.size()).get(out);
    if (error == simdjson::SUCCESS || simdjson::validate_utf8(json)) return error;
    repaired = core::utils::repair_utf8(json);
    return parser.parse(repaired).get(out);
}

[[nodiscard]] bool is_catalogue_header_key(std::string_view key) noexcept {
    return key == "version" || key == "session_id" || key == "name"
        || key == "created_at" || key == "last_active_at" || key == "working_dir"
        || key == "provider" || key == "model" || key == "mode" || key == "stats";
}

/// Legacy files end with `,"stats":{...}}`. Stats holds only numbers, so the
/// last '{' before the closing braces opens it.
[[nodiscard]] std::optional<std::string> read_legacy_trailing_stats(std::istream& input) {
    input.clear();
    input.seekg(0, std::ios::end);
    const auto end = static_cast<std::streamoff>(input.tellg());
    if (end <= 0) return std::nullopt;
    const auto tail_size = std::min<std::streamoff>(end, kLegacyStatsTailBytes);
    input.seekg(end - tail_size);
    std::string tail(static_cast<std::size_t>(tail_size), '\0');
    input.read(tail.data(), tail_size);
    tail.resize(static_cast<std::size_t>(std::max<std::streamsize>(input.gcount(), 0)));

    std::string_view view = tail;
    const auto strip = [&view](std::string_view suffix) {
        while (!view.empty()
               && core::utils::json::is_whitespace(static_cast<unsigned char>(view.back()))) {
            view.remove_suffix(1);
        }
        if (!view.ends_with(suffix)) return false;
        view.remove_suffix(suffix.size());
        return true;
    };
    if (!strip("}") || !strip("}")) return std::nullopt;  // root, then stats
    const std::size_t open = view.rfind('{');
    if (open == std::string_view::npos) return std::nullopt;
    std::string stats{view.substr(open)};
    stats.push_back('}');
    view = view.substr(0, open);
    if (!strip(":") || !strip("\"stats\"") || !strip(",")) return std::nullopt;
    return stats;
}

/// Builds a catalogue row from a bounded read of @p input. nullopt means the
/// layout was not recognised and the caller should use the full decoder.
[[nodiscard]] std::optional<SessionInfo> scan_catalogue_row(std::istream& input) {
    FilePrefix prefix(input);
    PrefixCursor cursor(prefix);
    if (!cursor.consume('{')) return std::nullopt;

    simdjson::dom::parser parser;
    std::string repaired;
    std::string header = "{";
    const auto append_member = [&header](std::string_view key, std::string_view raw) {
        if (header.size() > 1) header.push_back(',');
        header.push_back('"');
        header.append(key);
        header.append("\":");
        header.append(raw);
    };

    bool stats_seen = false;
    bool preview_seen = false;
    std::string preview;
    while (!(preview_seen && stats_seen)) {
        const char next = cursor.peek();
        if (next == '}') break;
        if (next == ',') {
            static_cast<void>(cursor.consume(','));
            continue;
        }
        const auto key = cursor.key();
        if (!key.has_value()) return std::nullopt;

        if (*key == "messages") {
            if (!cursor.consume('[')) return std::nullopt;
            while (!preview_seen) {
                const char ch = cursor.peek();
                if (ch == ']') {
                    static_cast<void>(cursor.consume(']'));
                    break;
                }
                if (ch == ',') {
                    static_cast<void>(cursor.consume(','));
                    continue;
                }
                const std::size_t begin = cursor.position();
                if (!cursor.skip_value()) return std::nullopt;
                simdjson::dom::element element;
                if (parse_repairing(parser, cursor.slice(begin, cursor.position()),
                                    repaired, element) != simdjson::SUCCESS) {
                    return std::nullopt;
                }
                const auto message = decode_message(element);
                if (message.role != "user" || message.synthetic) continue;
                preview = collapse_preview(core::llm::message_text_for_display(message));
                preview_seen = !preview.empty();
            }
            if (preview_seen && !stats_seen) {
                // Legacy layout: stats trail the conversation. The tail read
                // moves the stream, so any failure must use the full decoder.
                auto stats = read_legacy_trailing_stats(input);
                if (!stats.has_value()) return std::nullopt;
                append_member("stats", *stats);
                stats_seen = true;
            }
            continue;
        }

        static_cast<void>(cursor.peek());
        const std::size_t begin = cursor.position();
        if (!cursor.skip_value()) return std::nullopt;
        if (is_catalogue_header_key(*key)) {
            append_member(*key, cursor.slice(begin, cursor.position()));
            stats_seen = stats_seen || *key == "stats";
        }
    }
    header.push_back('}');

    simdjson::dom::element doc;
    if (parse_repairing(parser, header, repaired, doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    int64_t version = 1;
    if (doc["version"].get(version) != simdjson::SUCCESS) version = 1;
    if (version > SessionData::kVersion) return std::nullopt;

    SessionInfo info;
    const auto get_str = [&doc](const char* key, std::string& out) {
        std::string_view sv;
        if (doc[key].get(sv) == simdjson::SUCCESS) out = std::string(sv);
    };
    get_str("session_id",     info.session_id);
    get_str("name",           info.name);
    get_str("created_at",     info.created_at);
    get_str("last_active_at", info.last_active_at);
    get_str("working_dir",    info.working_dir);
    get_str("provider",       info.provider);
    get_str("model",          info.model);
    get_str("mode",           info.mode);
    // Unusual layouts may carry identity after the conversation.
    if (info.session_id.empty()) return std::nullopt;
    int64_t turns = 0;
    if (doc["stats"]["turn_count"].get(turns) == simdjson::SUCCESS) {
        info.turn_count = static_cast<int32_t>(turns);
    }
    info.preview = std::move(preview);
    return info;
}

// ── Persistent catalogue cache ──────────────────────────────────────────────
//
// Rows are keyed by file name and validated against (size, mtime) on every
// listing, like git's index: unchanged files cost one stat(), changed files
// one bounded scan. The cache is derived data — any damage just rebuilds it.

constexpr std::string_view kCatalogueCacheName = ".catalog-cache";
constexpr std::string_view kCatalogueMagic = "FILOCAT1";
// A file modified within the timestamp granularity of our read could change
// again without changing its mtime; such rows are served but not cached.
constexpr auto kRacyWindow = std::chrono::seconds(2);

struct CatalogueRow {
    std::uint64_t size = 0;
    std::int64_t mtime_ns = 0;
    bool valid = false;
    SessionInfo info;
};
using CatalogueCache = std::unordered_map<std::string, CatalogueRow>;

void put_u64(std::string& out, std::uint64_t value) {
    char bytes[sizeof(value)];
    std::memcpy(bytes, &value, sizeof(value));
    out.append(bytes, sizeof(value));
}

void put_string(std::string& out, std::string_view value) {
    put_u64(out, value.size());
    out.append(value);
}

class ByteReader {
public:
    explicit ByteReader(std::string_view data) : data_(data) {}

    [[nodiscard]] bool ok() const noexcept { return ok_; }

    std::uint64_t u64() {
        std::uint64_t value = 0;
        if (!ok_ || data_.size() - pos_ < sizeof(value)) {
            ok_ = false;
            return 0;
        }
        std::memcpy(&value, data_.data() + pos_, sizeof(value));
        pos_ += sizeof(value);
        return value;
    }

    std::string string() {
        const std::uint64_t size = u64();
        if (!ok_ || data_.size() - pos_ < size) {
            ok_ = false;
            return {};
        }
        std::string value{data_.substr(pos_, static_cast<std::size_t>(size))};
        pos_ += static_cast<std::size_t>(size);
        return value;
    }

private:
    std::string_view data_;
    std::size_t pos_ = 0;
    bool ok_ = true;
};

[[nodiscard]] std::string encode_catalogue_cache(const CatalogueCache& cache) {
    std::string out;
    out.reserve(64 + cache.size() * 384);
    out.append(kCatalogueMagic);
    put_u64(out, static_cast<std::uint64_t>(SessionData::kVersion));
    put_u64(out, cache.size());
    for (const auto& [file_name, row] : cache) {
        put_string(out, file_name);
        put_u64(out, row.size);
        put_u64(out, static_cast<std::uint64_t>(row.mtime_ns));
        put_u64(out, row.valid ? 1U : 0U);
        put_string(out, row.info.session_id);
        put_string(out, row.info.name);
        put_string(out, row.info.created_at);
        put_string(out, row.info.last_active_at);
        put_string(out, row.info.working_dir);
        put_string(out, row.info.provider);
        put_string(out, row.info.model);
        put_string(out, row.info.mode);
        put_string(out, row.info.preview);
        put_u64(out, static_cast<std::uint64_t>(static_cast<std::int64_t>(row.info.turn_count)));
    }
    return out;
}

[[nodiscard]] CatalogueCache read_catalogue_cache(const std::filesystem::path& path) {
    const auto file = read_whole_file(path);
    if (!file.has_value()) return {};
    const std::string& content = *file;
    if (!std::string_view(content).starts_with(kCatalogueMagic)) return {};

    ByteReader reader(std::string_view(content).substr(kCatalogueMagic.size()));
    if (reader.u64() != static_cast<std::uint64_t>(SessionData::kVersion)) return {};
    const std::uint64_t count = reader.u64();
    CatalogueCache cache;
    for (std::uint64_t i = 0; reader.ok() && i < count; ++i) {
        std::string file_name = reader.string();
        CatalogueRow row;
        row.size = reader.u64();
        row.mtime_ns = static_cast<std::int64_t>(reader.u64());
        row.valid = reader.u64() != 0;
        row.info.session_id = reader.string();
        row.info.name = reader.string();
        row.info.created_at = reader.string();
        row.info.last_active_at = reader.string();
        row.info.working_dir = reader.string();
        row.info.provider = reader.string();
        row.info.model = reader.string();
        row.info.mode = reader.string();
        row.info.preview = reader.string();
        row.info.turn_count = static_cast<int32_t>(static_cast<std::int64_t>(reader.u64()));
        cache.insert_or_assign(std::move(file_name), std::move(row));
    }
    if (!reader.ok()) return {};
    return cache;
}

} // namespace

std::optional<SessionInfo> SessionStore::read_session_header(
    const std::filesystem::path& path) const {
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) return std::nullopt;
        if (auto info = scan_catalogue_row(in); info.has_value()) {
            info->path = path;
            return info;
        }
    }
    // Unusual layouts and damaged files keep the canonical decoder's
    // compatibility and UTF-8 repair behavior.
    if (auto data = load_by_path(path); data.has_value()) {
        return session_info_from_data(*data, path);
    }
    return std::nullopt;
}

std::optional<SessionData> SessionStore::load_by_path(
    const std::filesystem::path& path) const {
    const auto content = read_whole_file(path);
    if (!content.has_value()) return std::nullopt;
    return from_json(*content);
}

std::vector<SessionInfo> SessionStore::list() const {
    std::vector<SessionInfo> result;
    std::error_code ec;
    if (!std::filesystem::exists(sessions_dir_, ec)) return result;

    const auto cache_path = sessions_dir_ / kCatalogueCacheName;
    CatalogueCache cached = read_catalogue_cache(cache_path);
    CatalogueCache fresh;
    fresh.reserve(cached.size() + 8);
    std::size_t cache_hits = 0;
    bool changed = false;
    const auto racy_after = std::filesystem::file_time_type::clock::now() - kRacyWindow;

    for (const auto& entry : std::filesystem::directory_iterator(sessions_dir_, ec)) {
        if (ec) break;
        const auto& path = entry.path();
        std::string file_name = path.filename().string();
        if (path.extension() != ".json" || !file_name.starts_with("session-")) continue;

        std::error_code stat_ec;
        const std::filesystem::directory_entry file{path, stat_ec};
        if (stat_ec || !file.is_regular_file(stat_ec)) continue;
        const std::uint64_t size = file.file_size(stat_ec);
        if (stat_ec) continue;
        const auto mtime = file.last_write_time(stat_ec);
        if (stat_ec) continue;
        const auto mtime_ns = static_cast<std::int64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                mtime.time_since_epoch()).count());

        CatalogueRow row;
        bool cacheable = true;
        if (auto it = cached.find(file_name);
            it != cached.end() && it->second.size == size
            && it->second.mtime_ns == mtime_ns) {
            row = std::move(it->second);
            ++cache_hits;
        } else {
            row.size = size;
            row.mtime_ns = mtime_ns;
            if (auto info = read_session_header(path); info.has_value()) {
                row.valid = true;
                row.info = std::move(*info);
            }
            cacheable = mtime <= racy_after;
            changed = changed || cacheable;
        }

        if (row.valid) {
            SessionInfo info = row.info;
            info.path = path;
            result.push_back(std::move(info));
        }
        if (cacheable) {
            row.info.path.clear();
            fresh.insert_or_assign(std::move(file_name), std::move(row));
        }
    }

    // Deleted files leave unmatched cached rows behind.
    if (changed || cache_hits != cached.size()) {
        static_cast<void>(core::utils::atomic_write_file(
            cache_path, encode_catalogue_cache(fresh)));
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
    for (const auto& info : list()) {
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

    for (const auto& info : list()) {
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
    for (const auto& info : list()) {
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
