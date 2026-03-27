#include "SessionStore.hpp"
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

std::string SessionStore::now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    return std::format("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec);
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
    out += '}';
}

} // namespace

std::string SessionStore::to_json(const SessionData& data) {
    std::string out;
    out.reserve(4096 + data.messages.size() * 512);

    out += "{\"version\":";
    out += std::to_string(data.version);
    out += ",\"session_id\":\"";
    core::utils::append_escaped(out, data.session_id);
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
    out += "\",\"context_summary\":\"";
    core::utils::append_escaped(out, data.context_summary);
    out += "\",\"messages\":[";

    for (std::size_t i = 0; i < data.messages.size(); ++i) {
        if (i > 0) out += ',';
        append_message_json(out, data.messages[i]);
    }

    out += "],\"stats\":{";
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
    out += "}}";

    return out;
}

std::optional<SessionData> SessionStore::from_json(std::string_view json) {
    try {
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        if (parser.parse(json.data(), json.size()).get(doc) != simdjson::SUCCESS) {
            return std::nullopt;
        }

        SessionData data;

        int64_t version = 1;
        if (doc["version"].get(version) != simdjson::SUCCESS) {
            version = 1;
        }
        data.version = static_cast<int>(version);

        auto get_str = [&](const char* key, std::string& out) {
            std::string_view sv;
            if (doc[key].get(sv) == simdjson::SUCCESS) out = std::string(sv);
        };
        get_str("session_id",      data.session_id);
        get_str("created_at",      data.created_at);
        get_str("last_active_at",  data.last_active_at);
        get_str("working_dir",     data.working_dir);
        get_str("provider",        data.provider);
        get_str("model",           data.model);
        get_str("mode",            data.mode);
        get_str("context_summary", data.context_summary);

        simdjson::dom::array messages_arr;
        if (doc["messages"].get(messages_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element msg_el : messages_arr) {
                core::llm::Message msg;
                std::string_view sv;
                if (msg_el["role"].get(sv)         == simdjson::SUCCESS) msg.role         = std::string(sv);
                if (msg_el["content"].get(sv)      == simdjson::SUCCESS) msg.content      = std::string(sv);
                if (msg_el["name"].get(sv)         == simdjson::SUCCESS) msg.name         = std::string(sv);
                if (msg_el["tool_call_id"].get(sv) == simdjson::SUCCESS) msg.tool_call_id = std::string(sv);

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
                data.messages.push_back(std::move(msg));
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
    const auto tmp_path    = std::filesystem::path(target_path.string() + ".tmp");

    const std::string json = to_json(data);
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            if (error) *error = std::format("Cannot write session file '{}'", tmp_path.string());
            return false;
        }
        out.write(json.data(), static_cast<std::streamsize>(json.size()));
        if (!out) {
            if (error) *error = std::format("Failed to write session data to '{}'", tmp_path.string());
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, target_path, ec);
    if (ec) {
        if (error) {
            *error = std::format("Cannot finalize session file '{}': {}",
                                 target_path.string(), ec.message());
        }
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

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
        auto data_opt = from_json(content);
        if (!data_opt.has_value()) continue;

        const auto& d = *data_opt;
        SessionInfo info;
        info.session_id     = d.session_id;
        info.created_at     = d.created_at;
        info.last_active_at = d.last_active_at;
        info.provider       = d.provider;
        info.model          = d.model;
        info.mode           = d.mode;
        info.turn_count     = d.stats.turn_count;
        info.path           = p;
        result.push_back(std::move(info));
    }

    // Most recently active first (ISO 8601 sorts lexicographically correctly).
    std::ranges::sort(result, [](const SessionInfo& a, const SessionInfo& b) {
        const auto& ta = a.last_active_at.empty() ? a.created_at : a.last_active_at;
        const auto& tb = b.last_active_at.empty() ? b.created_at : b.last_active_at;
        return ta > tb;
    });
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

        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        const std::string content{
            std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
        return from_json(content);
    }
    return std::nullopt;
}

std::optional<SessionData> SessionStore::load_by_index(int index) const {
    const auto infos = list();
    if (index < 1 || index > static_cast<int>(infos.size())) return std::nullopt;
    return load_by_id(infos[static_cast<std::size_t>(index - 1)].session_id);
}

std::optional<SessionData> SessionStore::load_most_recent() const {
    return load_by_index(1);
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
    return load_by_id(id_or_index);
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
