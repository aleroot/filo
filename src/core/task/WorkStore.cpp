#include "WorkStore.hpp"

#include "../session/SessionStore.hpp"
#include "../utils/JsonUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

namespace core::task {

namespace {

[[nodiscard]] uint64_t fnv1a_64(std::string_view text) noexcept {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : text) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::string sanitize_scope_label(std::string_view text) {
    std::string out;
    out.reserve(std::min<std::size_t>(text.size(), 32));
    bool previous_dash = false;
    for (const unsigned char ch : text) {
        const bool keep = std::isalnum(ch) || ch == '-' || ch == '_';
        const char normalized = keep ? static_cast<char>(ch) : '-';
        if (normalized == '-') {
            if (previous_dash || out.empty()) continue;
            previous_dash = true;
        } else {
            previous_dash = false;
        }
        out.push_back(normalized);
        if (out.size() >= 32) break;
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out.empty() ? "workspace" : out;
}

void append_string_array(std::string& out, const std::vector<std::string>& values) {
    out.push_back('[');
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out.push_back(',');
        out.push_back('"');
        core::utils::append_escaped(out, values[i]);
        out.push_back('"');
    }
    out.push_back(']');
}

} // namespace

WorkStore::WorkStore(std::filesystem::path root_dir)
    : root_dir_(std::move(root_dir)) {}

std::filesystem::path WorkStore::default_work_dir() {
    return core::session::SessionStore::default_sessions_dir().parent_path() / "work-items";
}

std::filesystem::path WorkStore::default_work_dir(const core::context::SessionContext& context) {
    const auto primary = context.effective_workspace().primary.lexically_normal();
    const std::string scope_key = primary.empty()
        ? std::string("workspace:default")
        : primary.generic_string();
    const std::string workspace_name =
        primary.filename().empty() ? std::string("workspace") : primary.filename().string();
    const std::string scope_label = sanitize_scope_label(workspace_name);
    return default_work_dir() / std::format("{}-{:016x}", scope_label, fnv1a_64(scope_key));
}

std::string WorkStore::generate_work_id() {
    return "work_" + core::session::SessionStore::generate_id();
}

bool WorkStore::is_valid_work_id(std::string_view work_id) noexcept {
    static constexpr std::string_view kPrefix = "work_";
    if (!work_id.starts_with(kPrefix)) return false;
    if (work_id.size() != kPrefix.size() + 8) return false;

    return std::ranges::all_of(work_id.substr(kPrefix.size()), [](const unsigned char ch) {
        return std::isxdigit(ch) != 0;
    });
}

std::filesystem::path WorkStore::compute_path(std::string_view work_id) const {
    if (!is_valid_work_id(work_id)) {
        return root_dir_ / "__invalid_work_id__.json";
    }
    return root_dir_ / (std::string(work_id) + ".json");
}

bool WorkStore::ensure_dir(std::string* error) const {
    std::error_code ec;
    std::filesystem::create_directories(root_dir_, ec);
    if (!ec) return true;

    if (error != nullptr) {
        *error = std::format(
            "Cannot create work directory '{}': {}",
            root_dir_.string(),
            ec.message());
    }
    return false;
}

bool WorkStore::save(const WorkItem& item, std::string* error) const {
    if (!is_valid_work_id(item.work_id)) {
        if (error != nullptr) {
            *error = std::format("Invalid work_id '{}'.", item.work_id);
        }
        return false;
    }

    if (!ensure_dir(error)) return false;

    const auto path = compute_path(item.work_id);
    const auto temp_path = path.string() + ".tmp";
    const std::string json = to_json(item);

    {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            if (error != nullptr) {
                *error = std::format("Cannot open '{}' for writing.", temp_path);
            }
            return false;
        }
        out.write(json.data(), static_cast<std::streamsize>(json.size()));
        out.flush();
        if (!out) {
            if (error != nullptr) {
                *error = std::format("Failed to write '{}'.", temp_path);
            }
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(temp_path, path, ec);
    if (!ec) return true;

    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(temp_path, path, ec);
    if (!ec) return true;

    if (error != nullptr) {
        *error = std::format("Failed to persist work item '{}': {}", item.work_id, ec.message());
    }
    return false;
}

std::optional<WorkItem> WorkStore::load(std::string_view work_id) const {
    if (!is_valid_work_id(work_id)) return std::nullopt;

    const auto path = compute_path(work_id);
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::string json((std::istreambuf_iterator<char>(in)), {});
    return from_json(json);
}

std::vector<WorkInfo> WorkStore::list() const {
    std::vector<WorkInfo> out;
    std::error_code ec;
    if (!std::filesystem::is_directory(root_dir_, ec)) {
        return out;
    }

    for (std::filesystem::directory_iterator it(root_dir_, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        if (!it->is_regular_file(ec)) {
            ec.clear();
            continue;
        }
        if (it->path().extension() != ".json") continue;

        std::ifstream in(it->path(), std::ios::binary);
        if (!in) continue;
        std::string json((std::istreambuf_iterator<char>(in)), {});
        const auto item = from_json(json);
        if (!item.has_value()) continue;

        out.push_back(WorkInfo{
            .work_id = item->work_id,
            .title = item->title,
            .status = item->status,
            .worker = item->worker,
            .provider = item->provider,
            .model = item->model,
            .working_dir = item->working_dir,
            .updated_at = item->updated_at,
        });
    }

    std::ranges::sort(out, [](const WorkInfo& lhs, const WorkInfo& rhs) {
        return lhs.updated_at > rhs.updated_at;
    });
    return out;
}

std::string WorkStore::to_json(const WorkItem& item) {
    std::string out;
    out.reserve(2048 + item.result.size() + item.summary.size() + item.handoff_summary.size());

    out += "{\"version\":";
    out += std::to_string(item.version);
    out += ",\"work_id\":\"";
    core::utils::append_escaped(out, item.work_id);
    out += "\",\"title\":\"";
    core::utils::append_escaped(out, item.title);
    out += "\",\"status\":\"";
    core::utils::append_escaped(out, item.status);
    out += "\",\"worker\":\"";
    core::utils::append_escaped(out, item.worker);
    out += "\",\"mode\":\"";
    core::utils::append_escaped(out, item.mode);
    out += "\",\"provider\":\"";
    core::utils::append_escaped(out, item.provider);
    out += "\",\"model\":\"";
    core::utils::append_escaped(out, item.model);
    out += "\",\"working_dir\":\"";
    core::utils::append_escaped(out, item.working_dir);
    out += "\",\"session_id\":\"";
    core::utils::append_escaped(out, item.session_id);
    out += "\",\"created_at\":\"";
    core::utils::append_escaped(out, item.created_at);
    out += "\",\"updated_at\":\"";
    core::utils::append_escaped(out, item.updated_at);
    out += "\",\"summary\":\"";
    core::utils::append_escaped(out, item.summary);
    out += "\",\"result\":\"";
    core::utils::append_escaped(out, item.result);
    out += "\",\"handoff_summary\":\"";
    core::utils::append_escaped(out, item.handoff_summary);
    out += "\",\"steps\":";
    out += std::to_string(item.steps);
    out += ",\"tool_calls\":";
    out += std::to_string(item.tool_calls);
    out += ",\"failed_tool_calls\":";
    out += std::to_string(item.failed_tool_calls);
    out += ",\"files_touched\":";
    append_string_array(out, item.files_touched);
    out += ",\"commands_run\":";
    append_string_array(out, item.commands_run);
    out += '}';
    return out;
}

std::optional<WorkItem> WorkStore::from_json(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    WorkItem item;
    int64_t version = item.kVersion;
    if (doc["version"].get(version) == simdjson::SUCCESS) {
        item.version = static_cast<int>(version);
    }

    auto get_string = [&](const char* key, std::string& out) {
        std::string_view value;
        if (doc[key].get(value) == simdjson::SUCCESS) {
            out = std::string(value);
        }
    };
    get_string("work_id", item.work_id);
    get_string("title", item.title);
    get_string("status", item.status);
    get_string("worker", item.worker);
    get_string("mode", item.mode);
    get_string("provider", item.provider);
    get_string("model", item.model);
    get_string("working_dir", item.working_dir);
    get_string("session_id", item.session_id);
    get_string("created_at", item.created_at);
    get_string("updated_at", item.updated_at);
    get_string("summary", item.summary);
    get_string("result", item.result);
    get_string("handoff_summary", item.handoff_summary);

    int64_t value = 0;
    if (doc["steps"].get(value) == simdjson::SUCCESS) item.steps = static_cast<int32_t>(value);
    if (doc["tool_calls"].get(value) == simdjson::SUCCESS) item.tool_calls = static_cast<int32_t>(value);
    if (doc["failed_tool_calls"].get(value) == simdjson::SUCCESS) {
        item.failed_tool_calls = static_cast<int32_t>(value);
    }

    simdjson::dom::array array;
    if (doc["files_touched"].get(array) == simdjson::SUCCESS) {
        for (auto entry : array) {
            std::string_view text;
            if (entry.get(text) == simdjson::SUCCESS) {
                item.files_touched.emplace_back(text);
            }
        }
    }
    if (doc["commands_run"].get(array) == simdjson::SUCCESS) {
        for (auto entry : array) {
            std::string_view text;
            if (entry.get(text) == simdjson::SUCCESS) {
                item.commands_run.emplace_back(text);
            }
        }
    }

    if (!is_valid_work_id(item.work_id)) return std::nullopt;
    return item;
}

} // namespace core::task
