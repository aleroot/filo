#include "MemoryTool.hpp"

#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"
#include "../memory/MemoryRelevance.hpp"

#include <simdjson.h>

#include <format>
#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

namespace core::tools {
namespace {

[[nodiscard]] std::vector<std::string> json_tags(simdjson::dom::object object) {
    std::vector<std::string> tags;
    simdjson::dom::array values;
    if (object["tags"].get(values) != simdjson::SUCCESS) {
        return tags;
    }
    for (simdjson::dom::element value : values) {
        std::string_view tag;
        if (value.get(tag) != simdjson::SUCCESS) continue;
        auto clean = core::utils::str::trim_ascii_copy(tag);
        if (!clean.empty()) tags.push_back(std::move(clean));
    }
    return tags;
}

void append_capacity(core::utils::JsonWriter& writer,
                     const core::memory::MemoryScopeUsage& usage) {
    writer.raw("{").kv_str("scope", usage.scope).comma()
          .kv_num("active_entries", usage.active_entries).comma()
          .kv_num("limit", usage.limit).comma()
          .kv_num("remaining", usage.active_entries < usage.limit ? usage.limit - usage.active_entries : 0).comma()
          .kv_bool("near_capacity", usage.near_capacity()).raw("}");
}

[[nodiscard]] std::string entries_json(const std::vector<core::memory::MemoryEntry>& entries,
                                      std::size_t total, std::size_t offset) {
    core::utils::JsonWriter writer(512 + entries.size() * 160);
    {
        auto _ = writer.object();
        writer.kv_bool("ok", true).comma().kv_num("total", total).comma()
              .kv_num("offset", offset).comma()
              .kv_bool("has_more", offset < total && entries.size() < total - offset).comma()
              .key("entries");
        {
            auto _ = writer.array();
            for (std::size_t i = 0; i < entries.size(); ++i) {
                if (i > 0) writer.comma();
                const auto& entry = entries[i];
                writer.raw("{")
                      .kv_str("id", entry.id).comma()
                      .kv_str("content", entry.content).comma()
                      .kv_str("scope", entry.scope).comma()
                      .kv_str("source", entry.source).comma()
                      .kv_str("project_root", entry.project_root).comma()
                      .kv_bool("archived", entry.archived)
                      .comma().key("supersedes").raw("[");
                for (std::size_t j = 0; j < entry.supersedes.size(); ++j) {
                    if (j > 0) writer.comma();
                    writer.str(entry.supersedes[j]);
                }
                writer.raw("]")
                      .raw("}");
            }
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string mutation_json(const core::memory::MemoryMutationResult& result) {
    core::utils::JsonWriter writer(512);
    {
        auto _ = writer.object();
        writer.kv_bool("ok", result.ok).comma()
              .kv_str(result.ok ? "message" : "error", result.message);
        if (!result.code.empty()) writer.comma().kv_str("code", result.code);
        if (result.capacity) {
            writer.comma().key("capacity");
            append_capacity(writer, *result.capacity);
        }
        if (result.entry.has_value()) {
            writer.comma().key("entry");
            {
                auto _ = writer.object();
                writer.kv_str("id", result.entry->id).comma()
                      .kv_str("content", result.entry->content).comma()
                      .kv_str("scope", result.entry->scope);
            }
        }
    }
    return std::move(writer).take();
}

} // namespace

MemoryTool::MemoryTool(core::memory::MemoryStore store)
    : store_(std::move(store)) {}

bool MemoryTool::is_mutating_action(std::string_view action) noexcept {
    return action == "remember" || action == "update" || action == "merge"
        || action == "forget" || action == "clean";
}

bool MemoryTool::committed_mutation(std::string_view tool_name,
                                    std::string_view args,
                                    std::string_view result) {
    if (tool_name != names::kMemory) {
        return false;
    }
    if (!is_mutating_action(core::utils::json::string_field(args, "action"))) {
        return false;
    }
    return core::utils::json::bool_field(result, "ok", false);
}

ToolDefinition MemoryTool::get_definition() const {
    return ToolDefinition{
        .name = std::string(names::kMemory),
        .title = "Memory",
        .description =
            "Project memory (default scope: project), <=1200 UTF-8 bytes/fact. "
            "list: query/id/scope, limit=24, offset, include_archived. "
            "update: id+exact expected_content. merge: entries[{id,expected_content}], overlapping same-scope facts. "
            "Originals archived. At capacity never evict unrelated facts; skip capture and continue.",
        .input_schema =
            R"({"type":"object","properties":{"action":{"type":"string","enum":["remember","update","merge","list","forget","clean","status"]},"content":{"type":"string"},"id":{"type":"string"},"expected_content":{"type":"string"},"entries":{"type":"array","minItems":2,"items":{"type":"object","properties":{"id":{"type":"string"},"expected_content":{"type":"string"}},"required":["id","expected_content"],"additionalProperties":false}},"scope":{"type":"string","enum":["project","session"]},"tags":{"type":"array","items":{"type":"string"}},"include_archived":{"type":"boolean"},"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":50},"offset":{"type":"integer","minimum":0}},"required":["action"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint = false,
            .destructive_hint = true,
            .idempotent_hint = false,
            .open_world_hint = false,
        },
    };
}

std::string MemoryTool::execute(const std::string& json_args,
                                const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments for memory tool."})";
    }
    simdjson::dom::object object;
    if (doc.get(object) != simdjson::SUCCESS) {
        return R"({"error":"Memory tool arguments must be a JSON object."})";
    }

    const std::string action = core::utils::str::trim_ascii_copy(
        core::utils::json::string_field(object, "action"));
    const auto store = store_.for_context(context);
    std::string settings_error;
    const auto settings = store.settings(&settings_error);
    if (!settings_error.empty()) {
        return mutation_json({.ok = false, .message = settings_error});
    }
    if (action == "status") {
        const auto state = store.load(&settings_error);
        if (!settings_error.empty()) return mutation_json({.ok = false, .message = settings_error});
        core::utils::JsonWriter writer(512);
        {
            auto _ = writer.object();
            writer.kv_bool("ok", true).comma().kv_bool("enabled", state.settings.enabled).comma()
                  .kv_bool("auto_capture", state.settings.auto_capture).comma()
                  .kv_str("path", store_.path().string()).comma()
                  .kv_num("max_auto_memory_bytes", core::memory::kMaxAutoMemoryBytes).comma()
                  .key("scopes");
            auto scopes = writer.array();
            for (std::size_t i = 0; i < state.scope_usage.size(); ++i) {
                if (i > 0) writer.comma();
                append_capacity(writer, state.scope_usage[i]);
            }
        }
        return std::move(writer).take();
    }

    if (!settings.enabled) {
        return R"({"error":"Filo memory is disabled. Ask the user to run /memory auto on to enable saving memories, or /memory on for recall only."})";
    }

    if (action == "list") {
        if (!context.memory_policy.use_memories) {
            return R"({"error":"Thread memory use is disabled."})";
        }
        auto entries = store.list(core::utils::json::bool_field(object, "include_archived"), &settings_error);
        if (!settings_error.empty()) return mutation_json({.ok = false, .message = settings_error});
        const auto scope = core::utils::json::string_field(object, "scope");
        const auto id = core::utils::json::string_field(object, "id");
        if (!scope.empty() && scope != "project" && scope != "session") {
            return R"({"ok":false,"error":"Use project or session scope."})";
        }
        std::erase_if(entries, [&](const auto& entry) {
            return (!scope.empty() && entry.scope != scope) || (!id.empty() && entry.id != id);
        });
        const auto query = core::utils::json::string_field(object, "query");
        if (!query.empty()) {
            const auto ranked = core::memory::rank_memories(entries, query);
            std::vector<core::memory::MemoryEntry> ordered;
            ordered.reserve(entries.size());
            for (const auto index : ranked) ordered.push_back(std::move(entries[index]));
            entries = std::move(ordered);
        }
        const auto total = entries.size();
        const auto offset = static_cast<std::size_t>(std::max(0, core::utils::json::int_field(object, "offset")));
        const auto limit = static_cast<std::size_t>(std::clamp(core::utils::json::int_field(object, "limit", 24), 1, 50));
        if (offset >= total) entries.clear();
        else {
            entries.erase(entries.begin(), entries.begin() + offset);
            if (entries.size() > limit) entries.resize(limit);
        }
        return entries_json(entries, total, offset);
    }
    if (action == "clean") {
        if (!context.memory_policy.generate_memories) {
            return R"({"error":"Thread memory generation is disabled."})";
        }
        return mutation_json(store.clean());
    }
    if (action == "forget") {
        if (!context.memory_policy.generate_memories) {
            return R"({"error":"Thread memory generation is disabled."})";
        }
        return mutation_json(store.forget(core::utils::json::string_field(object, "id")));
    }
    if (action == "remember" || action == "update" || action == "merge") {
        if (!context.memory_policy.generate_memories) {
            return R"({"error":"Thread memory generation is disabled."})";
        }
        if (!settings.auto_capture) {
            return R"({"error":"Automatic memory capture is disabled. Ask the user to run /memory auto on or use /memory add."})";
        }
        if (action == "update" || action == "merge") {
            std::vector<core::memory::MemoryRevisionTarget> targets;
            if (action == "update") {
                targets.push_back({core::utils::json::string_field(object, "id"),
                                   core::utils::json::string_field(object, "expected_content")});
            } else {
                simdjson::dom::array values;
                if (object["entries"].get(values) != simdjson::SUCCESS || values.size() < 2) {
                    return R"({"ok":false,"code":"invalid_revision","error":"Merge requires at least two entries with id and expected_content."})";
                }
                for (auto value : values) {
                    simdjson::dom::object target;
                    if (value.get(target) != simdjson::SUCCESS) {
                        return R"({"ok":false,"code":"invalid_revision","error":"Each merge entry must be an object with id and expected_content."})";
                    }
                    targets.push_back({core::utils::json::string_field(target, "id"),
                                       core::utils::json::string_field(target, "expected_content")});
                }
            }
            return mutation_json(store.revise(targets, core::utils::json::string_field(object, "content")));
        }
        const std::string scope = core::utils::json::string_field(object, "scope");
        return mutation_json(store.remember(
            core::utils::json::string_field(object, "content"),
            scope,
            json_tags(object),
            "agent"));
    }
    return R"({"error":"Unknown memory action. Use remember, update, merge, list, forget, clean, or status."})";
}

} // namespace core::tools
