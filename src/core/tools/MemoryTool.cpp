#include "MemoryTool.hpp"

#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"

#include <simdjson.h>

#include <format>
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

[[nodiscard]] std::string entries_json(const std::vector<core::memory::MemoryEntry>& entries) {
    core::utils::JsonWriter writer(512 + entries.size() * 160);
    {
        auto _ = writer.object();
        writer.kv_bool("ok", true).comma().key("entries");
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
                      .kv_bool("archived", entry.archived)
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
    return action == "remember" || action == "forget" || action == "clean";
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
            "Stores, lists, archives, and cleans durable Filo memories. Use only for stable "
            "user preferences, reusable workflows, or durable project facts after the user has "
            "enabled memory.",
        .input_schema =
            R"({"type":"object","properties":{"action":{"type":"string","enum":["remember","list","forget","clean","status"]},"content":{"type":"string","description":"Concise memory content for action=remember."},"id":{"type":"string","description":"Memory id for action=forget."},"scope":{"type":"string","description":"global, project, or session. Defaults to global."},"tags":{"type":"array","items":{"type":"string"}}},"required":["action"],"additionalProperties":false})",
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
    if (action == "status") {
        const auto settings = store_.settings();
        return std::format(
            R"({{"ok":true,"enabled":{},"auto_capture":{},"path":"{}"}})",
            settings.enabled ? "true" : "false",
            settings.auto_capture ? "true" : "false",
            core::utils::escape_json_string(store_.path().string()));
    }

    const auto settings = store_.settings();
    if (!settings.enabled) {
        return R"({"error":"Filo memory is disabled. Ask the user to run /memory on first."})";
    }

    if (action == "list") {
        return entries_json(store_.list(false));
    }
    if (action == "clean") {
        if (!context.memory_policy.generate_memories) {
            return R"({"error":"Thread memory generation is disabled."})";
        }
        return mutation_json(store_.clean());
    }
    if (action == "forget") {
        if (!context.memory_policy.generate_memories) {
            return R"({"error":"Thread memory generation is disabled."})";
        }
        return mutation_json(store_.forget(core::utils::json::string_field(object, "id")));
    }
    if (action == "remember") {
        if (!context.memory_policy.generate_memories) {
            return R"({"error":"Thread memory generation is disabled."})";
        }
        if (!settings.auto_capture) {
            return R"({"error":"Automatic memory capture is disabled. Ask the user to run /memory auto on or use /memory add."})";
        }
        const std::string scope = core::utils::json::string_field(object, "scope");
        return mutation_json(store_.remember(
            core::utils::json::string_field(object, "content"),
            scope.empty() ? "global" : scope,
            json_tags(object),
            "agent"));
    }
    return R"({"error":"Unknown memory action. Use remember, list, forget, clean, or status."})";
}

} // namespace core::tools
