#pragma once

#include "Tool.hpp"
#include "ToolNames.hpp"
#include "../workspace/SessionWorkspace.hpp"
#include "../utils/JsonWriter.hpp"
#include <vector>
#include <string>

namespace core::tools {

/**
 * @brief Exposes the current workspace configuration (allowed paths and enforcement).
 * 
 * This tool allows an MCP client to programmatically discover which folders
 * the server is allowed to operate on.
 */
class GetWorkspaceConfigTool : public Tool {
public:
    ToolDefinition get_definition() const override {
        return {
            .name        = std::string(names::kGetWorkspaceConfig),
            .title       = "Get Workspace Configuration",
            .description = "Return active workspace roots and path-enforcement state.",
            .output_schema =
                R"({"type":"object","properties":{"primary_directory":{"type":"string","description":"The primary workspace directory."},"enforcement_enabled":{"type":"boolean","description":"Whether path enforcement is enabled for filesystem tools."},"additional_directories":{"type":"array","items":{"type":"string"},"description":"Additional allowed workspace directories."},"attached_files":{"type":"array","items":{"type":"string"},"description":"Individual files granted by attachment; present only when non-empty."},"workspace_version":{"type":"integer","description":"Monotonic version of the effective workspace selection for this session."}},"required":["primary_directory","enforcement_enabled","additional_directories","workspace_version"],"additionalProperties":false})",
            .annotations = { 
                .read_only_hint = true, 
                .idempotent_hint = true,
                .open_world_hint = false
            },
        };
    }

    std::string execute([[maybe_unused]] const std::string& json_args, const core::context::SessionContext& context) override {
        const auto& snapshot = context.effective_workspace();
        
        core::utils::JsonWriter w(512);
        {
            auto _root = w.object();
            w.kv_str("primary_directory", snapshot.primary.string()).comma()
             .kv_bool("enforcement_enabled", snapshot.enforce).comma()
             .key("additional_directories");
            {
                auto _arr = w.array();
                bool first = true;
                for (const auto& dir : core::workspace::additional_directories(snapshot)) {
                    if (!first) w.comma();
                    first = false;
                    w.str(dir.string());
                }
            }
            std::vector<std::string> attached;
            for (const auto& path : snapshot.additional) {
                if (core::workspace::is_attached_file(snapshot, path)) {
                    attached.push_back(path.string());
                }
            }
            if (!attached.empty()) {
                w.comma().key("attached_files");
                auto _arr = w.array();
                for (std::size_t i = 0; i < attached.size(); ++i) {
                    if (i > 0) w.comma();
                    w.str(attached[i]);
                }
            }
            w.comma().kv_num("workspace_version", snapshot.version);
        }
        return std::move(w).take();
    }
};

} // namespace core::tools
