#include "MutationScope.hpp"

#include "../tools/PatchTargets.hpp"
#include "../tools/ToolNames.hpp"
#include "../utils/JsonUtils.hpp"

#include <array>
#include <optional>
#include <string>

namespace core::changes {
namespace {

using Context = core::context::SessionContext;
using ScopeBuilder = MutationScope (*)(std::string_view args, const Context& context);

std::optional<std::filesystem::path> path_arg(std::string_view args,
                                              std::string_view field,
                                              const Context& context) {
    const auto value = core::utils::json::first_string_field(args, {field});
    if (!value || value->empty()) {
        return std::nullopt;
    }
    return context.resolve_path(*value);
}

MutationScope single_file(std::string_view args, const Context& context) {
    MutationScope scope;
    if (auto path = path_arg(args, "file_path", context)) {
        scope.paths.push_back(std::move(*path));
    }
    return scope;
}

MutationScope move_file(std::string_view args, const Context& context) {
    MutationScope scope;
    auto source = path_arg(args, "source", context);
    auto destination = path_arg(args, "destination", context);
    if (!source || !destination) {
        return scope;
    }
    scope.paths = {*source, *destination};
    scope.moves.emplace_back(std::move(*source), std::move(*destination));
    return scope;
}

// Resolved exactly as ApplyPatchTool resolves them: headers relative to
// `working_dir`, which itself defaults to the workspace.
MutationScope apply_patch(std::string_view args, const Context& context) {
    MutationScope scope;
    const auto patch = core::utils::json::first_string_field(args, {"patch"});
    if (!patch) {
        return scope;
    }
    const auto base = path_arg(args, "working_dir", context)
        .value_or(context.resolve_path("."));
    for (const auto& target : core::tools::patch_target_paths(*patch)) {
        const std::filesystem::path requested(target);
        scope.paths.push_back(requested.is_absolute() ? requested : base / requested);
    }
    return scope;
}

/// A tool Filo understands that cannot change what a summary reports: either it
/// writes no file at all, or the only file it writes is Filo's own state
/// outside every workspace. Creating a directory can only fail on a path that
/// is already a file; it never rewrites one, and an empty directory holds no
/// content to summarize. Memory keeps its store in Filo's config directory
/// (MemoryStore::default_path), never in a workspace, so treating it as a write
/// would flag every turn that remembered something as one whose file list may
/// be incomplete.
MutationScope inert(std::string_view, const Context&) {
    return {};
}

struct ScopeRule {
    std::string_view tool;
    ScopeBuilder build;
};

namespace names = core::tools::names;

constexpr std::array kRules{
    ScopeRule{names::kWriteFile, single_file},
    ScopeRule{names::kReplace, single_file},
    ScopeRule{names::kReplaceInFile, single_file},
    ScopeRule{names::kSearchReplace, single_file},
    ScopeRule{names::kDeleteFile, single_file},
    ScopeRule{names::kMoveFile, move_file},
    ScopeRule{names::kApplyPatch, apply_patch},
    ScopeRule{names::kCreateDirectory, inert},
    ScopeRule{names::kMemory, inert},
};

} // namespace

MutationScope mutation_scope(std::string_view tool_name,
                             std::string_view json_args,
                             const Context& context,
                             const core::tools::ToolAnnotations& annotations) {
    const auto canonical = names::canonical_alias(tool_name);
    for (const auto& rule : kRules) {
        if (rule.tool == canonical) {
            return rule.build(json_args, context);
        }
    }
    // No rule knows this tool, so fall back on what it promises. A tool that
    // declares it never writes to disk cannot change a summary; anything else
    // is treated as able to write anywhere, which the tracker then reconciles
    // and names, so a reader can tell a build from a shell command.
    if (annotations.read_only_hint) {
        return {};
    }
    return MutationScope{.unbounded = true, .tool = std::string(canonical)};
}

} // namespace core::changes
