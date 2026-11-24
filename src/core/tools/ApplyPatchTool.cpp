#include "ApplyPatchTool.hpp"
#include "CodexPatch.hpp"
#include "PatchTargets.hpp"
#include "ToolNames.hpp"
#include "shell/ShellUtils.hpp"
#include "shell/ShellExecutorFactory.hpp"
#include "shell/FsUtils.hpp"
#include "../landrun/LandrunPolicyCompiler.hpp"
#include "../landrun/LandrunSettings.hpp"
#include "../utils/JsonUtils.hpp"
#include "ToolArgumentUtils.hpp"
#include <simdjson.h>
#include <fstream>
#include <array>
#include <memory>
#include <cstdio>
#include <format>
#include <filesystem>
#include <utility>
#include <vector>
#include <algorithm>
#include <atomic>
#include <optional>

namespace core::tools {

using detail::shell_single_quote;

// Check once whether the 'patch' binary is reachable on this system.
// Prefer /usr/bin/patch (macOS base, most Linux distros) then /usr/local/bin/patch.
static bool find_patch_binary(std::string& out_path) {
    for (const char* p : {"/usr/bin/patch", "/usr/local/bin/patch"}) {
        if (access(p, X_OK) == 0) { out_path = p; return true; }
    }
    return false;
}

static const std::string kPatchBin = [] {
    std::string p;
    return find_patch_binary(p) ? p : std::string{};
}();

// Unique temp-file suffix: PID (stable per process) + monotonically increasing counter.
static std::string unique_suffix() {
    static std::atomic<unsigned> counter{0};
    return std::to_string(getpid()) + "_" + std::to_string(counter.fetch_add(1));
}

static std::optional<std::string> validate_patch_paths(
    std::string_view patch_text,
    const std::filesystem::path& base_dir,
    const core::context::SessionContext& context)
{
    for (const auto& patch_path : patch_target_paths(patch_text)) {
        const std::filesystem::path requested(patch_path);
        const std::filesystem::path resolved =
            requested.is_absolute() ? requested : (base_dir / requested);

        if (const auto access_error =
                detail::check_workspace_access(
                    resolved,
                    patch_path,
                    context,
                    nullptr,
                    names::kApplyPatch)) {
            return access_error;
        }
    }

    return std::nullopt;
}

[[nodiscard]] std::string json_error(std::string_view message) {
    return std::format(R"({{"error":"{}"}})", core::utils::escape_json_string(message));
}

[[nodiscard]] bool blank_line(std::string_view line) noexcept {
    return std::ranges::all_of(line, [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r';
    });
}

// Models of every provider sometimes wrap a tool string in a fence or a
// heredoc. Only an outer wrapper is removed; a diff that merely mentions
// those markers is left unchanged.
[[nodiscard]] std::string unwrap_patch_text(std::string_view text) {
    const std::string original(text);
    std::vector<std::string> lines;
    while (!text.empty()) {
        const auto newline = text.find('\n');
        auto line = text.substr(0, newline == std::string_view::npos ? text.size() : newline);
        if (line.ends_with('\r')) line.remove_suffix(1);
        lines.emplace_back(line);
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    std::size_t begin = 0;
    std::size_t end = lines.size();
    while (begin < end && blank_line(lines[begin])) ++begin;
    while (end > begin && blank_line(lines[end - 1])) --end;
    bool wrapped = false;
    if (end - begin >= 2 && lines[begin].starts_with("```") && lines[end - 1] == "```") {
        ++begin;
        --end;
        wrapped = true;
        while (begin < end && blank_line(lines[begin])) ++begin;
        while (end > begin && blank_line(lines[end - 1])) --end;
    }
    if (end - begin >= 2) {
        const auto& head = lines[begin];
        const bool heredoc = (head == "<<EOF" || head == "<<'EOF'" || head == "<<\"EOF\"")
            && lines[end - 1] == "EOF";
        if (heredoc) {
            ++begin;
            --end;
            wrapped = true;
        }
    }
    if (!wrapped) return original;
    std::string out;
    for (std::size_t i = begin; i < end; ++i) {
        if (!out.empty()) out += '\n';
        out += lines[i];
    }
    if (!out.empty()) out += '\n';
    return out;
}

// `rename` replaces an existing file on POSIX and fails on Windows. Stage
// beside the destination so a failure can put the previous bytes back.
[[nodiscard]] bool install_file(const std::filesystem::path& temporary,
                                const std::filesystem::path& destination,
                                const std::filesystem::path& backup,
                                bool existed,
                                std::error_code& ec) {
    if (!existed) {
        std::filesystem::rename(temporary, destination, ec);
        return !ec;
    }
    std::filesystem::remove(backup, ec);
    ec.clear();
    std::filesystem::rename(destination, backup, ec);
    if (ec) return false;
    std::filesystem::rename(temporary, destination, ec);
    if (!ec) return true;
    std::error_code restore;
    std::filesystem::rename(backup, destination, restore);
    return false;
}

void restore_installed(const std::vector<std::pair<std::filesystem::path, std::filesystem::path>>& installed,
                       const std::vector<std::pair<std::filesystem::path, std::filesystem::path>>& removed) {
    std::error_code ec;
    for (const auto& [path, backup] : removed) {
        std::filesystem::rename(backup, path, ec);
    }
    for (auto it = installed.rbegin(); it != installed.rend(); ++it) {
        const auto& [path, backup] = *it;
        std::filesystem::remove(path, ec);
        if (!backup.empty()) std::filesystem::rename(backup, path, ec);
    }
}

[[nodiscard]] std::string apply_codex_envelope(
    std::string_view patch,
    const std::filesystem::path& base_dir,
    const core::context::SessionContext& context)
{
    std::string error;
    const auto plan = plan_codex_patch(
        patch,
        [&](std::string_view relative) -> std::optional<std::string> {
            const std::filesystem::path requested{std::string(relative)};
            const auto resolved = requested.is_absolute() ? requested : base_dir / requested;
            std::error_code ec;
            if (!std::filesystem::is_regular_file(resolved, ec)) return std::nullopt;
            return detail::read_whole_file(resolved);
        },
        error);
    if (!plan) return json_error(error);

    struct ResolvedAction {
        CodexAction action;
        std::filesystem::path path;
    };
    std::vector<ResolvedAction> resolved;
    resolved.reserve(plan->actions.size());
    for (const auto& action : plan->actions) {
        const std::filesystem::path requested{action.path};
        std::filesystem::path full;
        if (const auto access_error = detail::check_workspace_access(
                requested.is_absolute() ? requested : base_dir / requested,
                action.path,
                context,
                &full,
                names::kApplyPatch)) {
            return *access_error;
        }
        resolved.push_back({action, std::move(full)});
    }

    struct Prepared {
        CodexAction::Kind kind = CodexAction::Kind::Write;
        std::string display;
        std::filesystem::path path;
        std::filesystem::path temporary;
        std::filesystem::path backup;
        bool existed = false;
    };
    std::vector<Prepared> prepared;
    prepared.reserve(resolved.size());
    std::error_code ec;
    auto discard_temps = [&] {
        for (const auto& item : prepared) {
            if (!item.temporary.empty()) std::filesystem::remove(item.temporary, ec);
        }
    };

    for (const auto& action : resolved) {
        Prepared item;
        item.kind = action.action.kind;
        item.display = action.action.path;
        item.path = action.path;
        if (item.kind == CodexAction::Kind::Remove) {
            prepared.push_back(std::move(item));
            continue;
        }
        if (item.path.has_parent_path()) {
            std::filesystem::create_directories(item.path.parent_path(), ec);
            if (ec) {
                discard_temps();
                return json_error(std::format(
                    "Could not create the directory for {}: {}", item.display, ec.message()));
            }
        }
        item.existed = std::filesystem::is_regular_file(item.path, ec);
        ec.clear();
        item.temporary = item.path.string() + ".filo-patch-tmp";
        item.backup = item.existed ? std::filesystem::path(item.path.string() + ".filo-patch-bak")
                                   : std::filesystem::path{};
        {
            std::ofstream out(item.temporary, std::ios::binary | std::ios::trunc);
            if (!out) {
                discard_temps();
                return json_error(std::format("Could not write {}.", item.display));
            }
            out << action.action.content;
            if (!out) {
                discard_temps();
                return json_error(std::format("Could not write {}.", item.display));
            }
        }
        prepared.push_back(std::move(item));
    }

    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> installed;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> removed;
    auto fail = [&](std::string message) {
        restore_installed(installed, removed);
        discard_temps();
        return json_error(std::move(message));
    };
    for (auto& item : prepared) {
        if (item.kind == CodexAction::Kind::Remove) continue;
        if (!install_file(item.temporary, item.path, item.backup, item.existed, ec)) {
            return fail(std::format("Could not replace {}: {}", item.display, ec.message()));
        }
        installed.emplace_back(item.path, item.backup);
    }
    for (auto& item : prepared) {
        if (item.kind != CodexAction::Kind::Remove) continue;
        item.backup = item.path.string() + ".filo-patch-bak";
        std::filesystem::rename(item.path, item.backup, ec);
        if (ec) return fail(std::format("Could not remove {}: {}", item.display, ec.message()));
        removed.emplace_back(item.path, item.backup);
    }
    for (const auto& [path, backup] : installed) {
        if (!backup.empty()) std::filesystem::remove(backup, ec);
    }
    for (const auto& [path, backup] : removed) {
        std::filesystem::remove(backup, ec);
    }
    return std::format(R"({{"success":true,"output":"{}"}})",
                       core::utils::escape_json_string(plan->output));
}

ToolDefinition ApplyPatchTool::get_definition() const {
    return {
        .name  = std::string(names::kApplyPatch),
        .title = "Patch",
        .description =
            "Apply a patch in the workspace. The text may be a unified diff, or an envelope "
            "that starts with *** Begin Patch and uses *** Update File, *** Add File, "
            "*** Delete File, *** Move to, and *** End Patch. working_dir defaults to the active workspace.",
        .parameters = {
            {"patch",       "string", "Unified diff, or a *** Begin Patch envelope.", true},
            {"working_dir", "string",
             "Directory in which to apply the patch; defaults to the workspace.", false}
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether the patch applied successfully."},"output":{"type":"string","description":"Combined stdout and stderr from the patch command."}},"required":["success","output"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // modifies files on disk
        },
    };
}

std::string ApplyPatchTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments provided to apply_patch."})";
    }

    std::string_view patch_view;
    if (doc["patch"].get(patch_view) != simdjson::SUCCESS) {
        return R"({"error":"Missing 'patch' argument."})";
    }
    const std::string patch_text = unwrap_patch_text(patch_view);
    patch_view = patch_text;

    // Optional working directory.
    std::string working_dir;
    std::string_view wd_view;
    if (doc["working_dir"].get(wd_view) == simdjson::SUCCESS && !wd_view.empty()) {
        working_dir = std::string(wd_view);
    }

    std::error_code ec;
    const std::filesystem::path requested_patch_base_dir = [&]() {
        if (!working_dir.empty()) {
            return std::filesystem::path(working_dir);
        }
        return std::filesystem::path(".");
    }();

    std::filesystem::path patch_base_dir;
    if (const auto access_error = detail::check_workspace_access(
            requested_patch_base_dir,
            working_dir.empty() ? std::string(".") : working_dir,
            context,
            &patch_base_dir,
            names::kApplyPatch)) {
        return *access_error;
    }

    if (!std::filesystem::is_directory(patch_base_dir, ec)) {
        return std::format(R"({{"error":"working_dir does not exist or is not a directory: '{}'"}})",
                           core::utils::escape_json_string(
                               working_dir.empty() ? patch_base_dir.string() : working_dir));
    }
    if (const auto access_error = validate_patch_paths(patch_view, patch_base_dir, context)) {
        return *access_error;
    }

    if (is_codex_patch(patch_view)) {
        return apply_codex_envelope(patch_view, patch_base_dir, context);
    }
    if (const auto diagnostic = unified_diff_diagnostic(patch_view)) {
        return std::format(R"({{"error":"{}"}})", core::utils::escape_json_string(*diagnostic));
    }

    if (kPatchBin.empty()) {
        return R"({"error":"'patch' binary not found. On macOS it is at /usr/bin/patch (always present). On Linux install it with: apt install patch  or  dnf install patch"})";
    }

    // Write patch to a collision-safe temp file.
    auto temp_root = core::landrun::LandrunSettings::instance().runtime_tmp();
    if (temp_root.empty()) {
        temp_root = std::filesystem::temp_directory_path(ec);
    }
    const auto tmp_patch_path = temp_root / ("filo_patch_" + unique_suffix() + ".diff");
    const auto reject_path = temp_root / ("filo_patch_" + unique_suffix() + ".rej");
    const std::string tmp_patch_file = tmp_patch_path.string();
    {
        std::ofstream out(tmp_patch_file);
        if (!out) {
            return R"({"error":"Failed to create temporary patch file in /tmp."})";
        }
        out << patch_view;
    }

    // -i reads the diff from a file. Feeding it on stdin makes Apple patch 2.0
    // consume the rest of the diff as the answer to "File to patch?".
    // -f refuses a reversed hunk instead of asking, and does not apply it backwards.
    std::string command = kPatchBin;
    if (!patch_base_dir.empty()) {
        command += " -d '" + shell_single_quote(patch_base_dir.string()) + "'";
    }
    command += " -p1 -f --no-backup-if-mismatch";
    command += " -i '" + shell_single_quote(tmp_patch_file) + "'";
    command += " -r '" + shell_single_quote(reject_path.string()) + "'";
    command += " 2>&1";

    auto executor = shell::make_shell_executor();
    executor->configure_landrun(core::landrun::LandrunPolicyCompiler::compile(
        context.workspace_view(),
        core::landrun::LandrunSettings::instance().mode()));
    const auto execution = executor->run(command, {}, std::chrono::seconds{30});
    std::filesystem::remove(tmp_patch_file);
    std::filesystem::remove(reject_path, ec);

    std::string escaped_output = core::utils::escape_json_string(execution.output);

    if (execution.exit_code != 0) {
        return std::format(R"({{"error":"Patch failed","output":"{}"}})", escaped_output);
    }
    return std::format(R"({{"success":true,"output":"{}"}})", escaped_output);
}

} // namespace core::tools
