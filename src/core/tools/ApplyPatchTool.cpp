#include "ApplyPatchTool.hpp"
#include "ToolNames.hpp"
#include "shell/ShellUtils.hpp"
#include "../utils/JsonUtils.hpp"
#include "ToolArgumentUtils.hpp"
#include <simdjson.h>
#include <fstream>
#include <array>
#include <memory>
#include <cstdio>
#include <format>
#include <filesystem>
#include <atomic>
#include <cctype>
#include <optional>
#include <sstream>

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

static std::string trim_ascii(std::string_view input) {
    std::size_t start = 0;
    while (start < input.size()
           && std::isspace(static_cast<unsigned char>(input[start])) != 0) {
        ++start;
    }

    std::size_t end = input.size();
    while (end > start
           && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return std::string(input.substr(start, end - start));
}

static std::optional<std::string> extract_patch_path(std::string_view line) {
    if (!(line.starts_with("--- ") || line.starts_with("+++ "))) {
        return std::nullopt;
    }

    std::string raw = trim_ascii(line.substr(4));
    if (raw.empty()) {
        return std::nullopt;
    }

    const auto tab_pos = raw.find('\t');
    if (tab_pos != std::string::npos) {
        raw.erase(tab_pos);
        raw = trim_ascii(raw);
    }

    if (raw == "/dev/null") {
        return std::nullopt;
    }

    if ((raw.starts_with("a/") || raw.starts_with("b/")) && raw.size() > 2) {
        raw.erase(0, 2);
    }

    if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') {
        raw = raw.substr(1, raw.size() - 2);
    }

    if (raw.empty()) {
        return std::nullopt;
    }
    return raw;
}

static std::optional<std::string> validate_patch_paths(
    std::string_view patch_text,
    const std::filesystem::path& base_dir,
    const core::context::SessionContext& context)
{
    std::istringstream lines{std::string(patch_text)};
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        const auto patch_path = extract_patch_path(line);
        if (!patch_path) {
            continue;
        }

        const std::filesystem::path requested(*patch_path);
        const std::filesystem::path resolved =
            requested.is_absolute() ? requested : (base_dir / requested);

        if (const auto access_error =
                detail::check_workspace_access(
                    resolved,
                    *patch_path,
                    context,
                    nullptr,
                    names::kApplyPatch)) {
            return access_error;
        }
    }

    return std::nullopt;
}

ToolDefinition ApplyPatchTool::get_definition() const {
    return {
        .name  = std::string(names::kApplyPatch),
        .title = "Apply Patch",
        .description =
            "Applies a unified diff patch to the filesystem. "
            "Requires the 'patch' utility (available by default on macOS and most Linux systems). "
            "Provide 'working_dir' so the patch is applied relative to your project root; "
            "relative paths resolve against the effective workspace root for this MCP session, "
            "and the active workspace root is used when omitted. "
            "On success returns the list of patched files in 'output'.",
        .parameters = {
            {"patch",       "string", "The unified diff patch content to apply (unified diff format).", true},
            {"working_dir", "string",
             "Absolute or relative path to the directory where 'patch -p1' should run "
             "(typically the project root). Defaults to the active workspace root.", false}
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether the patch applied successfully."},"output":{"type":"string","description":"Combined stdout and stderr from the patch command."}},"required":["success","output"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // modifies files on disk
        },
    };
}

std::string ApplyPatchTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    if (kPatchBin.empty()) {
        return R"({"error":"'patch' binary not found. On macOS it is at /usr/bin/patch (always present). On Linux install it with: apt install patch  or  dnf install patch"})";
    }

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments provided to apply_patch."})";
    }

    std::string_view patch_view;
    if (doc["patch"].get(patch_view) != simdjson::SUCCESS) {
        return R"({"error":"Missing 'patch' argument."})";
    }

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

    // Write patch to a collision-safe temp file.
    std::string tmp_patch_file = "/tmp/filo_patch_" + unique_suffix() + ".diff";
    {
        std::ofstream out(tmp_patch_file);
        if (!out) {
            return R"({"error":"Failed to create temporary patch file in /tmp."})";
        }
        out << patch_view;
    }

    // Build command: patch [-d working_dir] -p1 < tmpfile
    std::string command = kPatchBin;
    if (!patch_base_dir.empty()) {
        command += " -d '" + shell_single_quote(patch_base_dir.string()) + "'";
    }
    command += " -p1 < '" + shell_single_quote(tmp_patch_file) + "' 2>&1";

    std::array<char, 4096> buffer;
    std::string result;

    auto pclose_wrapper = [](FILE* f) { pclose(f); };
    std::unique_ptr<FILE, decltype(pclose_wrapper)> pipe(popen(command.c_str(), "r"), pclose_wrapper);
    if (!pipe) {
        std::filesystem::remove(tmp_patch_file);
        return R"({"error":"popen() failed to execute patch command."})";
    }

    size_t bytes_read;
    while ((bytes_read = std::fread(buffer.data(), 1, buffer.size(), pipe.get())) > 0) {
        result.append(buffer.data(), bytes_read);
    }

    int ret = pclose(pipe.release());
    std::filesystem::remove(tmp_patch_file);

    std::string escaped_output = core::utils::escape_json_string(result);

    if (ret != 0) {
        return std::format(R"({{"error":"Patch failed","output":"{}"}})", escaped_output);
    }
    return std::format(R"({{"success":true,"output":"{}"}})", escaped_output);
}

} // namespace core::tools
