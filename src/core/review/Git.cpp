#include "Git.hpp"

#include "../utils/StringUtils.hpp"

#include <array>
#include <cstdio>
#include <format>
#include <sstream>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace core::review {
namespace {

using core::utils::str::to_lower_ascii_copy;
using core::utils::str::trim_ascii_copy;

FILE* open_pipe_read(const char* command) {
#if defined(_WIN32)
    return _popen(command, "r");
#else
    return popen(command, "r");
#endif
}

int close_pipe(FILE* pipe) {
#if defined(_WIN32)
    return _pclose(pipe);
#else
    return pclose(pipe);
#endif
}

int normalize_pipe_exit_status(int status) {
    if (status == -1) {
        return -1;
    }
#if defined(_WIN32)
    return status;
#else
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return status;
#endif
}

bool run_command_capture(std::string_view command,
                         std::string& output,
                         std::string& error,
                         int* exit_code = nullptr) {
    output.clear();
    error.clear();

    const std::string command_str(command);
    FILE* pipe = open_pipe_read(command_str.c_str());
    if (!pipe) {
        error = std::format("could not execute `{}`", command_str);
        if (exit_code) *exit_code = -1;
        return false;
    }

    std::array<char, 4096> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = close_pipe(pipe);
    const int normalized = normalize_pipe_exit_status(status);
    if (exit_code) {
        *exit_code = normalized;
    }
    if (normalized != 0) {
        error = std::format("`{}` exited with status {}", command_str, normalized);
        return false;
    }
    return true;
}

std::string shell_quote(std::string_view value) {
#if defined(_WIN32)
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            out += "\\\"";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('"');
    return out;
#else
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('\'');
    for (char ch : value) {
        if (ch == '\'') out += "'\\''";
        else out.push_back(ch);
    }
    out.push_back('\'');
    return out;
#endif
}

std::optional<std::string> run_git_stdout(std::string_view args,
                                          std::string* error_detail = nullptr) {
    std::string output;
    std::string error;
    int exit_code = 0;
    const std::string command = std::format("git {} 2>&1", args);
    if (!run_command_capture(command, output, error, &exit_code)) {
        if (error_detail) {
            const std::string detail = trim_ascii_copy(output);
            *error_detail = detail.empty() ? error : detail;
        }
        return std::nullopt;
    }
    return trim_ascii_copy(output);
}

std::optional<std::string> run_git_stdout_allow_exit(std::string_view args,
                                                     int allowed_exit_code,
                                                     std::string* error_detail = nullptr) {
    std::string output;
    std::string error;
    int exit_code = 0;
    const std::string command = std::format("git {} 2>&1", args);
    if (run_command_capture(command, output, error, &exit_code)
        || exit_code == allowed_exit_code) {
        return trim_ascii_copy(output);
    }
    if (error_detail) {
        const std::string detail = trim_ascii_copy(output);
        *error_detail = detail.empty() ? error : detail;
    }
    return std::nullopt;
}

std::vector<std::string> split_lines(std::string_view text) {
    std::vector<std::string> lines;
    std::size_t cursor = 0;
    while (cursor <= text.size()) {
        const auto next = text.find('\n', cursor);
        const auto end = next == std::string_view::npos ? text.size() : next;
        std::string_view line = text.substr(cursor, end - cursor);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (!line.empty()) {
            lines.emplace_back(line);
        }
        if (next == std::string_view::npos) {
            break;
        }
        cursor = next + 1;
    }
    return lines;
}

std::optional<std::string> resolve_upstream_ref(std::string_view branch) {
    return run_git_stdout(std::format(
        "rev-parse --abbrev-ref --symbolic-full-name {}",
        shell_quote(std::format("{}@{{upstream}}", std::string(branch)))));
}

std::optional<std::string> build_untracked_patch(std::string_view paths,
                                                 std::string& error_detail) {
#if defined(_WIN32)
    constexpr std::string_view kNullFile = "NUL";
#else
    constexpr std::string_view kNullFile = "/dev/null";
#endif

    std::string patch;
    for (const auto& path : split_lines(paths)) {
        auto diff = run_git_stdout_allow_exit(
            std::format("diff --no-index -- {} {}",
                        shell_quote(kNullFile),
                        shell_quote(path)),
            1,
            &error_detail);
        if (!diff.has_value()) {
            return std::nullopt;
        }
        if (!patch.empty()) {
            patch += "\n\n";
        }
        patch += *diff;
    }
    return patch;
}

void split_commit_header(GitSnapshot& snapshot) {
    const auto pos = snapshot.patch.find("\ndiff --git ");
    const auto start = snapshot.patch.starts_with("diff --git ")
        ? std::string::npos
        : (pos == std::string::npos ? snapshot.patch.find("diff --git ") : pos + 1);
    if (start == std::string::npos || start == 0) {
        return;
    }
    const auto prefix = trim_ascii_copy(snapshot.patch.substr(0, start));
    if (!prefix.starts_with("commit ")) {
        return;
    }
    snapshot.commit_header = prefix;
    snapshot.patch = snapshot.patch.substr(start);
}

std::expected<GitSnapshot, std::string> collect_snapshot(
    std::string_view stat_args,
    std::string_view diff_args,
    bool include_untracked) {
    std::string error_detail;
    auto root = run_git_stdout("rev-parse --show-toplevel", &error_detail);
    if (!root.has_value()) {
        return std::unexpected(error_detail.empty()
            ? std::string("could not resolve git worktree root")
            : error_detail);
    }

    auto status = run_git_stdout("status --short", &error_detail);
    if (!status.has_value()) {
        return std::unexpected(error_detail.empty()
            ? std::string("could not read git status")
            : error_detail);
    }

    auto stat = run_git_stdout(stat_args, &error_detail);
    if (!stat.has_value()) {
        return std::unexpected(error_detail.empty()
            ? std::string("could not read git diff stat")
            : error_detail);
    }

    auto diff = run_git_stdout(diff_args, &error_detail);
    if (!diff.has_value()) {
        return std::unexpected(error_detail.empty()
            ? std::string("could not read git patch")
            : error_detail);
    }

    GitSnapshot snapshot;
    snapshot.worktree_root = std::move(*root);
    snapshot.status = std::move(*status);
    snapshot.stat = std::move(*stat);
    snapshot.patch = std::move(*diff);

    if (include_untracked) {
        auto paths = run_git_stdout("ls-files --others --exclude-standard", &error_detail);
        if (!paths.has_value()) {
            return std::unexpected(error_detail.empty()
                ? std::string("could not list untracked files")
                : error_detail);
        }
        snapshot.untracked_paths = std::move(*paths);

        if (!trim_ascii_copy(snapshot.untracked_paths).empty()) {
            auto untracked_patch = build_untracked_patch(snapshot.untracked_paths, error_detail);
            if (!untracked_patch.has_value()) {
                return std::unexpected(error_detail.empty()
                    ? std::string("could not diff untracked files")
                    : error_detail);
            }
            if (!trim_ascii_copy(*untracked_patch).empty()) {
                if (!snapshot.patch.empty() && snapshot.patch.back() != '\n') {
                    snapshot.patch.push_back('\n');
                }
                snapshot.patch += "\n# Untracked file patches\n";
                snapshot.patch += *untracked_patch;
            }
        }
    }

    split_commit_header(snapshot);
    return snapshot;
}

} // namespace

bool GitClient::is_repository(std::string* error_detail) {
    std::string local_error;
    const auto inside = run_git_stdout("rev-parse --is-inside-work-tree",
                                       error_detail ? error_detail : &local_error);
    if (!inside.has_value()) {
        return false;
    }
    return to_lower_ascii_copy(*inside) == "true";
}

std::optional<std::string> GitClient::rev_parse(std::string_view ref) {
    return run_git_stdout(std::format("rev-parse --verify {}", shell_quote(ref)));
}

std::optional<std::string> GitClient::merge_base_with_head(std::string_view branch) {
    const auto head_sha = rev_parse("HEAD");
    if (!head_sha.has_value()) {
        return std::nullopt;
    }

    auto preferred_ref = rev_parse(branch);
    if (!preferred_ref.has_value()) {
        return std::nullopt;
    }

    if (const auto upstream = resolve_upstream_ref(branch); upstream.has_value()) {
        if (const auto counts = run_git_stdout(std::format(
                "rev-list --left-right --count {}",
                shell_quote(std::format("{}...{}", std::string(branch), *upstream))));
            counts.has_value()) {
            std::stringstream ss(*counts);
            long long left = 0;
            long long right = 0;
            if (ss >> left >> right; right > 0) {
                (void)left;
                if (const auto upstream_ref = rev_parse(*upstream); upstream_ref.has_value()) {
                    preferred_ref = upstream_ref;
                }
            }
        }
    }

    return run_git_stdout(std::format("merge-base {} {}",
                                      shell_quote(*head_sha),
                                      shell_quote(*preferred_ref)));
}

std::string GitClient::uncommitted_base_ref() {
    return rev_parse("HEAD").value_or(std::string(kGitEmptyTreeSha));
}

std::expected<GitSnapshot, std::string> GitClient::collect_uncommitted() {
    const std::string base_ref = uncommitted_base_ref();
    return collect_snapshot(
        std::format("diff --stat {} --", shell_quote(base_ref)),
        std::format("diff --patch --find-renames {} --", shell_quote(base_ref)),
        true);
}

std::expected<GitSnapshot, std::string> GitClient::collect_staged() {
    return collect_snapshot(
        "diff --staged --stat --",
        "diff --staged --patch --find-renames --",
        false);
}

std::expected<GitSnapshot, std::string> GitClient::collect_against_ref(std::string_view ref) {
    return collect_snapshot(
        std::format("diff --stat {} --", shell_quote(ref)),
        std::format("diff --patch --find-renames {} --", shell_quote(ref)),
        false);
}

std::expected<GitSnapshot, std::string> GitClient::collect_commit(std::string_view sha) {
    auto snapshot = collect_snapshot(
        std::format("show --stat --format=fuller {}", shell_quote(sha)),
        std::format("show --format=fuller --patch --find-renames {}", shell_quote(sha)),
        false);
    return snapshot;
}

} // namespace core::review
