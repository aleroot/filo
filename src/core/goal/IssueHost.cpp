#include "IssueHost.hpp"

#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <poll.h>
#include <ranges>
#include <signal.h>
#include <simdjson.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

extern char** environ;

namespace core::goal {
namespace {

using core::utils::str::to_lower_ascii_copy;
using core::utils::str::trim_ascii_copy;

constexpr std::size_t kMaxOutputBytes = 256 * 1024;

[[nodiscard]] std::string one_line(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        out.push_back(c == '\n' || c == '\r' ? ' ' : c);
    }
    return trim_ascii_copy(out);
}

[[nodiscard]] std::string clamp_text(std::string_view text, std::size_t limit) {
    if (text.size() <= limit) return std::string(text);
    std::string out(text.substr(0, limit));
    out += "...";
    return out;
}

[[nodiscard]] bool digits_only(std::string_view text) noexcept {
    return !text.empty() && std::ranges::all_of(text, [](unsigned char c) {
        return std::isdigit(c) != 0;
    });
}

[[nodiscard]] int label_priority(std::string_view label) noexcept {
    const std::string lower = to_lower_ascii_copy(label);
    if (lower == "p0" || lower == "priority-0" || lower == "critical") return 0;
    if (lower == "p1" || lower == "priority-1" || lower == "high") return 1;
    if (lower == "p2" || lower == "priority-2" || lower == "medium") return 2;
    return 3;
}

[[nodiscard]] std::expected<std::string, std::string>
require_success(const ProgramOutput& result, std::string_view what) {
    if (result.timed_out) {
        return std::unexpected(std::format("{} timed out", what));
    }
    if (result.exit_code != 0) {
        const std::string detail = trim_ascii_copy(result.output);
        return std::unexpected(detail.empty()
                                   ? std::format("{} exited {}", what, result.exit_code)
                                   : detail);
    }
    return trim_ascii_copy(result.output);
}

[[nodiscard]] std::filesystem::path worktree_directory(
    const std::filesystem::path& repository, std::string_view number) {
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(repository, error);
    const std::string key = error ? repository.string() : canonical.string();
    std::uint64_t hash = 14695981039346656037ull;
    for (const unsigned char c : key) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    return std::filesystem::temp_directory_path()
        / "filo-issue-worktrees"
        / std::format("{:08x}", static_cast<std::uint32_t>(hash))
        / std::format("issue-{}", number);
}

void set_env(std::vector<std::string>& env, std::string_view key, std::string_view value) {
    const std::string prefix = std::string(key) + "=";
    const std::string entry = prefix + std::string(value);
    for (auto& slot : env) {
        if (slot.starts_with(prefix)) {
            slot = entry;
            return;
        }
    }
    env.push_back(entry);
}

} // namespace

ProgramOutput run_program(const std::filesystem::path& cwd,
                          const std::vector<std::string>& args,
                          std::chrono::milliseconds timeout) {
    ProgramOutput result;
    if (args.empty() || args.front().empty() || timeout.count() <= 0) {
        result.output = "program arguments are empty";
        return result;
    }
#if defined(_WIN32)
    result.output = "host program execution is not available on this platform";
    return result;
#else
    int pipefd[2] = {-1, -1};
    if (::pipe(pipefd) != 0) {
        result.output = "could not create an output pipe";
        return result;
    }
    auto close_fd = [](int& fd) {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    };

    std::vector<std::string> arg_storage = args;
    std::vector<char*> argv;
    argv.reserve(arg_storage.size() + 1);
    for (auto& value : arg_storage) argv.push_back(value.data());
    argv.push_back(nullptr);

    std::vector<std::string> env_storage;
    if (environ != nullptr) {
        for (char** entry = environ; *entry != nullptr; ++entry) {
            env_storage.emplace_back(*entry);
        }
    }
    set_env(env_storage, "GIT_TERMINAL_PROMPT", "0");
    set_env(env_storage, "GIT_EDITOR", "true");
    set_env(env_storage, "GH_PROMPT_DISABLED", "1");
    set_env(env_storage, "GH_NO_UPDATE_NOTIFIER", "1");
    std::vector<char*> envp;
    envp.reserve(env_storage.size() + 1);
    for (auto& value : env_storage) envp.push_back(value.data());
    envp.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
        close_fd(pipefd[0]);
        close_fd(pipefd[1]);
        result.output = "could not initialize spawn actions";
        return result;
    }
    const std::string directory = cwd.empty() ? std::string{} : cwd.string();
    bool actions_ok =
        ::posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0) == 0
        && ::posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO) == 0
        && ::posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO) == 0
        && ::posix_spawn_file_actions_addclose(&actions, pipefd[0]) == 0
        && ::posix_spawn_file_actions_addclose(&actions, pipefd[1]) == 0;
    if (actions_ok && !directory.empty()) {
        actions_ok = ::posix_spawn_file_actions_addchdir(&actions, directory.c_str()) == 0;
    }
    posix_spawnattr_t attr;
    const bool attr_ready = ::posix_spawnattr_init(&attr) == 0;
    const bool attr_ok = attr_ready
        && ::posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP) == 0
        && ::posix_spawnattr_setpgroup(&attr, 0) == 0;
    pid_t child = -1;
    int spawn_error = EINVAL;
    if (actions_ok && attr_ok) {
        spawn_error = ::posix_spawnp(&child, argv[0], &actions, &attr, argv.data(), envp.data());
    }
    if (attr_ready) ::posix_spawnattr_destroy(&attr);
    ::posix_spawn_file_actions_destroy(&actions);
    close_fd(pipefd[1]);
    if (spawn_error != 0) {
        close_fd(pipefd[0]);
        result.output = std::format("could not start {}: {}", args.front(), std::strerror(spawn_error));
        return result;
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    bool eof = false;
    bool reaped = false;
    int status = 0;
    std::array<char, 4096> buffer{};
    while (!eof || !reaped) {
        if (std::chrono::steady_clock::now() >= deadline) {
            ::kill(-child, SIGKILL);
            ::kill(child, SIGKILL);
            while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
            close_fd(pipefd[0]);
            result.timed_out = true;
            result.exit_code = -1;
            if (!result.output.empty()) result.output.push_back('\n');
            result.output += "timed out";
            return result;
        }
        if (!reaped) {
            const pid_t waited = ::waitpid(child, &status, WNOHANG);
            if (waited == child) reaped = true;
            else if (waited < 0 && errno != EINTR) reaped = true;
        }
        if (!eof) {
            pollfd poll_fd{.fd = pipefd[0], .events = POLLIN, .revents = 0};
            const int polled = ::poll(&poll_fd, 1, 50);
            if (polled > 0 && (poll_fd.revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
                const ssize_t n = ::read(pipefd[0], buffer.data(), buffer.size());
                if (n > 0) {
                    if (result.output.size() < kMaxOutputBytes) {
                        const std::size_t room = kMaxOutputBytes - result.output.size();
                        result.output.append(buffer.data(), std::min(room, static_cast<std::size_t>(n)));
                    }
                } else if (n == 0 || (n < 0 && errno != EINTR)) {
                    eof = true;
                }
            } else if (polled < 0 && errno != EINTR) {
                eof = true;
            }
        } else if (!reaped) {
            ::usleep(20 * 1000);
        }
    }
    close_fd(pipefd[0]);
    if (WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) result.exit_code = 128 + WTERMSIG(status);
    else result.exit_code = status;
    return result;
#endif
}

std::expected<std::vector<IssueRecord>, std::string>
parse_open_issues(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string(trim_ascii_copy(json))};
    simdjson::dom::array rows;
    if (parser.parse(padded).get(rows) != simdjson::SUCCESS) {
        return std::unexpected("gh issue list did not return a JSON array");
    }
    std::vector<IssueRecord> issues;
    for (auto element : rows) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        IssueRecord issue;
        std::uint64_t number = 0;
        std::string_view text;
        if (object["number"].get(number) == simdjson::SUCCESS) {
            issue.number = std::to_string(number);
        } else if (object["number"].get(text) == simdjson::SUCCESS) {
            issue.number = std::string(text);
        }
        if (!digits_only(issue.number)) continue;
        if (object["title"].get(text) == simdjson::SUCCESS) issue.title = std::string(text);
        if (object["url"].get(text) == simdjson::SUCCESS) issue.url = std::string(text);
        if (object["body"].get(text) == simdjson::SUCCESS) {
            issue.body = clamp_text(text, 8000);
        }
        issue.priority = 3;
        simdjson::dom::array labels;
        if (object["labels"].get(labels) == simdjson::SUCCESS) {
            for (auto label : labels) {
                std::string_view name;
                simdjson::dom::object label_object;
                if (label.get(name) == simdjson::SUCCESS) {
                    issue.priority = std::min(issue.priority, label_priority(name));
                } else if (label.get(label_object) == simdjson::SUCCESS
                           && label_object["name"].get(name) == simdjson::SUCCESS) {
                    issue.priority = std::min(issue.priority, label_priority(name));
                }
            }
        }
        issues.push_back(std::move(issue));
    }
    std::ranges::sort(issues, [](const IssueRecord& left, const IssueRecord& right) {
        if (left.priority != right.priority) return left.priority < right.priority;
        return std::stoull(left.number) < std::stoull(right.number);
    });
    if (issues.size() > static_cast<std::size_t>(IssueHost::kListLimit)) {
        issues.resize(static_cast<std::size_t>(IssueHost::kListLimit));
    }
    return issues;
}

IssueHost::IssueHost(std::filesystem::path repository)
    : repository_(std::move(repository)) {}

ProgramOutput IssueHost::git(const std::vector<std::string>& args,
                             std::chrono::milliseconds timeout) const {
    std::vector<std::string> full{"git", "-C", repository_.string()};
    full.insert(full.end(), args.begin(), args.end());
    return run_program(repository_, full, timeout);
}

std::expected<std::string, std::string> IssueHost::base_ref() const {
    std::string name = "HEAD";
    if (auto origin = require_success(
            git({"symbolic-ref", "--short", "refs/remotes/origin/HEAD"}, std::chrono::seconds(15)),
            "git symbolic-ref");
        origin.has_value() && !origin->empty()) {
        name = *origin;
    } else {
        for (const char* candidate : {"main", "master"}) {
            auto verified = git(
                {"rev-parse", "--verify", "--quiet", candidate}, std::chrono::seconds(15));
            if (verified.exit_code == 0 && !trim_ascii_copy(verified.output).empty()) {
                name = candidate;
                break;
            }
        }
    }
    return require_success(
        git({"rev-parse", "--verify", name}, std::chrono::seconds(15)),
        "git rev-parse");
}

std::string IssueHost::open_pull_request(const IssueRecord& issue) const {
    if (!digits_only(issue.number)) return {};
    const auto listed = run_program(
        repository_,
        {"gh", "pr", "list", "--head", "filo/issue-" + issue.number, "--state", "open",
         "--json", "url", "--limit", "1"},
        std::chrono::seconds(20));
    if (listed.exit_code != 0) return {};
    simdjson::dom::parser parser;
    simdjson::padded_string padded{trim_ascii_copy(listed.output)};
    simdjson::dom::array rows;
    if (parser.parse(padded).get(rows) != simdjson::SUCCESS) return {};
    for (auto element : rows) {
        simdjson::dom::object object;
        std::string_view url;
        if (element.get(object) == simdjson::SUCCESS
            && object["url"].get(url) == simdjson::SUCCESS) {
            return std::string(url);
        }
        break;
    }
    return {};
}

std::expected<std::vector<IssueRecord>, std::string> IssueHost::list_open() const {
    const auto listed = run_program(
        repository_,
        {"gh", "issue", "list", "--state", "open", "--limit", std::to_string(kListLimit),
         "--json", "number,title,url,body,labels"},
        std::chrono::seconds(30));
    auto output = require_success(listed, "gh issue list");
    if (!output.has_value()) return std::unexpected(output.error());
    return parse_open_issues(*output);
}

std::expected<PreparedWorkspace, std::string>
IssueHost::prepare(const IssueRecord& issue) const {
    if (!digits_only(issue.number)) {
        return std::unexpected("issue number must be digits");
    }
    auto base = base_ref();
    if (!base.has_value()) return std::unexpected(base.error());
    const auto path = worktree_directory(repository_, issue.number);
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        return std::unexpected(std::format("could not create {}: {}", path.parent_path().string(), error.message()));
    }
    if (std::filesystem::exists(path)) {
        const auto inside = run_program(
            path,
            {"git", "-C", path.string(), "rev-parse", "--is-inside-work-tree"},
            std::chrono::seconds(15));
        const auto branch_name = run_program(
            path,
            {"git", "-C", path.string(), "branch", "--show-current"},
            std::chrono::seconds(15));
        const std::string branch = trim_ascii_copy(branch_name.output);
        if (inside.exit_code == 0 && trim_ascii_copy(inside.output) == "true" && !branch.empty()) {
            return PreparedWorkspace{.path = path, .branch = branch, .base_ref = *base};
        }
        release(PreparedWorkspace{.path = path, .branch = {}, .base_ref = *base});
    }
    const std::string branch = std::format("filo/issue-{}", issue.number);
    const auto existing = git(
        {"rev-parse", "--verify", "--quiet", "refs/heads/" + branch}, std::chrono::seconds(15));
    if (existing.exit_code == 0) {
        auto added = git({"worktree", "add", path.string(), branch}, std::chrono::seconds(60));
        if (added.exit_code == 0) {
            return PreparedWorkspace{.path = path, .branch = branch, .base_ref = *base};
        }
        std::filesystem::remove_all(path, error);
        return std::unexpected(trim_ascii_copy(added.output).empty()
                                   ? std::string("git worktree add failed")
                                   : trim_ascii_copy(added.output));
    }
    auto added = git(
        {"worktree", "add", "-b", branch, path.string(), *base}, std::chrono::seconds(60));
    if (added.exit_code == 0) {
        return PreparedWorkspace{.path = path, .branch = branch, .base_ref = *base};
    }
    std::filesystem::remove_all(path, error);
    const std::string last_error = trim_ascii_copy(added.output);
    return std::unexpected(last_error.empty() ? std::string("git worktree add failed") : last_error);
}

void IssueHost::release(const PreparedWorkspace& workspace) const {
    if (workspace.path.empty()) return;
    (void)git({"worktree", "remove", "--force", workspace.path.string()}, std::chrono::seconds(30));
    std::error_code error;
    std::filesystem::remove_all(workspace.path, error);
    (void)git({"worktree", "prune"}, std::chrono::seconds(15));
}

bool IssueHost::has_changes(const PreparedWorkspace& workspace) const {
    const auto status = run_program(
        workspace.path, {"git", "-C", workspace.path.string(), "status", "--porcelain"},
        std::chrono::seconds(15));
    if (status.exit_code == 0 && !trim_ascii_copy(status.output).empty()) return true;
    if (workspace.base_ref.empty()) return false;
    const auto ahead = run_program(
        workspace.path,
        {"git", "-C", workspace.path.string(), "rev-list", "--count",
         workspace.base_ref + "..HEAD"},
        std::chrono::seconds(15));
    return ahead.exit_code == 0 && trim_ascii_copy(ahead.output) != "0";
}

std::expected<std::vector<std::string>, std::string>
IssueHost::file_follow_ups(const IssueRecord& issue,
                           const core::review::ClassifiedFindings& findings) const {
    std::vector<std::string> urls;
    const std::size_t limit = std::min(findings.follow_ups.size(),
                                        static_cast<std::size_t>(kMaxFollowUps));
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& finding = findings.follow_ups[i];
        const std::string title = clamp_text(
            std::format("Follow-up from #{}: {}", issue.number, one_line(finding.title)), 120);
        std::string body = std::format("Found while fixing #{}.\n\n", issue.number);
        if (!finding.absolute_file_path.empty()) {
            body += std::format("{}:{}\n\n", finding.absolute_file_path, finding.line_start);
        }
        body += finding.body;
        body = clamp_text(body, 4000);
        auto created = run_program(
            repository_,
            {"gh", "issue", "create", "--title", title, "--body", body},
            std::chrono::seconds(30));
        auto url = require_success(created, "gh issue create");
        if (!url.has_value()) return std::unexpected(url.error());
        if (!url->empty()) urls.push_back(*url);
    }
    return urls;
}

std::expected<std::string, std::string>
IssueHost::publish(const IssueRecord& issue,
                   const PreparedWorkspace& workspace,
                   std::string_view summary) const {
    const auto status = run_program(
        workspace.path, {"git", "-C", workspace.path.string(), "status", "--porcelain"},
        std::chrono::seconds(15));
    if (status.exit_code != 0) {
        auto failed = require_success(status, "git status");
        return std::unexpected(failed.error());
    }
    if (!trim_ascii_copy(status.output).empty()) {
        auto added = require_success(
            run_program(workspace.path,
                        {"git", "-C", workspace.path.string(), "add", "-A"},
                        std::chrono::seconds(30)),
            "git add");
        if (!added.has_value()) return std::unexpected(added.error());
        const std::string message = clamp_text(
            std::format("Fix #{}: {}", issue.number, one_line(issue.title)), 120);
        auto committed = require_success(
            run_program(workspace.path,
                        {"git", "-C", workspace.path.string(), "commit", "-m", message},
                        std::chrono::seconds(30)),
            "git commit");
        if (!committed.has_value()) return std::unexpected(committed.error());
    }
    if (!has_changes(workspace)) {
        return std::unexpected("nothing to publish");
    }
    auto pushed = require_success(
        run_program(workspace.path,
                    {"git", "-C", workspace.path.string(), "push", "-u", "origin", "HEAD"},
                    std::chrono::seconds(120)),
        "git push");
    if (!pushed.has_value()) return std::unexpected(pushed.error());
    const std::string title = clamp_text(
        std::format("Fix #{}: {}", issue.number, one_line(issue.title)), 120);
    std::string body = std::format("Fixes #{}.\n\n", issue.number);
    body += clamp_text(summary, 2000);
    auto created = require_success(
        run_program(workspace.path,
                    {"gh", "pr", "create", "--title", title, "--body", body},
                    std::chrono::seconds(60)),
        "gh pr create");
    if (!created.has_value()) return std::unexpected(created.error());
    if (created->empty()) {
        return std::unexpected(std::string("gh pr create returned no URL"));
    }
    return *created;
}

} // namespace core::goal
