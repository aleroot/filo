#pragma once

#include "IssueQueue.hpp"

#include <chrono>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace core::goal {

struct ProgramOutput {
    int exit_code = -1;
    std::string output;
    bool timed_out = false;
};

// argv execution. No shell. `args.front()` is the program looked up on PATH.
// Git and gh prompts are disabled so a missing credential fails instead of hanging.
[[nodiscard]] ProgramOutput run_program(
    const std::filesystem::path& cwd,
    const std::vector<std::string>& args,
    std::chrono::milliseconds timeout = std::chrono::seconds(60));

[[nodiscard]] std::expected<std::vector<IssueRecord>, std::string>
parse_open_issues(std::string_view json);

class IssueHost {
public:
    static constexpr int kListLimit = 50;
    static constexpr int kMaxFollowUps = 8;

    explicit IssueHost(std::filesystem::path repository);

    [[nodiscard]] std::expected<std::vector<IssueRecord>, std::string> list_open() const;
    /// URL of an open PR whose head is `filo/issue-<number>`, or empty.
    [[nodiscard]] std::string open_pull_request(const IssueRecord& issue) const;
    [[nodiscard]] std::expected<PreparedWorkspace, std::string>
    prepare(const IssueRecord& issue) const;
    void release(const PreparedWorkspace& workspace) const;
    [[nodiscard]] bool has_changes(const PreparedWorkspace& workspace) const;
    [[nodiscard]] std::expected<std::vector<std::string>, std::string>
    file_follow_ups(const IssueRecord& issue,
                    const core::review::ClassifiedFindings& findings) const;
    [[nodiscard]] std::expected<std::string, std::string>
    publish(const IssueRecord& issue,
            const PreparedWorkspace& workspace,
            std::string_view summary) const;

private:
    [[nodiscard]] ProgramOutput git(const std::vector<std::string>& args,
                                    std::chrono::milliseconds timeout) const;
    [[nodiscard]] std::expected<std::string, std::string> base_ref() const;

    std::filesystem::path repository_;
};

} // namespace core::goal
