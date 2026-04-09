#include "ReviewExecutor.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <simdjson.h>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace core::commands {

namespace {

std::string_view trim(std::string_view s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string trim_copy(std::string_view input) {
    return std::string(trim(input));
}

std::string join(std::string_view separator, const std::vector<std::string>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += separator;
        out += values[i];
    }
    return out;
}

std::string to_lower_ascii(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (const unsigned char ch : input) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

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

enum class ReviewTargetKind {
    UncommittedChanges,
    BaseBranch,
    Commit,
    Custom,
    StagedChanges, // Backward-compatible alias for legacy /review --staged flow.
};

struct ReviewTargetSelection {
    ReviewTargetKind kind = ReviewTargetKind::UncommittedChanges;
    std::string branch;
    std::string sha;
    std::string title;
    std::string instructions;
};

struct ParsedReviewCommand {
    bool show_help = false;
    std::string error;
    ReviewTargetSelection target;
};

struct ResolvedReviewRequest {
    ReviewTargetSelection target;
    std::string prompt;
    std::string user_facing_hint;
};

struct ReviewFindingRecord {
    std::string title;
    std::string body;
    std::optional<int> priority;
    std::string absolute_file_path;
    int line_start = 0;
    int line_end = 0;
};

struct ReviewOutputRecord {
    std::vector<ReviewFindingRecord> findings;
    std::string overall_correctness;
    std::string overall_explanation;
    std::optional<double> overall_confidence_score;
};

constexpr std::string_view kReviewFallbackMessage =
    "Reviewer failed to output a response.";

constexpr std::string_view kReviewUncommittedPrompt =
    "Review the current code changes (staged, unstaged, and untracked files) and "
    "provide prioritized findings.";

constexpr std::string_view kReviewBaseBranchPrompt =
    "Review the code changes against the base branch '{{base_branch}}'. "
    "The merge base commit for this comparison is {{merge_base_sha}}. "
    "Run `git diff {{merge_base_sha}}` to inspect the changes relative to "
    "{{base_branch}}. Provide prioritized, actionable findings.";

constexpr std::string_view kReviewBaseBranchPromptBackup =
    "Review the code changes against the base branch '{{branch}}'. Start by finding "
    "the merge diff between the current branch and {{branch}}'s upstream e.g. "
    "(`git merge-base HEAD \"$(git rev-parse --abbrev-ref \"{{branch}}@{upstream}\")\"`), "
    "then run `git diff` against that SHA to see what changes we would merge into "
    "the {{branch}} branch. Provide prioritized, actionable findings.";

constexpr std::string_view kReviewCommitPromptWithTitle =
    "Review the code changes introduced by commit {{sha}} (\"{{title}}\"). "
    "Provide prioritized, actionable findings.";

constexpr std::string_view kReviewCommitPrompt =
    "Review the code changes introduced by commit {{sha}}. "
    "Provide prioritized, actionable findings.";

constexpr std::string_view kReviewStagedPrompt =
    "Review the staged code changes only. Run `git diff --staged` and provide "
    "prioritized findings.";

constexpr std::string_view kReviewRubric = R"__FILO_REVIEW__(# Review guidelines:

You are acting as a reviewer for a proposed code change made by another engineer.

Below are some default guidelines for determining whether the original author would appreciate the issue being flagged.

These are not the final word in determining whether an issue is a bug. In many cases, you will encounter other, more specific guidelines. These may be present elsewhere in a developer message, a user message, a file, or even elsewhere in this system message.
Those guidelines should be considered to override these general instructions.

Here are the general guidelines for determining whether something is a bug and should be flagged.

1. It meaningfully impacts the accuracy, performance, security, or maintainability of the code.
2. The bug is discrete and actionable (i.e. not a general issue with the codebase or a combination of multiple issues).
3. Fixing the bug does not demand a level of rigor that is not present in the rest of the codebase (e.g. one doesn't need very detailed comments and input validation in a repository of one-off scripts in personal projects)
4. The bug was introduced in the commit (pre-existing bugs should not be flagged).
5. The author of the original PR would likely fix the issue if they were made aware of it.
6. The bug does not rely on unstated assumptions about the codebase or author's intent.
7. It is not enough to speculate that a change may disrupt another part of the codebase, to be considered a bug, one must identify the other parts of the code that are provably affected.
8. The bug is clearly not just an intentional change by the original author.

When flagging a bug, you will also provide an accompanying comment. Once again, these guidelines are not the final word on how to construct a comment -- defer to any subsequent guidelines that you encounter.

1. The comment should be clear about why the issue is a bug.
2. The comment should appropriately communicate the severity of the issue. It should not claim that an issue is more severe than it actually is.
3. The comment should be brief. The body should be at most 1 paragraph. It should not introduce line breaks within the natural language flow unless it is necessary for the code fragment.
4. The comment should not include any chunks of code longer than 3 lines. Any code chunks should be wrapped in markdown inline code tags or a code block.
5. The comment should clearly and explicitly communicate the scenarios, environments, or inputs that are necessary for the bug to arise. The comment should immediately indicate that the issue's severity depends on these factors.
6. The comment's tone should be matter-of-fact and not accusatory or overly positive. It should read as a helpful AI assistant suggestion without sounding too much like a human reviewer.
7. The comment should be written such that the original author can immediately grasp the idea without close reading.
8. The comment should avoid excessive flattery and comments that are not helpful to the original author. The comment should avoid phrasing like "Great job ...", "Thanks for ...".

Below are some more detailed guidelines that you should apply to this specific review.

HOW MANY FINDINGS TO RETURN:

Output all findings that the original author would fix if they knew about it. If there is no finding that a person would definitely love to see and fix, prefer outputting no findings. Do not stop at the first qualifying finding. Continue until you've listed every qualifying finding.

GUIDELINES:

- Ignore trivial style unless it obscures meaning or violates documented standards.
- Use one comment per distinct issue (or a multi-line range if necessary).
- Use ```suggestion blocks ONLY for concrete replacement code (minimal lines; no commentary inside the block).
- In every ```suggestion block, preserve the exact leading whitespace of the replaced lines (spaces vs tabs, number of spaces).
- Do NOT introduce or remove outer indentation levels unless that is the actual fix.

The comments will be presented in the code review as inline comments. You should avoid providing unnecessary location details in the comment body. Always keep the line range as short as possible for interpreting the issue. Avoid ranges longer than 5-10 lines; instead, choose the most suitable subrange that pinpoints the problem.

At the beginning of the finding title, tag the bug with priority level. For example "[P1] Un-padding slices along wrong tensor dimensions". [P0] - Drop everything to fix. Blocking release, operations, or major usage. Only use for universal issues that do not depend on any assumptions about the inputs. [P1] - Urgent. Should be addressed in the next cycle [P2] - Normal. To be fixed eventually [P3] - Low. Nice to have.

Additionally, include a numeric priority field in the JSON output for each finding: set "priority" to 0 for P0, 1 for P1, 2 for P2, or 3 for P3. If a priority cannot be determined, omit the field or use null.

At the end of your findings, output an "overall correctness" verdict of whether or not the patch should be considered "correct".
Correct implies that existing code and tests will not break, and the patch is free of bugs and other blocking issues.
Ignore non-blocking issues such as style, formatting, typos, documentation, and other nits.

FORMATTING GUIDELINES:
The finding description should be one paragraph.

OUTPUT FORMAT:

## Output schema  - MUST MATCH exactly

```json
{
  "findings": [
    {
      "title": "<= 80 chars, imperative",
      "body": "<valid Markdown explaining why this is a problem; cite files/lines/functions>",
      "confidence_score": <float 0.0-1.0>,
      "priority": <int 0-3, optional>,
      "code_location": {
        "absolute_file_path": "<file path>",
        "line_range": {"start": <int>, "end": <int>}
      }
    }
  ],
  "overall_correctness": "patch is correct" | "patch is incorrect",
  "overall_explanation": "<1-3 sentence explanation justifying the overall_correctness verdict>",
  "overall_confidence_score": <float 0.0-1.0>
}
```

* Do not wrap the JSON in markdown fences or extra prose.
* The code_location field is required and must include absolute_file_path and line_range.
* Line ranges must be as short as possible for interpreting the issue (avoid ranges over 5-10 lines; pick the most suitable subrange).
* The code_location should overlap with the diff.
* Do not generate a PR fix.
)__FILO_REVIEW__";

std::string review_usage_text() {
    return
        "\n\xe2\x84\xb9  Usage: /review [--uncommitted | --base <branch> | --commit <sha> [--title <title>] | <custom instructions>]\n"
        "   /review                      Review current changes (staged, unstaged, and untracked).\n"
        "   /review --base main          Review changes against a base branch.\n"
        "   /review --commit <sha>       Review a specific commit.\n"
        "   /review --commit <sha> --title \"Subject\"\n"
        "                               Review a commit and show a custom title.\n"
        "   /review <instructions>       Run a custom review prompt.\n"
        "   /review --staged             Legacy alias: review staged changes only.\n";
}

std::optional<std::vector<std::string>> split_command_arguments(
    std::string_view input,
    std::string& error)
{
    std::vector<std::string> tokens;
    std::string current;
    bool in_single = false;
    bool in_double = false;
    bool escaping = false;

    for (char ch : input) {
        if (escaping) {
            current.push_back(ch);
            escaping = false;
            continue;
        }

        if (ch == '\\' && !in_single) {
            escaping = true;
            continue;
        }
        if (in_single) {
            if (ch == '\'') in_single = false;
            else current.push_back(ch);
            continue;
        }
        if (in_double) {
            if (ch == '"') in_double = false;
            else current.push_back(ch);
            continue;
        }
        if (ch == '\'') {
            in_single = true;
            continue;
        }
        if (ch == '"') {
            in_double = true;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }

    if (escaping) {
        current.push_back('\\');
    }
    if (in_single || in_double) {
        error = "Unterminated quote in /review arguments.";
        return std::nullopt;
    }
    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }
    return tokens;
}

std::string take_short_sha(std::string_view sha) {
    constexpr std::size_t kShortShaLen = 7;
    if (sha.size() <= kShortShaLen) {
        return std::string(sha);
    }
    return std::string(sha.substr(0, kShortShaLen));
}

ParsedReviewCommand parse_review_command(std::string_view raw_args) {
    ParsedReviewCommand parsed;
    parsed.target.kind = ReviewTargetKind::UncommittedChanges;

    const std::string trimmed_args = trim_copy(raw_args);
    if (trimmed_args.empty()) {
        return parsed;
    }

    std::string token_error;
    const auto tokens_opt = split_command_arguments(trimmed_args, token_error);
    if (!tokens_opt.has_value()) {
        parsed.error = token_error;
        return parsed;
    }

    const auto& tokens = *tokens_opt;
    if (tokens.empty()) {
        return parsed;
    }

    if (tokens.size() == 1 && (tokens[0] == "-h" || tokens[0] == "--help")) {
        parsed.show_help = true;
        return parsed;
    }

    bool staged = false;
    bool uncommitted = false;
    std::optional<std::string> base_branch;
    std::optional<std::string> commit_sha;
    std::optional<std::string> commit_title;
    std::vector<std::string> positional;
    bool end_of_options = false;

    auto consume_option_value = [&](std::size_t& index,
                                    std::string_view option,
                                    std::string_view token) -> std::optional<std::string> {
        const std::string option_str(option);
        const std::string prefix = option_str + "=";

        if (token == option) {
            if (index + 1 >= tokens.size()) {
                parsed.error = std::format("Missing value for {}.", option_str);
                return std::nullopt;
            }
            ++index;
            return tokens[index];
        }
        if (token.starts_with(prefix)) {
            const std::string value = std::string(token.substr(prefix.size()));
            if (value.empty()) {
                parsed.error = std::format("Missing value for {}.", option_str);
                return std::nullopt;
            }
            return value;
        }
        return std::nullopt;
    };

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::string_view token = tokens[i];

        if (end_of_options) {
            positional.push_back(std::string(token));
            continue;
        }
        if (token == "--") {
            end_of_options = true;
            continue;
        }
        if (token == "--staged" || token == "--cached") {
            staged = true;
            continue;
        }
        if (token == "--uncommitted") {
            uncommitted = true;
            continue;
        }
        if (token == "--base" || token.starts_with("--base=")) {
            if (base_branch.has_value()) {
                parsed.error = "Duplicate --base option.";
                return parsed;
            }
            auto value = consume_option_value(i, "--base", token);
            if (!value.has_value()) {
                if (!parsed.error.empty()) return parsed;
                parsed.error = "Missing value for --base.";
                return parsed;
            }
            base_branch = std::move(*value);
            continue;
        }
        if (token == "--commit" || token.starts_with("--commit=")) {
            if (commit_sha.has_value()) {
                parsed.error = "Duplicate --commit option.";
                return parsed;
            }
            auto value = consume_option_value(i, "--commit", token);
            if (!value.has_value()) {
                if (!parsed.error.empty()) return parsed;
                parsed.error = "Missing value for --commit.";
                return parsed;
            }
            commit_sha = std::move(*value);
            continue;
        }
        if (token == "--title" || token.starts_with("--title=")) {
            if (commit_title.has_value()) {
                parsed.error = "Duplicate --title option.";
                return parsed;
            }
            auto value = consume_option_value(i, "--title", token);
            if (!value.has_value()) {
                if (!parsed.error.empty()) return parsed;
                parsed.error = "Missing value for --title.";
                return parsed;
            }
            commit_title = std::move(*value);
            continue;
        }
        if (token.starts_with('-')) {
            parsed.error = std::format("Unknown /review option '{}'.", std::string(token));
            return parsed;
        }
        positional.push_back(std::string(token));
    }

    if (commit_title.has_value() && !commit_sha.has_value()) {
        parsed.error = "--title requires --commit.";
        return parsed;
    }

    int mode_count = 0;
    mode_count += staged ? 1 : 0;
    mode_count += uncommitted ? 1 : 0;
    mode_count += base_branch.has_value() ? 1 : 0;
    mode_count += commit_sha.has_value() ? 1 : 0;

    if (mode_count > 1) {
        parsed.error = "Choose only one review target: --uncommitted, --base, --commit, or --staged.";
        return parsed;
    }
    if (mode_count > 0 && !positional.empty()) {
        parsed.error = "Cannot combine custom instructions with --uncommitted/--base/--commit/--staged.";
        return parsed;
    }

    if (mode_count == 0) {
        if (!positional.empty()) {
            parsed.target.kind = ReviewTargetKind::Custom;
            parsed.target.instructions = join(" ", positional);
        }
        return parsed;
    }

    if (staged) {
        parsed.target.kind = ReviewTargetKind::StagedChanges;
        return parsed;
    }
    if (uncommitted) {
        parsed.target.kind = ReviewTargetKind::UncommittedChanges;
        return parsed;
    }
    if (base_branch.has_value()) {
        parsed.target.kind = ReviewTargetKind::BaseBranch;
        parsed.target.branch = std::move(*base_branch);
        return parsed;
    }
    if (commit_sha.has_value()) {
        parsed.target.kind = ReviewTargetKind::Commit;
        parsed.target.sha = std::move(*commit_sha);
        parsed.target.title = commit_title.value_or("");
        return parsed;
    }

    parsed.error = "Unable to parse /review arguments.";
    return parsed;
}

std::string shell_quote(std::string_view value) {
#if defined(_WIN32)
    // _popen() runs through cmd.exe on Windows, where single quotes are
    // not argument delimiters. Use double-quoted args so refs like "HEAD"
    // are interpreted correctly.
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
            const std::string detail = trim_copy(output);
            *error_detail = detail.empty() ? error : detail;
        }
        return std::nullopt;
    }
    return trim_copy(output);
}

bool is_git_repository(std::string& error_detail) {
    const auto inside = run_git_stdout("rev-parse --is-inside-work-tree", &error_detail);
    if (!inside.has_value()) {
        return false;
    }
    return to_lower_ascii(trim_copy(*inside)) == "true";
}

std::optional<std::string> resolve_git_ref(std::string_view ref_name) {
    return run_git_stdout(std::format("rev-parse --verify {}", shell_quote(ref_name)));
}

std::optional<std::string> resolve_upstream_ref(std::string_view branch) {
    return run_git_stdout(std::format(
        "rev-parse --abbrev-ref --symbolic-full-name {}",
        shell_quote(std::format("{}@{{upstream}}", std::string(branch)))));
}

std::optional<std::string> resolve_merge_base_with_head(std::string_view branch) {
    const auto head_sha = resolve_git_ref("HEAD");
    if (!head_sha.has_value()) {
        return std::nullopt;
    }

    auto preferred_ref = resolve_git_ref(branch);
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
                if (const auto upstream_ref = resolve_git_ref(*upstream);
                    upstream_ref.has_value()) {
                    preferred_ref = upstream_ref;
                }
            }
        }
    }

    return run_git_stdout(std::format("merge-base {} {}",
                                      shell_quote(*head_sha),
                                      shell_quote(*preferred_ref)));
}

std::string render_template(
    std::string_view input,
    std::initializer_list<std::pair<std::string_view, std::string_view>> vars)
{
    std::string out(input);
    for (const auto& [key, value] : vars) {
        const std::string needle = std::format("{{{{{}}}}}", key);
        std::size_t pos = 0;
        while ((pos = out.find(needle, pos)) != std::string::npos) {
            out.replace(pos, needle.size(), value);
            pos += value.size();
        }
    }
    return out;
}

std::string review_user_hint(const ReviewTargetSelection& target) {
    switch (target.kind) {
        case ReviewTargetKind::UncommittedChanges:
            return "current changes";
        case ReviewTargetKind::BaseBranch:
            return std::format("changes against '{}'", target.branch);
        case ReviewTargetKind::Commit:
            if (!trim(target.title).empty()) {
                return std::format("commit {}: {}", take_short_sha(target.sha), target.title);
            }
            return std::format("commit {}", take_short_sha(target.sha));
        case ReviewTargetKind::Custom:
            return trim_copy(target.instructions);
        case ReviewTargetKind::StagedChanges:
            return "staged changes";
    }
    return "current changes";
}

std::optional<ResolvedReviewRequest> resolve_review_request(
    const ReviewTargetSelection& target,
    std::string& error_detail)
{
    ResolvedReviewRequest resolved;
    resolved.target = target;
    resolved.user_facing_hint = review_user_hint(target);

    switch (target.kind) {
        case ReviewTargetKind::UncommittedChanges:
            resolved.prompt = std::string(kReviewUncommittedPrompt);
            return resolved;
        case ReviewTargetKind::StagedChanges:
            resolved.prompt = std::string(kReviewStagedPrompt);
            return resolved;
        case ReviewTargetKind::BaseBranch: {
            if (trim(target.branch).empty()) {
                error_detail = "Base branch name cannot be empty.";
                return std::nullopt;
            }
            const auto merge_base = resolve_merge_base_with_head(target.branch);
            if (merge_base.has_value()) {
                resolved.prompt = render_template(
                    kReviewBaseBranchPrompt,
                    {
                        {"base_branch", target.branch},
                        {"merge_base_sha", *merge_base},
                    });
            } else {
                resolved.prompt = render_template(
                    kReviewBaseBranchPromptBackup,
                    {
                        {"branch", target.branch},
                    });
            }
            return resolved;
        }
        case ReviewTargetKind::Commit:
            if (trim(target.sha).empty()) {
                error_detail = "Commit SHA cannot be empty.";
                return std::nullopt;
            }
            if (!trim(target.title).empty()) {
                resolved.prompt = render_template(
                    kReviewCommitPromptWithTitle,
                    {
                        {"sha", target.sha},
                        {"title", target.title},
                    });
            } else {
                resolved.prompt = render_template(
                    kReviewCommitPrompt,
                    {
                        {"sha", target.sha},
                    });
            }
            return resolved;
        case ReviewTargetKind::Custom: {
            const std::string prompt = trim_copy(target.instructions);
            if (prompt.empty()) {
                error_detail = "Review prompt cannot be empty.";
                return std::nullopt;
            }
            resolved.prompt = prompt;
            return resolved;
        }
    }

    error_detail = "Unknown review target.";
    return std::nullopt;
}

std::string build_review_submission_prompt(const ResolvedReviewRequest& request) {
    std::string prompt;
    prompt.reserve(request.prompt.size() + std::size(kReviewRubric) + 512);
    prompt += "You are running Filo's standalone /review task.\n";
    prompt += "Use available tools to inspect the relevant git changes and code.\n";
    prompt += "Return only valid JSON that matches the schema below.\n";
    prompt += "Do not wrap JSON in markdown fences.\n\n";
    prompt += "[Review task]\n";
    prompt += request.prompt;
    if (!request.prompt.empty() && request.prompt.back() != '\n') {
        prompt.push_back('\n');
    }
    prompt += "\n";
    prompt += kReviewRubric;
    return prompt;
}

std::optional<std::string> object_string_field(const simdjson::dom::object& object, const char* key) {
    std::string_view value;
    if (object[key].get(value) == simdjson::SUCCESS) {
        return std::string(value);
    }
    return std::nullopt;
}

std::optional<std::int64_t> element_to_int64(const simdjson::dom::element& element) {
    std::int64_t i = 0;
    if (element.get(i) == simdjson::SUCCESS) {
        return i;
    }

    std::uint64_t u = 0;
    if (element.get(u) == simdjson::SUCCESS) {
        if (u <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return static_cast<std::int64_t>(u);
        }
        return std::nullopt;
    }

    double d = 0.0;
    if (element.get(d) == simdjson::SUCCESS) {
        return static_cast<std::int64_t>(d);
    }
    return std::nullopt;
}

std::optional<double> element_to_double(const simdjson::dom::element& element) {
    double d = 0.0;
    if (element.get(d) == simdjson::SUCCESS) {
        return d;
    }

    std::int64_t i = 0;
    if (element.get(i) == simdjson::SUCCESS) {
        return static_cast<double>(i);
    }

    std::uint64_t u = 0;
    if (element.get(u) == simdjson::SUCCESS) {
        return static_cast<double>(u);
    }
    return std::nullopt;
}

std::optional<std::int64_t> object_int_field(const simdjson::dom::object& object, const char* key) {
    simdjson::dom::element element;
    if (object[key].get(element) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return element_to_int64(element);
}

std::optional<double> object_double_field(const simdjson::dom::object& object, const char* key) {
    simdjson::dom::element element;
    if (object[key].get(element) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return element_to_double(element);
}

std::optional<ReviewOutputRecord> try_parse_review_output_json(std::string_view text) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(text).get(root) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    simdjson::dom::object object;
    if (root.get(object) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    ReviewOutputRecord out;
    if (const auto value = object_string_field(object, "overall_correctness"); value.has_value()) {
        out.overall_correctness = *value;
    }
    if (const auto value = object_string_field(object, "overall_explanation"); value.has_value()) {
        out.overall_explanation = *value;
    }
    out.overall_confidence_score = object_double_field(object, "overall_confidence_score");

    simdjson::dom::array findings;
    if (object["findings"].get(findings) == simdjson::SUCCESS) {
        for (const auto finding_value : findings) {
            simdjson::dom::object finding_object;
            if (finding_value.get(finding_object) != simdjson::SUCCESS) {
                continue;
            }

            ReviewFindingRecord finding;
            if (const auto value = object_string_field(finding_object, "title"); value.has_value()) {
                finding.title = *value;
            }
            if (const auto value = object_string_field(finding_object, "body"); value.has_value()) {
                finding.body = *value;
            }
            if (const auto value = object_int_field(finding_object, "priority"); value.has_value()) {
                finding.priority = static_cast<int>(*value);
            }

            simdjson::dom::object location_object;
            if (finding_object["code_location"].get(location_object) == simdjson::SUCCESS) {
                if (const auto value =
                        object_string_field(location_object, "absolute_file_path");
                    value.has_value()) {
                    finding.absolute_file_path = *value;
                }

                simdjson::dom::object line_range;
                if (location_object["line_range"].get(line_range) == simdjson::SUCCESS) {
                    if (const auto value = object_int_field(line_range, "start");
                        value.has_value()) {
                        finding.line_start = static_cast<int>(*value);
                    }
                    if (const auto value = object_int_field(line_range, "end");
                        value.has_value()) {
                        finding.line_end = static_cast<int>(*value);
                    }
                }
            }

            out.findings.push_back(std::move(finding));
        }
    }

    return out;
}

ReviewOutputRecord parse_review_output_event(std::string_view text) {
    if (auto parsed = try_parse_review_output_json(text); parsed.has_value()) {
        return *parsed;
    }

    if (const auto start = text.find('{'); start != std::string_view::npos) {
        if (const auto end = text.rfind('}'); end != std::string_view::npos && start < end) {
            if (auto parsed = try_parse_review_output_json(text.substr(start, end - start + 1));
                parsed.has_value()) {
                return *parsed;
            }
        }
    }

    ReviewOutputRecord fallback;
    fallback.overall_explanation = std::string(text);
    return fallback;
}

std::string format_review_location(const ReviewFindingRecord& finding) {
    const std::string path = finding.absolute_file_path.empty()
        ? std::string("<unknown-file>")
        : finding.absolute_file_path;
    return std::format("{}:{}-{}", path, finding.line_start, finding.line_end);
}

std::string format_review_findings_block(const std::vector<ReviewFindingRecord>& findings) {
    std::vector<std::string> lines;
    lines.reserve(findings.size() * 4 + 2);
    lines.push_back("");
    lines.push_back(findings.size() > 1 ? "Full review comments:" : "Review comment:");

    for (const auto& finding : findings) {
        lines.push_back("");
        lines.push_back(std::format(
            "- {} - {}",
            finding.title.empty() ? std::string("Untitled finding") : finding.title,
            format_review_location(finding)));
        if (!finding.body.empty()) {
            std::string_view body_view(finding.body);
            std::size_t cursor = 0;
            while (cursor <= body_view.size()) {
                const std::size_t end = body_view.find('\n', cursor);
                const std::string_view line = end == std::string_view::npos
                    ? body_view.substr(cursor)
                    : body_view.substr(cursor, end - cursor);
                lines.push_back(std::format("  {}", std::string(line)));
                if (end == std::string_view::npos) break;
                cursor = end + 1;
            }
        }
    }
    return join("\n", lines);
}

std::string render_review_output_text(const ReviewOutputRecord& output) {
    std::vector<std::string> sections;
    const std::string explanation = trim_copy(output.overall_explanation);
    if (!explanation.empty()) {
        sections.push_back(explanation);
    }
    if (!output.findings.empty()) {
        const std::string block = trim_copy(format_review_findings_block(output.findings));
        if (!block.empty()) {
            sections.push_back(block);
        }
    }
    if (sections.empty()) {
        return std::string(kReviewFallbackMessage);
    }
    return join("\n\n", sections);
}

} // namespace

void ReviewExecutor::execute(const CommandContext& ctx, std::string_view raw_args) {
    const ParsedReviewCommand parsed = parse_review_command(raw_args);
    if (parsed.show_help) {
        ctx.append_history_fn(review_usage_text());
        return;
    }
    if (!parsed.error.empty()) {
        ctx.append_history_fn(std::format(
            "\n✗  {}\n{}",
            parsed.error,
            review_usage_text()));
        return;
    }

    if (!ctx.agent) {
        ctx.append_history_fn("\n✗  Review requires an active agent session.\n");
        return;
    }

    if (parsed.target.kind != ReviewTargetKind::Custom) {
        std::string git_error;
        if (!is_git_repository(git_error)) {
            const std::string detail = trim_copy(git_error);
            ctx.append_history_fn(std::format(
                "\n✗  /review requires a git repository. {}\n",
                detail.empty() ? std::string() : std::format("({})", detail)));
            return;
        }
    }

    std::string resolve_error;
    auto resolved = resolve_review_request(parsed.target, resolve_error);
    if (!resolved.has_value()) {
        ctx.append_history_fn(std::format(
            "\n✗  Could not start review: {}\n",
            resolve_error.empty() ? std::string("unknown review setup error") : resolve_error));
        return;
    }

    const std::string prompt = build_review_submission_prompt(*resolved);

    const std::string review_hint = trim_copy(resolved->user_facing_hint);
    ctx.append_history_fn(std::format("\n»  Code review started: {}…\n",
                                      review_hint.empty()
                                          ? std::string("current changes")
                                          : review_hint));

    auto append_fn = ctx.append_history_fn;
    auto collected_output = std::make_shared<std::string>();
    auto interrupted = std::make_shared<bool>(false);
    ctx.agent->send_message(
        prompt,
        [collected_output, interrupted](const std::string& chunk) {
            *collected_output += chunk;
            if (chunk.find("[Generation stopped by user]") != std::string::npos) {
                *interrupted = true;
            }
        },
        [](const std::string&, const std::string&) {
            // Keep /review output focused on final findings.
        },
        [append_fn, collected_output, interrupted]() {
            const std::string raw = trim_copy(*collected_output);
            if (*interrupted || raw.find("[Generation stopped by user]") != std::string::npos) {
                append_fn(
                    "\nℹ  Review was interrupted. Re-run /review and wait for it to complete.\n");
                return;
            }

            const ReviewOutputRecord output = parse_review_output_event(raw);
            const std::string rendered = render_review_output_text(output);
            append_fn(std::format("\n{}\n", rendered));
        });
}

} // namespace core::commands
