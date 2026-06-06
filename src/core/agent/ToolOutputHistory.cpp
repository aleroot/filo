#include "ToolOutputHistory.hpp"

#include "../tools/ToolNames.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <mutex>
#include <optional>
#include <simdjson.h>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::agent::tool_output_history {

namespace {

[[nodiscard]] bool looks_like_error_payload(std::string_view payload) {
    return payload.contains("\"error\"");
}

[[nodiscard]] uint64_t fnv1a64(std::string_view text) noexcept {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::string digest_hex(std::string_view text) {
    return std::format("{:016x}", fnv1a64(text));
}

constexpr auto kCodeSignalPrefixes = std::to_array<std::string_view>({
    "#include", "#define", "import ", "from ", "use ", "mod ",
    "pub ", "class ", "struct ", "enum ", "interface ", "type ",
    "trait ", "impl ", "def ", "async def ", "fn ", "async fn ",
    "function ", "export ", "const ", "let ", "var ",
});

constexpr auto kCodeSignalNeedles = std::to_array<std::string_view>({
    " class ", " struct ", " enum ", " interface ", " function ",
    " operator", ") const", ") noexcept", ") override", ") ->",
    "=>", "throws ", "extends ", "implements ",
});

constexpr auto kShellSignalNeedles = std::to_array<std::string_view>({
    "error", "failed", "failure", "fatal", "panic", "exception",
    "warning", "warn", "passed", "success", "succeeded", "test",
    "undefined", "not found", "denied", "timeout", "traceback",
});

constexpr auto kGitDiffSummaryFlags = std::to_array<std::string_view>({
    "--stat", "--shortstat", "--numstat", "--name-only", "--name-status", "--summary",
});

constexpr auto kBuildTestCommandPrefixes = std::to_array<std::string_view>({
    "cargo ", "npm ", "pnpm ", "yarn ", "bun ", "pytest",
    "go test", "cmake ", "make", "ninja", "./gradlew",
    "gradle ", "mvn ", "pip ",
});

constexpr auto kBuildDiagnosticNeedles = std::to_array<std::string_view>({
    "error[", "error:", " error ", "warning[", "warning:", " failed",
    "failed ", "failure", "panic", "panicked at", "exception", "traceback",
    "cannot find", "not found", "undefined", "unresolved", "expected ",
    "mismatched types", "aborting due to", "could not compile",
});

constexpr auto kGitStatusSignalPrefixes = std::to_array<std::string_view>({
    "on branch ", "your branch ", "changes ", "modified:", "new file:",
    "deleted:", "renamed:", "untracked files:", "nothing to commit",
    "no changes added", "all conflicts fixed",
});

constexpr auto kDiffSignalPrefixes = std::to_array<std::string_view>({
    "diff --git ", "+++ ", "--- ", "@@ ", "rename from ", "rename to ",
    "new file mode ", "deleted file mode ", "index ",
    "similarity index ", "dissimilarity index ", "old mode ", "new mode ",
    "copy from ", "copy to ", "Binary files ", "GIT binary patch",
    "commit ", "Author:", "Date:",
});

enum class CompressionMode {
    Off,
    Light,
    Full,
    Ultra,
};

enum class ShellSummaryDetail {
    WithGenericPreview,
    SignalsOnly,
};

struct CompressionProfile {
    CompressionMode mode = CompressionMode::Off;
    Limits limits{};
    std::string_view label = "off";
    std::size_t max_shell_signals = 160;
    ShellSummaryDetail shell_summary_detail = ShellSummaryDetail::WithGenericPreview;
};

[[nodiscard]] CompressionMode parse_compression_mode(std::string_view mode) {
    const std::string normalized = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(mode));
    if (normalized == "light"
        || normalized == "on"
        || normalized == "true"
        || normalized == "1") {
        return CompressionMode::Light;
    }
    if (normalized == "full") {
        return CompressionMode::Full;
    }
    if (normalized == "ultra") {
        return CompressionMode::Ultra;
    }
    return CompressionMode::Off;
}

[[nodiscard]] bool uses_full_compression(CompressionMode mode) {
    return mode == CompressionMode::Full || mode == CompressionMode::Ultra;
}

[[nodiscard]] std::string_view compression_label(CompressionMode mode) {
    switch (mode) {
        case CompressionMode::Light:
            return "light";
        case CompressionMode::Full:
            return "full";
        case CompressionMode::Ultra:
            return "ultra";
        case CompressionMode::Off:
            return "off";
    }
    return "off";
}

[[nodiscard]] Limits sanitize(Limits limits) {
    if (limits.max_chars == 0) {
        limits.max_chars = 1;
    }
    if (limits.head_chars + limits.tail_chars == 0) {
        limits.head_chars = limits.max_chars;
        limits.tail_chars = 0;
        return limits;
    }

    if (limits.head_chars + limits.tail_chars > limits.max_chars) {
        const auto total = limits.head_chars + limits.tail_chars;
        const auto maxc = limits.max_chars;
        limits.head_chars = (limits.head_chars * maxc) / total;
        limits.tail_chars = maxc - limits.head_chars;
    }
    return limits;
}

[[nodiscard]] CompressionProfile compression_profile(std::string_view tool_name,
                                                     Limits limits,
                                                     CompressionMode mode) {
    CompressionProfile profile{
        .mode = mode,
        .limits = sanitize(limits),
        .label = compression_label(mode),
    };
    if (mode != CompressionMode::Ultra) {
        return profile;
    }

    if (tool_name == core::tools::names::kReadFile) {
        profile.limits = sanitize(Limits{
            .max_chars = std::min<std::size_t>(profile.limits.max_chars, 4 * 1024),
            .head_chars = std::min<std::size_t>(profile.limits.head_chars, 2200),
            .tail_chars = std::min<std::size_t>(profile.limits.tail_chars, 1200),
        });
        return profile;
    }

    if (tool_name == core::tools::names::kRunTerminalCommand) {
        profile.limits = sanitize(Limits{
            .max_chars = std::min<std::size_t>(profile.limits.max_chars, 5 * 1024),
            .head_chars = std::min<std::size_t>(profile.limits.head_chars, 2600),
            .tail_chars = std::min<std::size_t>(profile.limits.tail_chars, 1400),
        });
        profile.max_shell_signals = 48;
        profile.shell_summary_detail = ShellSummaryDetail::SignalsOnly;
        return profile;
    }

    return profile;
}

[[nodiscard]] std::pair<std::string_view, std::string_view>
split_preview(std::string_view text, Limits limits) {
    limits = sanitize(limits);
    if (text.size() <= limits.max_chars) {
        return {text, {}};
    }

    const std::size_t head = std::min(limits.head_chars, text.size());
    const std::size_t remaining = text.size() - head;
    const std::size_t tail = std::min(limits.tail_chars, remaining);

    std::string_view head_view = text.substr(0, head);
    std::string_view tail_view = tail > 0 ? text.substr(text.size() - tail, tail) : std::string_view{};
    return {head_view, tail_view};
}

[[nodiscard]] std::size_t count_lines(std::string_view text) noexcept {
    if (text.empty()) return 0;
    std::size_t count = 1;
    for (const char ch : text) {
        if (ch == '\n') ++count;
    }
    return count;
}

[[nodiscard]] std::optional<std::string> json_string_field(std::string_view raw_output,
                                                           std::string_view field_name) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(raw_output.data(), raw_output.size()).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    std::string_view value;
    if (doc[field_name.data()].get(value) == simdjson::SUCCESS) {
        return std::string(value);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<int64_t> json_optional_int_field(std::string_view raw_output,
                                                            std::string_view field_name) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(raw_output.data(), raw_output.size()).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    int64_t value = 0;
    if (doc[field_name.data()].get(value) == simdjson::SUCCESS) {
        return value;
    }
    return std::nullopt;
}

[[nodiscard]] int64_t json_int_field(std::string_view raw_output,
                                     std::string_view field_name,
                                     int64_t fallback = 0) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(raw_output.data(), raw_output.size()).get(doc) != simdjson::SUCCESS) {
        return fallback;
    }
    int64_t value = fallback;
    if (doc[field_name.data()].get(value) == simdjson::SUCCESS) {
        return value;
    }
    return fallback;
}

[[nodiscard]] bool is_code_signal_line(std::string_view trimmed) {
    if (trimmed.empty()) return false;
    const std::string lower = core::utils::str::to_lower_ascii_copy(trimmed);

    return std::ranges::any_of(
               kCodeSignalPrefixes,
               [&lower](std::string_view prefix) { return lower.starts_with(prefix); })
        || std::ranges::any_of(
               kCodeSignalNeedles,
               [&lower](std::string_view needle) { return lower.contains(needle); });
}

[[nodiscard]] bool is_instruction_file(std::string_view path) {
    const std::string lower = core::utils::str::to_lower_ascii_copy(path);
    return lower.ends_with("agents.md")
        || lower.ends_with("filo.md")
        || lower.ends_with("skill.md")
        || lower.ends_with("rules.md")
        || lower.ends_with("gemini.md")
        || lower.ends_with("claude.md")
        || lower.ends_with("system.md")
        || lower.ends_with(".cursorrules")
        || lower.contains("/.filo/steering/")
        || lower.contains("/skills/");
}

[[nodiscard]] std::string structural_file_summary(std::string_view content,
                                                  std::size_t max_chars) {
    constexpr std::size_t kMaxSignals = 160;
    constexpr std::size_t kMaxLineChars = 180;
    std::vector<std::pair<std::size_t, std::string>> signals;
    signals.reserve(64);

    std::istringstream lines{std::string(content)};
    std::string line;
    std::size_t line_no = 1;
    while (std::getline(lines, line)) {
        const std::string trimmed = core::utils::str::trim_ascii_copy(line);
        if (is_code_signal_line(trimmed)) {
            std::string kept = trimmed;
            if (kept.size() > kMaxLineChars) {
                kept.resize(kMaxLineChars);
                kept += "...";
            }
            signals.emplace_back(line_no, std::move(kept));
            if (signals.size() >= kMaxSignals) break;
        }
        ++line_no;
    }

    const std::size_t head_chars = std::min<std::size_t>(max_chars / 5, 2400);
    const std::size_t tail_chars = std::min<std::size_t>(max_chars / 5, 1800);
    const auto [head, tail] = split_preview(content, Limits{
        .max_chars = head_chars + tail_chars,
        .head_chars = head_chars,
        .tail_chars = tail_chars,
    });

    std::ostringstream out;
    out << "[light read_file summary]\n";
    out << "Original chars: " << content.size() << "\n";
    out << "Use read_file with offset_line/limit_lines for exact source slices.\n";
    if (!signals.empty()) {
        out << "\nStructural lines:\n";
        const std::size_t signal_budget = max_chars > tail_chars + 512
            ? max_chars - tail_chars - 512
            : max_chars;
        for (const auto& [line_number, text] : signals) {
            out << std::format("{:>5} | {}\n", line_number, text);
            if (out.tellp() > static_cast<std::streampos>(signal_budget)) {
                out << "[structural lines truncated]\n";
                break;
            }
        }
    }
    out << "\nHead:\n" << head;
    if (!tail.empty()) {
        out << "\n\nTail:\n" << tail;
    }

    std::string result = out.str();
    if (result.size() > max_chars) {
        result.resize(max_chars);
        result += "\n[light summary truncated]\n";
    }
    return result;
}

[[nodiscard]] bool is_shell_signal_line(std::string_view trimmed) {
    const std::string lower = core::utils::str::to_lower_ascii_copy(trimmed);
    return std::ranges::any_of(
        kShellSignalNeedles,
        [&lower](std::string_view needle) { return lower.contains(needle); });
}

[[nodiscard]] std::string shell_output_summary(std::string_view output,
                                               int64_t exit_code,
                                               std::size_t max_chars) {
    constexpr std::size_t kMaxSignals = 120;
    constexpr std::size_t kMaxLineChars = 220;
    std::vector<std::string> signals;
    signals.reserve(48);

    std::istringstream lines{std::string(output)};
    std::string line;
    while (std::getline(lines, line)) {
        std::string trimmed = core::utils::str::trim_ascii_copy(line);
        if (!is_shell_signal_line(trimmed)) continue;
        if (trimmed.size() > kMaxLineChars) {
            trimmed.resize(kMaxLineChars);
            trimmed += "...";
        }
        if (std::ranges::find(signals, trimmed) == signals.end()) {
            signals.push_back(std::move(trimmed));
        }
        if (signals.size() >= kMaxSignals) break;
    }

    const std::size_t head_chars = std::min<std::size_t>(max_chars / 4, 3000);
    const std::size_t tail_chars = std::min<std::size_t>(max_chars / 4, 3000);
    const auto [head, tail] = split_preview(output, Limits{
        .max_chars = head_chars + tail_chars,
        .head_chars = head_chars,
        .tail_chars = tail_chars,
    });

    std::ostringstream out;
    out << "[light shell summary]\n";
    out << "Exit code: " << exit_code << "\n";
    out << "Original chars: " << output.size() << "\n";
    if (!signals.empty()) {
        out << "\nSignal lines:\n";
        const std::size_t signal_budget = max_chars > tail_chars + 512
            ? max_chars - tail_chars - 512
            : max_chars;
        for (const auto& signal : signals) {
            out << "- " << signal << "\n";
            if (out.tellp() > static_cast<std::streampos>(signal_budget)) {
                out << "[signal lines truncated]\n";
                break;
            }
        }
    }
    out << "\nHead:\n" << head;
    if (!tail.empty()) {
        out << "\n\nTail:\n" << tail;
    }

    std::string result = out.str();
    if (result.size() > max_chars) {
        result.resize(max_chars);
        result += "\n[light summary truncated]\n";
    }
    return result;
}

[[nodiscard]] std::string oversized_full_file_summary(std::string_view content,
                                                      std::size_t max_chars) {
    std::string summary = structural_file_summary(content, max_chars);
    constexpr std::string_view kLightHeader = "[light read_file summary]";
    if (summary.starts_with(kLightHeader)) {
        summary.replace(0, kLightHeader.size(), "[full read_file summary]");
    }
    return summary;
}

[[nodiscard]] std::string wrapped_light_payload(std::string_view tool_name,
                                                std::string_view field_name,
                                                std::string_view original_payload,
                                                std::string_view compressed_payload) {
    core::utils::JsonWriter writer(512 + compressed_payload.size());
    {
        auto object = writer.object();
        writer.kv_bool("compressed", true).comma()
              .kv_str("compression", "light").comma()
              .kv_str("tool", tool_name).comma()
              .kv_num("original_chars", static_cast<int64_t>(original_payload.size())).comma()
              .kv_num("kept_chars", static_cast<int64_t>(compressed_payload.size())).comma()
              .kv_str("digest_fnv1a64", digest_hex(original_payload)).comma()
              .kv_str(field_name, compressed_payload);
    }
    return std::move(writer).take();
}

struct FileCacheEntry {
    std::string digest;
    std::size_t line_count = 0;
    std::size_t original_chars = 0;
    std::size_t read_count = 0;
};

class FullCompressionCache {
public:
    struct TouchResult {
        std::size_t read_count = 0;
        bool unchanged = false;
    };

    TouchResult touch(std::string_view session_id,
                      std::string_view path,
                      std::string_view digest,
                      std::size_t line_count,
                      std::size_t original_chars) {
        const std::string session = session_id.empty() ? "default" : std::string(session_id);
        const std::string path_key = path.empty()
            ? std::string("content:") + std::string(digest)
            : std::string(path);
        const std::string key = session + '\n' + path_key;

        std::lock_guard lock(mutex_);
        auto it = files_.find(key);
        if (it != files_.end()) {
            FileCacheEntry& entry = it->second;
            const bool unchanged = entry.digest == digest;
            ++entry.read_count;
            entry.digest = std::string(digest);
            entry.line_count = line_count;
            entry.original_chars = original_chars;
            return TouchResult{
                .read_count = entry.read_count,
                .unchanged = unchanged,
            };
        }

        if (files_.size() >= kMaxEntries) {
            files_.erase(files_.begin());
        }

        FileCacheEntry entry;
        entry.digest = std::string(digest);
        entry.line_count = line_count;
        entry.original_chars = original_chars;
        entry.read_count = 1;

        TouchResult result{
            .read_count = entry.read_count,
            .unchanged = false,
        };
        files_[key] = std::move(entry);
        return result;
    }

private:
    static constexpr std::size_t kMaxEntries = 512;
    std::mutex mutex_;
    std::unordered_map<std::string, FileCacheEntry> files_;
};

FullCompressionCache& full_cache() {
    static FullCompressionCache cache;
    return cache;
}

[[nodiscard]] bool is_line_range_request(std::string_view tool_arguments) {
    const auto offset_line = json_optional_int_field(tool_arguments, "offset_line");
    const auto limit_lines = json_optional_int_field(tool_arguments, "limit_lines");
    return (offset_line.has_value() && *offset_line > 1)
        || (limit_lines.has_value() && *limit_lines > 0);
}

[[nodiscard]] std::string compact_path_label(std::string_view path) {
    if (path.empty()) return "<content>";
    constexpr std::size_t kMaxPathChars = 120;
    if (path.size() <= kMaxPathChars) return std::string(path);
    return "..." + std::string(path.substr(path.size() - kMaxPathChars + 3));
}

[[nodiscard]] std::string wrapped_full_payload(std::string_view tool_name,
                                               std::string_view field_name,
                                               std::string_view path_or_command,
                                               std::string_view original_payload,
                                               std::string_view compressed_payload,
                                               std::string_view compression) {
    core::utils::JsonWriter writer(768 + compressed_payload.size());
    {
        auto object = writer.object();
        writer.kv_bool("compressed", true).comma()
              .kv_str("compression", compression).comma()
              .kv_str("tool", tool_name).comma()
              .kv_num("original_chars", static_cast<int64_t>(original_payload.size())).comma()
              .kv_num("kept_chars", static_cast<int64_t>(compressed_payload.size())).comma()
              .kv_str("digest_fnv1a64", digest_hex(original_payload));
        if (!path_or_command.empty()) {
            writer.comma().kv_str(
                tool_name == core::tools::names::kRunTerminalCommand ? "command" : "path",
                path_or_command);
        }
        writer.comma().kv_str(field_name, compressed_payload);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string cached_file_stub(std::string_view path,
                                           std::size_t line_count,
                                           std::size_t original_chars,
                                           std::size_t read_count,
                                           std::string_view digest) {
    return std::format(
        "[cached read] {} unchanged ({} lines, {} chars, read {}x, digest {}). "
        "Use read_file with offset_line/limit_lines for exact source slices.",
        compact_path_label(path),
        line_count,
        original_chars,
        read_count,
        digest);
}

[[nodiscard]] std::optional<std::string> full_compress_read_file(
    std::string_view raw_output,
    Context context,
    const CompressionProfile& profile) {
    const auto content = json_string_field(raw_output, "content");
    if (!content.has_value()) {
        return std::nullopt;
    }

    const auto path = json_string_field(context.tool_arguments, "path").value_or("");
    if (is_instruction_file(path)) {
        return std::nullopt;
    }

    const std::string digest = digest_hex(*content);
    const std::size_t line_count = count_lines(*content);
    const auto touch = full_cache().touch(
        context.session_id,
        path,
        digest,
        line_count,
        content->size());

    if (touch.unchanged && !is_line_range_request(context.tool_arguments)) {
        const std::string stub = cached_file_stub(
            path,
            line_count,
            content->size(),
            touch.read_count,
            digest);
        return wrapped_full_payload(
            core::tools::names::kReadFile,
            "content",
            path,
            *content,
            stub,
            profile.label);
    }

    if (raw_output.size() > profile.limits.max_chars) {
        const std::string summary =
            oversized_full_file_summary(*content, profile.limits.max_chars);
        if (summary.size() < raw_output.size()) {
            return wrapped_full_payload(
                core::tools::names::kReadFile,
                "content",
                path,
                *content,
                summary,
                profile.label);
        }
    }

    return std::nullopt;
}

[[nodiscard]] std::string command_family(std::string_view command) {
    const std::string lower = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(command));
    if (lower.starts_with("git status")) return "git-status";
    if (lower.starts_with("git diff") || lower.starts_with("git show")) {
        if (std::ranges::any_of(
                kGitDiffSummaryFlags,
                [&lower](std::string_view flag) { return lower.contains(flag); })) {
            return "git-diff-summary";
        }
        return "git-diff";
    }
    if (lower.starts_with("git log")) return "git-log";
    if (std::ranges::any_of(
            kBuildTestCommandPrefixes,
            [&lower](std::string_view prefix) { return lower.starts_with(prefix); })) {
        return "build-test";
    }
    return "generic";
}

[[nodiscard]] bool is_build_test_diagnostic_output(std::string_view output,
                                                   int64_t exit_code) {
    if (exit_code != 0) return true;

    const std::string lower = core::utils::str::to_lower_ascii_copy(output);
    return std::ranges::any_of(
        kBuildDiagnosticNeedles,
        [&lower](std::string_view needle) { return lower.contains(needle); });
}

[[nodiscard]] bool is_porcelain_status_line(std::string_view line) {
    if (line.size() < 2) return false;
    if (line.starts_with("##")) return true;
    if (line.starts_with("??") || line.starts_with("!!")) return true;
    constexpr std::string_view kStatusChars = " MADRCU?!";
    const bool has_status =
        kStatusChars.find(line[0]) != std::string_view::npos
        && kStatusChars.find(line[1]) != std::string_view::npos
        && (line[0] != ' ' || line[1] != ' ');
    if (!has_status) return false;
    return line.size() == 2
        || line[2] == ' '
        || line[2] == '\t'
        || line[2] == '"';
}

[[nodiscard]] bool is_git_status_signal(std::string_view line) {
    if (is_porcelain_status_line(line)) return true;
    const std::string trimmed = core::utils::str::trim_ascii_copy(line);
    const std::string lower = core::utils::str::to_lower_ascii_copy(trimmed);
    return std::ranges::any_of(
               kGitStatusSignalPrefixes,
               [&lower](std::string_view prefix) { return lower.starts_with(prefix); })
        || is_porcelain_status_line(trimmed);
}

[[nodiscard]] bool is_diff_signal(std::string_view line) {
    return std::ranges::any_of(
               kDiffSignalPrefixes,
               [line](std::string_view prefix) { return line.starts_with(prefix); })
        || line.starts_with('+')
        || line.starts_with('-');
}

[[nodiscard]] std::string diff_shell_summary(std::string_view output,
                                             std::string_view command,
                                             int64_t exit_code,
                                             std::size_t max_chars) {
    std::ostringstream out;
    out << "[full shell context pack]\n";
    out << "Command family: git-diff\n";
    out << "Exit code: " << exit_code << "\n";
    out << "Original chars: " << output.size() << "\n\n";
    out << "Structural diff lines:\n";

    bool truncated = false;
    const std::size_t budget = max_chars > 128 ? max_chars - 128 : max_chars;
    std::istringstream lines{std::string(output)};
    std::string line;
    while (std::getline(lines, line)) {
        if (!is_diff_signal(line)) continue;
        const std::streamoff next_size =
            static_cast<std::streamoff>(out.tellp())
            + static_cast<std::streamoff>(line.size())
            + 1;
        if (next_size > static_cast<std::streamoff>(budget)) {
            truncated = true;
            break;
        }
        out << line << '\n';
    }

    if (truncated) {
        out << "[structural diff lines truncated; rerun "
            << (command.empty() ? std::string("the diff command") : std::string(command))
            << " for exact output]\n";
    }

    std::string result = out.str();
    if (result.size() > max_chars) {
        result.resize(max_chars);
        result += "\n[full shell context pack truncated]\n";
    }
    return result;
}

[[nodiscard]] std::string pattern_shell_summary(std::string_view output,
                                                std::string_view command,
                                                int64_t exit_code,
                                                const CompressionProfile& profile) {
    const std::string family = command_family(command);
    if (family == "git-diff") {
        return diff_shell_summary(
            output,
            command,
            exit_code,
            profile.limits.max_chars);
    }

    std::vector<std::string> signals;
    signals.reserve(96);

    std::istringstream lines{std::string(output)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string trimmed = core::utils::str::trim_ascii_copy(line);
        if (trimmed.empty()) continue;

        bool keep = false;
        if (family == "git-status") {
            keep = is_git_status_signal(line);
        } else if (family == "git-diff-summary") {
            keep = true;
        } else if (family == "git-log") {
            keep = !trimmed.empty();
        } else {
            keep = is_shell_signal_line(trimmed);
        }

        if (!keep) continue;
        constexpr std::size_t kMaxLineChars = 240;
        const bool keep_verbatim =
            family == "git-status" || family == "git-diff-summary";
        const bool preserve_duplicates = family == "git-diff-summary";
        std::string kept = keep_verbatim ? line : trimmed;
        if (kept.size() > kMaxLineChars) {
            kept.resize(kMaxLineChars);
            kept += "...";
        }
        if (preserve_duplicates || std::ranges::find(signals, kept) == signals.end()) {
            signals.push_back(std::move(kept));
        }
        if (signals.size() >= profile.max_shell_signals) break;
    }

    if (signals.empty()) {
        return shell_output_summary(output, exit_code, profile.limits.max_chars);
    }

    const auto [head, tail] = split_preview(output, Limits{
        .max_chars = std::min<std::size_t>(profile.limits.max_chars / 2, 4800),
        .head_chars = std::min<std::size_t>(profile.limits.max_chars / 4, 2400),
        .tail_chars = std::min<std::size_t>(profile.limits.max_chars / 4, 2400),
    });

    std::ostringstream out;
    out << "[full shell context pack]\n";
    out << "Command family: " << family << "\n";
    out << "Exit code: " << exit_code << "\n";
    out << "Original chars: " << output.size() << "\n\n";
    out << "Signal lines:\n";
    const std::size_t signal_budget = profile.limits.max_chars > 1024
        ? profile.limits.max_chars - 1024
        : profile.limits.max_chars;
    for (const auto& signal : signals) {
        out << "- " << signal << "\n";
        if (out.tellp() > static_cast<std::streampos>(signal_budget)) {
            out << "[signal lines truncated]\n";
            break;
        }
    }
    if (family == "generic"
        && profile.shell_summary_detail == ShellSummaryDetail::WithGenericPreview) {
        out << "\nHead:\n" << head;
        if (!tail.empty()) out << "\n\nTail:\n" << tail;
    }

    std::string result = out.str();
    if (result.size() > profile.limits.max_chars) {
        result.resize(profile.limits.max_chars);
        result += "\n[full shell context pack truncated]\n";
    }
    return result;
}

[[nodiscard]] std::optional<std::string> full_compress_shell_output(
    std::string_view raw_output,
    Context context,
    const CompressionProfile& profile) {
    const auto output = json_string_field(raw_output, "output");
    if (!output.has_value()) {
        return std::nullopt;
    }
    const auto command = json_string_field(context.tool_arguments, "command").value_or("");
    const int64_t exit_code = json_int_field(raw_output, "exit_code");
    const std::string family = command_family(command);

    if (family == "build-test" && is_build_test_diagnostic_output(*output, exit_code)) {
        return std::nullopt;
    }

    const bool known_family = family != "generic";
    if (!known_family && output->size() <= profile.limits.max_chars) {
        return std::nullopt;
    }

    const std::string summary =
        pattern_shell_summary(*output, command, exit_code, profile);
    if (summary.size() >= raw_output.size()) {
        return std::nullopt;
    }
    return wrapped_full_payload(
        core::tools::names::kRunTerminalCommand,
        "output",
        command,
        *output,
        summary,
        profile.label);
}

[[nodiscard]] std::optional<std::string> light_compress_tool_output(
    std::string_view tool_name,
    std::string_view raw_output,
    Limits limits) {
    if (tool_name == core::tools::names::kReadFile) {
        const auto content = json_string_field(raw_output, "content");
        if (!content.has_value() || content->size() <= limits.max_chars) {
            return std::nullopt;
        }
        const std::string compressed = structural_file_summary(*content, limits.max_chars);
        if (compressed.size() >= raw_output.size()) {
            return std::nullopt;
        }
        return wrapped_light_payload(tool_name, "content", *content, compressed);
    }

    if (tool_name == core::tools::names::kRunTerminalCommand) {
        const auto output = json_string_field(raw_output, "output");
        if (!output.has_value() || output->size() <= limits.max_chars) {
            return std::nullopt;
        }
        const int64_t exit_code = json_int_field(raw_output, "exit_code");
        const std::string compressed =
            shell_output_summary(*output, exit_code, limits.max_chars);
        if (compressed.size() >= raw_output.size()) {
            return std::nullopt;
        }
        return wrapped_light_payload(tool_name, "output", *output, compressed);
    }

    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> full_compress_tool_output(
    std::string_view tool_name,
    std::string_view raw_output,
    Context context,
    const CompressionProfile& profile) {
    if (tool_name == core::tools::names::kReadFile) {
        return full_compress_read_file(raw_output, context, profile);
    }
    if (tool_name == core::tools::names::kRunTerminalCommand) {
        return full_compress_shell_output(raw_output, context, profile);
    }
    return std::nullopt;
}

} // namespace

Limits limits_for_tool(std::string_view tool_name) {
    if (tool_name == core::tools::names::kReadFile) {
        return Limits{.max_chars = 12 * 1024, .head_chars = 8 * 1024, .tail_chars = 4 * 1024};
    }
    if (tool_name == core::tools::names::kRunTerminalCommand
        || tool_name == core::tools::names::kGrepSearch
        || tool_name == core::tools::names::kFileSearch
        || tool_name == core::tools::names::kListDirectory) {
        return Limits{.max_chars = 10 * 1024, .head_chars = 7 * 1024, .tail_chars = 3 * 1024};
    }
    if (tool_name == core::tools::names::kActivateSkill) {
        return Limits{
            .max_chars = 2 * 1024 * 1024,
            .head_chars = 2 * 1024 * 1024,
            .tail_chars = 0,
        };
    }
    return Limits{};
}

std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output) {
    return clamp_for_history(tool_name, raw_output, limits_for_tool(tool_name));
}

std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    std::string_view compression_mode) {
    return clamp_for_history(
        tool_name,
        raw_output,
        limits_for_tool(tool_name),
        compression_mode);
}

std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    std::string_view compression_mode,
    Context context) {
    return clamp_for_history(
        tool_name,
        raw_output,
        limits_for_tool(tool_name),
        compression_mode,
        context);
}

std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    Limits limits) {
    return clamp_for_history(tool_name, raw_output, limits, "off");
}

std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    Limits limits,
    std::string_view compression_mode) {
    return clamp_for_history(tool_name, raw_output, limits, compression_mode, {});
}

std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    Limits limits,
    std::string_view compression_mode,
    Context context) {
    const CompressionProfile profile =
        compression_profile(tool_name, limits, parse_compression_mode(compression_mode));
    if (tool_name == core::tools::names::kReadFile) {
        const auto path = json_string_field(context.tool_arguments, "path").value_or("");
        if (is_instruction_file(path)) {
            return std::string(raw_output);
        }
    }

    if (raw_output.size() <= profile.limits.max_chars) {
        if (uses_full_compression(profile.mode)) {
            if (auto compressed = full_compress_tool_output(
                    tool_name,
                    raw_output,
                    context,
                    profile);
                compressed.has_value() && compressed->size() < raw_output.size()) {
                return *compressed;
            }
        }
        return std::string(raw_output);
    }

    if (uses_full_compression(profile.mode)) {
        if (auto compressed = full_compress_tool_output(
                tool_name,
                raw_output,
                context,
                profile);
            compressed.has_value() && compressed->size() < raw_output.size()) {
            return *compressed;
        }
    }

    if (profile.mode == CompressionMode::Light) {
        if (auto compressed = light_compress_tool_output(tool_name, raw_output, profile.limits);
            compressed.has_value() && compressed->size() < raw_output.size()) {
            return *compressed;
        }
    }

    const bool is_error = looks_like_error_payload(raw_output);
    const auto [head, tail] = split_preview(raw_output, profile.limits);

    core::utils::JsonWriter writer(256 + head.size() + tail.size());
    {
        auto object = writer.object();
        if (is_error) {
            writer.kv_str("error", "Tool output truncated for history (oversized error payload).");
        } else {
            writer.kv_bool("truncated", true);
        }
        writer.comma();
        writer.kv_str("tool", tool_name);
        writer.comma();
        writer.kv_num("original_chars", static_cast<int64_t>(raw_output.size()));
        writer.comma();
        writer.kv_num("kept_chars", static_cast<int64_t>(head.size() + tail.size()));
        writer.comma();
        writer.kv_str("digest_fnv1a64", digest_hex(raw_output));
        writer.comma();
        writer.kv_str("head", head);
        if (!tail.empty()) {
            writer.comma();
            writer.kv_str("tail", tail);
        }
    }
    return std::move(writer).take();
}

} // namespace core::agent::tool_output_history
