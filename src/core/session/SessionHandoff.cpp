#include "SessionHandoff.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <optional>
#include <simdjson.h>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::session {

namespace {

struct HandoffSignals {
    std::string original_task;
    std::vector<std::string> recent_user_messages;
    std::string last_assistant;
    std::vector<std::string> recent_tools;
    std::vector<std::string> files_read;
    std::vector<std::string> files_modified;
    std::vector<std::string> recent_commands;
};

[[nodiscard]] std::string trim_copy(std::string_view value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(start, end - start + 1));
}

[[nodiscard]] std::string clamp_line(std::string_view input, std::size_t max_chars = 280) {
    std::string out = trim_copy(input);
    if (out.size() <= max_chars) return out;
    out.resize(max_chars);
    out += "...";
    return out;
}

[[nodiscard]] std::string lower_ascii_copy(std::string_view input) {
    std::string out(input);
    std::ranges::transform(out, out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

void append_unique_limited(std::vector<std::string>& values,
                           std::string value,
                           std::size_t max_items = 6,
                           std::size_t max_chars = 120) {
    value = clamp_line(value, max_chars);
    if (value.empty()) return;
    if (std::ranges::find(values, value) != values.end()) return;
    if (values.size() >= max_items) return;
    values.push_back(std::move(value));
}

void push_recent_limited(std::vector<std::string>& values,
                         std::string value,
                         std::size_t max_items = 3,
                         std::size_t max_chars = 200) {
    value = clamp_line(value, max_chars);
    if (value.empty()) return;
    if (!values.empty() && values.back() == value) return;
    values.push_back(std::move(value));
    if (values.size() > max_items) {
        values.erase(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(values.size() - max_items));
    }
}

[[nodiscard]] std::string basename_or_self(std::string_view path) {
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string_view::npos) return trim_copy(path);
    return trim_copy(path.substr(pos + 1));
}

[[nodiscard]] std::optional<std::string> extract_patch_path(std::string_view line) {
    if (!(line.starts_with("--- ") || line.starts_with("+++ "))) {
        return std::nullopt;
    }

    std::string raw = trim_copy(line.substr(4));
    if (raw.empty()) {
        return std::nullopt;
    }

    const auto tab_pos = raw.find('\t');
    if (tab_pos != std::string::npos) {
        raw.erase(tab_pos);
        raw = trim_copy(raw);
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

void capture_patch_paths(HandoffSignals& signals, std::string_view patch_text) {
    std::istringstream lines{std::string(patch_text)};
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        const auto patch_path = extract_patch_path(line);
        if (!patch_path.has_value()) {
            continue;
        }

        append_unique_limited(signals.files_modified, basename_or_self(*patch_path), 8, 96);
    }
}

[[nodiscard]] std::string join_items(const std::vector<std::string>& values,
                                     std::string_view separator = ", ") {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += separator;
        out += values[i];
    }
    return out;
}

[[nodiscard]] std::string json_string_field(
    std::string_view json,
    std::initializer_list<std::string_view> keys) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc) != simdjson::SUCCESS) {
        return {};
    }

    for (const auto key : keys) {
        std::string_view value;
        if (doc[key.data()].get(value) == simdjson::SUCCESS) {
            return std::string(value);
        }
    }
    return {};
}

[[nodiscard]] int64_t json_int_field(
    std::string_view json,
    std::initializer_list<std::string_view> keys) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc) != simdjson::SUCCESS) {
        return 0;
    }

    for (const auto key : keys) {
        int64_t value = 0;
        if (doc[key.data()].get(value) == simdjson::SUCCESS) {
            return value;
        }
    }
    return 0;
}

[[nodiscard]] std::string summarize_text_blob(std::string_view blob, std::size_t max_chars = 160) {
    const auto trimmed = trim_copy(blob);
    if (trimmed.empty()) return {};

    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start < trimmed.size()) {
        const auto end = trimmed.find('\n', start);
        const auto line = trim_copy(trimmed.substr(start, end == std::string_view::npos ? trimmed.size() - start
                                                                                        : end - start));
        if (!line.empty()) {
            lines.push_back(line);
        }
        if (end == std::string_view::npos) break;
        start = end + 1;
    }

    const std::array keywords = {
        "error", "failed", "warning", "passed", "succeeded", "success", "build", "test", "lint"
    };

    for (const auto& line : lines) {
        const auto lowered = lower_ascii_copy(line);
        if (std::ranges::any_of(keywords, [&](std::string_view keyword) {
                return lowered.find(keyword) != std::string::npos;
            })) {
            return clamp_line(line, max_chars);
        }
    }

    if (!lines.empty()) {
        return clamp_line(lines.front(), max_chars);
    }
    return clamp_line(trimmed, max_chars);
}

void capture_tool_effects(HandoffSignals& signals,
                          std::string_view tool_name,
                          std::string_view tool_args,
                          std::string_view tool_result) {
    const std::string tool = trim_copy(tool_name);
    if (tool.empty()) return;

    append_unique_limited(signals.recent_tools, tool, 8, 64);

    if (tool == "read_file") {
        append_unique_limited(signals.files_read, basename_or_self(json_string_field(tool_args, {"path"})), 6, 96);
    } else if (tool == "apply_patch") {
        capture_patch_paths(signals, json_string_field(tool_args, {"patch"}));
    } else if (tool == "write_file" || tool == "replace" || tool == "search_replace"
               || tool == "delete_file") {
        append_unique_limited(
            signals.files_modified,
            basename_or_self(json_string_field(tool_args, {"file_path", "path"})),
            8,
            96);
    } else if (tool == "move_file") {
        append_unique_limited(
            signals.files_modified,
            basename_or_self(json_string_field(tool_args, {"source", "from"})),
            8,
            96);
        append_unique_limited(
            signals.files_modified,
            basename_or_self(json_string_field(tool_args, {"destination", "to"})),
            8,
            96);
    }

    if (tool == "run_terminal_command") {
        std::string command = json_string_field(tool_args, {"command"});
        command = clamp_line(command, 120);
        if (!command.empty()) {
            std::string summary = command;
            const std::string error = json_string_field(tool_result, {"error"});
            const std::string output = json_string_field(tool_result, {"output", "content"});
            const int64_t exit_code = json_int_field(tool_result, {"exit_code"});

            if (!error.empty()) {
                summary += std::format(" => {}", clamp_line(error, 120));
            } else if (!output.empty()) {
                summary += std::format(" => {}", summarize_text_blob(output, 120));
            } else if (exit_code != 0) {
                summary += std::format(" => exit {}", exit_code);
            }
            push_recent_limited(signals.recent_commands, summary, 3, 220);
        }
    }
}

[[nodiscard]] HandoffSignals extract_signals(const std::vector<core::llm::Message>& messages) {
    HandoffSignals signals;
    std::unordered_map<std::string, core::llm::ToolCall> tool_calls_by_id;

    for (const auto& message : messages) {
        if (message.role == "user" && !message.content.empty()) {
            if (signals.original_task.empty()) {
                signals.original_task = clamp_line(message.content, 240);
            }
            push_recent_limited(signals.recent_user_messages, message.content, 3, 220);
            continue;
        }

        if (message.role == "assistant") {
            if (!message.content.empty()) {
                signals.last_assistant = clamp_line(message.content, 420);
            }
            for (const auto& tool_call : message.tool_calls) {
                append_unique_limited(signals.recent_tools, tool_call.function.name, 8, 64);
                if (!tool_call.id.empty()) {
                    tool_calls_by_id[tool_call.id] = tool_call;
                }
            }
            continue;
        }

        if (message.role == "tool") {
            std::string tool_name = message.name;
            std::string tool_args;
            if (const auto it = tool_calls_by_id.find(message.tool_call_id);
                it != tool_calls_by_id.end()) {
                tool_name = it->second.function.name;
                tool_args = it->second.function.arguments;
            }
            capture_tool_effects(signals, tool_name, tool_args, message.content);
        }
    }

    return signals;
}

} // namespace

bool has_handoff_summary(const SessionData& data) noexcept {
    return !data.handoff_summary.empty();
}

std::string build_handoff_summary(const SessionData& data) {
    if (!data.handoff_summary.empty()) {
        return data.handoff_summary;
    }

    const auto signals = extract_signals(data.messages);
    std::vector<std::string> sentences;
    sentences.push_back(std::format(
        "Continue the previous {} session transparently and efficiently.",
        data.mode.empty() ? "BUILD" : data.mode));
    sentences.push_back(
        "Keep working on the active task without asking the user to restate context unless something is ambiguous.");

    if (const auto summary = trim_copy(data.context_summary); !summary.empty()) {
        sentences.push_back(std::format("Saved summary: {}.", clamp_line(summary, 420)));
    }

    if (!signals.original_task.empty()) {
        sentences.push_back(std::format("Original task: {}.", signals.original_task));
    }
    if (!signals.recent_user_messages.empty()) {
        sentences.push_back(std::format(
            "Recent user directions: {}.",
            join_items(signals.recent_user_messages, " | ")));
    }
    if (!signals.last_assistant.empty()) {
        sentences.push_back(std::format("Latest assistant state: {}.", signals.last_assistant));
    }
    if (!signals.files_modified.empty()) {
        sentences.push_back(std::format(
            "Files changed recently: {}.",
            join_items(signals.files_modified)));
    }
    if (!signals.files_read.empty()) {
        sentences.push_back(std::format(
            "Files inspected recently: {}.",
            join_items(signals.files_read)));
    }
    if (!signals.recent_commands.empty()) {
        sentences.push_back(std::format(
            "Recent commands and outcomes: {}.",
            join_items(signals.recent_commands, " | ")));
    }
    if (!signals.recent_tools.empty()) {
        sentences.push_back(std::format(
            "Recent tools used: {}.",
            join_items(signals.recent_tools)));
    }

    if (!data.provider.empty() || !data.model.empty()) {
        sentences.push_back(std::format(
            "Previous backend: {}{}{}.",
            data.provider.empty() ? std::string("<unknown>") : data.provider,
            data.model.empty() ? std::string() : " / ",
            data.model));
    }

    std::string out;
    for (std::size_t i = 0; i < sentences.size(); ++i) {
        if (i > 0) out.push_back(' ');
        out += sentences[i];
    }
    return clamp_line(out, 1800);
}

std::string build_handoff_summary(
    const std::vector<core::llm::Message>& messages,
    std::string_view existing_context_summary,
    std::string_view mode) {
    std::string summary = trim_copy(existing_context_summary);
    if (!summary.empty()) {
        return summary;
    }

    const auto signals = extract_signals(messages);

    std::string out = std::format(
        "Continue the previous {} session efficiently. Original task: {}",
        mode.empty() ? "BUILD" : std::string(mode),
        signals.original_task.empty() ? "resume the coding task from the saved state." : signals.original_task);

    if (!signals.recent_user_messages.empty()) {
        const std::string& last_user = signals.recent_user_messages.back();
        if (last_user != signals.original_task) {
            out += std::format(" Most recent user direction: {}.", last_user);
        }
    }
    if (!signals.last_assistant.empty()) {
        out += std::format(" Latest assistant state: {}.", signals.last_assistant);
    }
    if (!signals.recent_tools.empty()) {
        out += std::format(" Recent tools used: {}.", join_items(signals.recent_tools));
    }

    return out;
}

} // namespace core::session
