#include "ToolCallDeduplicator.hpp"

#include "../tools/ToolNames.hpp"

#include <algorithm>
#include <format>

namespace core::agent {

namespace {

constexpr std::size_t kReminder1Start = 3;
constexpr std::size_t kReminder2Start = 5;
constexpr std::size_t kReminder3Start = 8;
constexpr std::size_t kForceStopStart = 12;

[[nodiscard]] std::string reminder1() {
    return "\n\n<system-reminder>\n"
           "You are repeating the exact same tool call with identical parameters. "
           "Carefully inspect the previous result and choose different parameters, "
           "a different method, or finish if enough evidence has been gathered.\n"
           "</system-reminder>";
}

[[nodiscard]] std::string reminder2(std::string_view key, std::size_t count) {
    return std::format(
        "\n\n<system-reminder>\n"
        "Repeated identical tool call detected.\n"
        "- repeated_times: {}\n"
        "- key: {}\n"
        "Do not call this exact tool with these exact arguments again unless the "
        "environment changed. Choose a different next action.\n"
        "</system-reminder>",
        count,
        key);
}

[[nodiscard]] std::string reminder3() {
    return "\n\n<system-reminder>\n"
           "You are stuck repeating the same tool call. Stop calling tools in the "
           "next response. Summarize what has been tried, what the latest result "
           "shows, and what different action or user input is needed.\n"
           "</system-reminder>";
}

} // namespace

void ToolCallDeduplicator::begin_step() {
    std::lock_guard lock(mutex_);
    step_entries_.clear();
    first_index_by_key_.clear();
}

ToolDedupDecision ToolCallDeduplicator::register_call(const core::llm::ToolCall& call) {
    std::lock_guard lock(mutex_);
    const std::size_t index = step_entries_.size();
    const std::string key = key_for_call(call);
    if (key.empty()) {
        step_entries_.push_back({.key = {}, .original_index = index});
        return {.duplicate_in_step = false, .original_index = index};
    }

    if (const auto it = first_index_by_key_.find(key); it != first_index_by_key_.end()) {
        step_entries_.push_back({
            .key = key,
            .duplicate = true,
            .original_index = it->second,
        });
        return {.duplicate_in_step = true, .original_index = it->second};
    }

    first_index_by_key_.emplace(key, index);
    step_entries_.push_back({.key = key, .original_index = index});
    return {.duplicate_in_step = false, .original_index = index};
}

void ToolCallDeduplicator::complete_original(std::size_t index, std::string result) {
    std::lock_guard lock(mutex_);
    if (index >= step_entries_.size()) {
        return;
    }
    step_entries_[index].result = std::move(result);
}

std::optional<std::string> ToolCallDeduplicator::duplicate_result(std::size_t original_index) const {
    std::lock_guard lock(mutex_);
    if (original_index >= step_entries_.size()) {
        return std::nullopt;
    }
    return step_entries_[original_index].result;
}

ToolDedupFinalization ToolCallDeduplicator::finalize_for_model(std::size_t index,
                                                               std::string result) {
    std::lock_guard lock(mutex_);
    if (index >= step_entries_.size()) {
        return {.result = std::move(result)};
    }
    const auto& entry = step_entries_[index];
    if (entry.key.empty() || entry.duplicate) {
        return {.result = std::move(result)};
    }

    std::string last_key = consecutive_key_;
    std::size_t streak = consecutive_count_;
    for (std::size_t i = 0; i <= index && i < step_entries_.size(); ++i) {
        const auto& key = step_entries_[i].key;
        if (key.empty() || step_entries_[i].duplicate) {
            continue;
        }
        if (key == last_key) {
            ++streak;
        } else {
            last_key = key;
            streak = 1;
        }
    }

    bool stop_turn = false;
    if (streak >= kForceStopStart) {
        result += reminder3();
        stop_turn = true;
    } else if (streak >= kReminder3Start) {
        result += reminder3();
    } else if (streak >= kReminder2Start) {
        result += reminder2(entry.key, streak);
    } else if (streak >= kReminder1Start) {
        result += reminder1();
    }

    return {.result = std::move(result), .stop_turn = stop_turn};
}

void ToolCallDeduplicator::end_step() {
    std::lock_guard lock(mutex_);
    for (const auto& entry : step_entries_) {
        if (entry.key.empty() || entry.duplicate) {
            continue;
        }
        if (entry.key == consecutive_key_) {
            ++consecutive_count_;
        } else {
            consecutive_key_ = entry.key;
            consecutive_count_ = 1;
        }
    }
}

std::string ToolCallDeduplicator::key_for_call(const core::llm::ToolCall& call) {
    if (!should_track_tool(call.function.name)) {
        return {};
    }
    return call.function.name + " " + canonicalize_arguments(call.function.arguments);
}

std::string ToolCallDeduplicator::canonicalize_arguments(std::string_view arguments) {
    std::string out;
    out.reserve(arguments.size());
    bool in_string = false;
    bool escaped = false;

    for (const char ch : arguments) {
        if (in_string) {
            out.push_back(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
            out.push_back(ch);
            continue;
        }
        if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') {
            continue;
        }
        out.push_back(ch);
    }
    return out;
}

bool ToolCallDeduplicator::should_track_tool(std::string_view tool_name) noexcept {
    using namespace core::tools::names;
    return tool_name == kReadFile
        || tool_name == kFileSearch
        || tool_name == kGrepSearch
        || tool_name == kListDirectory
        || tool_name == kGetWorkspaceConfig;
}

} // namespace core::agent
