#pragma once

#include "../llm/Models.hpp"

#include <optional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::agent {

struct ToolDedupDecision {
    bool duplicate_in_step = false;
    std::size_t original_index = 0;
};

struct ToolDedupFinalization {
    std::string result;
    bool stop_turn = false;
};

class ToolCallDeduplicator {
public:
    void begin_step();
    [[nodiscard]] ToolDedupDecision register_call(const core::llm::ToolCall& call);
    void complete_original(std::size_t index, std::string result);
    [[nodiscard]] std::optional<std::string> duplicate_result(std::size_t original_index) const;
    [[nodiscard]] ToolDedupFinalization finalize_for_model(std::size_t index, std::string result);
    void end_step();

    [[nodiscard]] static std::string key_for_call(const core::llm::ToolCall& call);
    [[nodiscard]] static std::string canonicalize_arguments(std::string_view arguments);
    [[nodiscard]] static bool should_track_tool(std::string_view tool_name) noexcept;

private:
    struct StepEntry {
        std::string key;
        bool duplicate = false;
        std::size_t original_index = 0;
        std::optional<std::string> result;
    };

    std::vector<StepEntry> step_entries_;
    std::unordered_map<std::string, std::size_t> first_index_by_key_;
    std::string consecutive_key_;
    std::size_t consecutive_count_ = 0;
    mutable std::mutex mutex_;
};

} // namespace core::agent
