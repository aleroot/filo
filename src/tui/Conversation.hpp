#pragma once

#include "DiffPreview.hpp"
#include "Constants.hpp"
#include <ftxui/dom/elements.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tui {

// ============================================================================
// Tool Activity (Individual Tool Call)
// ============================================================================

struct ToolActivity {
    enum class Status { 
        Pending,      // Waiting to execute
        Executing,    // Currently running
        Succeeded,    // Completed successfully
        Failed,       // Error occurred
        Denied,       // User denied permission
        Cancelled     // Cancelled by user/system
    };

    struct Result {
        std::string summary;                // Tool output/result
        std::optional<int> exit_code;       // Terminal command exit status (when available)
        bool truncated = false;             // True when output was truncated upstream

        [[nodiscard]] bool empty() const noexcept {
            return summary.empty();
        }

        void clear() {
            summary.clear();
            exit_code.reset();
            truncated = false;
        }
    };

    std::string id;                      // Unique tool call ID
    std::string name;                    // Tool name (e.g., "read_file")
    std::string args;                    // JSON arguments
    std::string description;             // Human-readable summary
    Result result;                       // Structured tool output metadata
    ToolDiffPreview diff_preview;        // Diff preview for file operations
    bool auto_approved = false;          // True when bypassed via YOLO/allow-list
    Status status = Status::Pending;
    
    // Progress tracking for long-running tools
    std::optional<int> progress;
    std::optional<int> progress_total;
    std::string progress_message;
};

// ============================================================================
// Message Types (Gemini CLI Parity)
// ============================================================================

enum class MessageType {
    User,           // User message (prompt)
    Assistant,      // Assistant text response
    Info,           // ℹ Informational status message
    Warning,        // ⚠ Warning message
    Error,          // ✗ Error message
    ToolGroup,      // Group of tool calls (bordered container)
    System,         // Generic system message
};

// ============================================================================
// Unified Message Structure
// ============================================================================

struct UiMessage {
    // Core identity
    MessageType type = MessageType::System;
    std::string id;                      // Unique identifier
    
    // Content (varies by type)
    std::string text;                    // Primary text content
    std::string secondary_text;          // Secondary/subtitle text
    std::string icon;                    // Custom icon override
    
    // Visual styling
    std::optional<ftxui::Color> custom_color;
    int margin_top = 0;
    int margin_bottom = 0;
    
    // Assistant-specific fields
    std::vector<ToolActivity> tools;     // Tool calls in this turn
    bool pending = false;                // Still streaming/receiving
    bool thinking = false;               // Show thinking indicator
    bool show_lightbulb = false;         // Show 💡 prefix
    
    // ToolGroup-specific
    bool tool_group_border_top = true;
    bool tool_group_border_bottom = true;
    
    // Metadata
    std::string timestamp;
    
    // Helper methods
    bool is_system_feedback() const {
        return type == MessageType::Info || 
               type == MessageType::Warning || 
               type == MessageType::Error;
    }
    
    bool has_tools() const {
        return !tools.empty();
    }
};

// ============================================================================
// Conversation State (History Separation - Gemini CLI Style)
// ============================================================================

struct ConversationState {
    // Completed conversation history (immutable past)
    std::vector<UiMessage> history;
    
    // Current turn (being built up)
    std::vector<UiMessage> pending;
    
    // Helper to get all messages for rendering
    std::vector<UiMessage> all_messages() const {
        std::vector<UiMessage> result;
        result.reserve(history.size() + pending.size());
        result.insert(result.end(), history.begin(), history.end());
        result.insert(result.end(), pending.begin(), pending.end());
        return result;
    }
    
    // Commit pending items to history (called when turn completes)
    void commit_pending() {
        history.insert(history.end(), pending.begin(), pending.end());
        pending.clear();
    }
    
    // Clear all messages
    void clear() {
        history.clear();
        pending.clear();
    }
    
    // Check if there's an active assistant turn
    bool has_active_turn() const {
        for (const auto& msg : pending) {
            if (msg.type == MessageType::Assistant && msg.pending) {
                return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Render Options
// ============================================================================

struct ConversationRenderOptions {
    bool        show_timestamps = true;
    bool        show_spinner = true;
    bool        expand_tool_results = false;
    std::size_t tool_result_preview_max_lines = kToolResultPreviewMaxLines;
    float       scroll_pos = 1.0f;  // 0.0 = top, 1.0 = bottom
};

// ============================================================================
// Factory Functions
// ============================================================================

// User message
UiMessage make_user_message(std::string text, std::string timestamp = {});

// Assistant message
UiMessage make_assistant_message(std::string text = {}, 
                                  std::string timestamp = {},
                                  bool pending = false);

// Status messages (Gemini CLI style)
UiMessage make_info_message(std::string text, 
                            std::optional<std::string> secondary_text = std::nullopt);
UiMessage make_warning_message(std::string text);
UiMessage make_error_message(std::string text);

// Tool group message
UiMessage make_tool_group_message(std::vector<ToolActivity> tools,
                                   bool border_top = true,
                                   bool border_bottom = true);

// System message
UiMessage make_system_message(std::string text);

// Tool activity factory
ToolActivity make_tool_activity(std::string id,
                                 std::string name,
                                 std::string args,
                                 std::string description = {});

// ============================================================================
// Utility Functions
// ============================================================================

auto current_time_str() -> std::string;

std::string summarize_tool_arguments(std::string_view tool_name, std::string_view tool_args);
std::string format_tool_description(std::string_view tool_name, std::string_view tool_args);
void apply_tool_result(ToolActivity& tool, std::string_view result_payload);

// Tool result summary (for backward compat in tests)
struct ToolResultSummary {
    enum class State { Succeeded, Failed, Denied };
    State state = State::Succeeded;
    std::string preview;
};

ToolResultSummary summarize_tool_result(std::string_view result);

// Tool status helpers
ftxui::Color tool_status_color(ToolActivity::Status status);
std::string_view tool_status_icon(ToolActivity::Status status);
std::string_view tool_status_label(ToolActivity::Status status);
std::string_view tool_status_spinner(std::size_t tick);

// Allow-list helpers
std::string make_allow_key(std::string_view tool_name, std::string_view tool_args);
std::string make_allow_label(std::string_view tool_name, std::string_view tool_args);

// Tool lookup
ToolActivity* find_tool_activity(UiMessage& message, std::string_view tool_id);
const ToolActivity* find_tool_activity(const UiMessage& message, std::string_view tool_id);

// Animation frames
std::string_view thinking_pulse_frame(std::size_t tick);
std::string_view spinner_frame(std::size_t tick);

// ============================================================================
// Rendering Functions
// ============================================================================

// Main render function
ftxui::Element render_history_panel(const std::vector<UiMessage>& messages,
                                    std::size_t tick,
                                    ConversationRenderOptions options = {});

// Render specific message types
ftxui::Element render_user_message(const UiMessage& msg, 
                                   const ConversationRenderOptions& options);
ftxui::Element render_assistant_message(const UiMessage& msg,
                                        std::size_t tick,
                                        const ConversationRenderOptions& options);
ftxui::Element render_info_message(const UiMessage& msg);
ftxui::Element render_warning_message(const UiMessage& msg);
ftxui::Element render_error_message(const UiMessage& msg);
ftxui::Element render_tool_group(const UiMessage& msg,
                                 std::size_t tick,
                                 const ConversationRenderOptions& options);
ftxui::Element render_system_message(const UiMessage& msg);

} // namespace tui
