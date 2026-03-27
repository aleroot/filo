#pragma once

#include "Autocomplete.hpp"
#include "DiffPreview.hpp"
#include "core/session/SessionStore.hpp"
#include "core/tools/AskUserQuestionTool.hpp"
#include <ftxui/dom/elements.hpp>
#include <filesystem>
#include <string_view>
#include <vector>
#include <utility>
#include <future>
#include <memory>

namespace tui {

struct SettingsPanelRow {
    std::string label;
    std::string value;
    std::string description;
    bool inherited = false;
};

struct LocalModelEntry {
    std::string name;              // display name: "dirname/" or "model.gguf"
    std::filesystem::path path;    // full absolute path
    bool is_directory;
};

struct ConversationSearchHit {
    int message_index = -1;
    std::string role;
    std::string snippet;
};

ftxui::Element render_default_prompt_panel(ftxui::Element input_line,
                                           std::string_view input_text);

ftxui::Element render_command_prompt_panel(const std::vector<CommandSuggestion>& suggestions,
                                           int selected_index,
                                           ftxui::Element input_line,
                                           std::string_view input_text);

ftxui::Element render_mention_prompt_panel(const std::vector<MentionSuggestion>& suggestions,
                                           int selected_index,
                                           ftxui::Element input_line,
                                           std::string_view input_text);

ftxui::Element render_permission_prompt_panel(std::string_view tool_name,
                                              std::string_view args_preview,
                                              const ToolDiffPreview& diff_preview,
                                              std::string_view allow_label,
                                              int selected_index);

ftxui::Element render_startup_banner_panel(std::string_view provider_name,
                                           std::string_view model_name,
                                           int mcp_server_count,
                                           std::string_view provider_setup_hint);

ftxui::Element render_model_selection_panel(int selected_index,
                                            std::string_view manual_description,
                                            std::string_view router_description,
                                            std::string_view auto_description,
                                            std::string_view router_policy,
                                            bool router_available,
                                            bool local_model_available);

ftxui::Element render_provider_selection_panel(const std::vector<std::string>& providers,
                                               int selected_index);

ftxui::Element render_settings_panel(std::string_view scope_label,
                                     std::string_view scope_path,
                                     const std::vector<SettingsPanelRow>& rows,
                                     int selected_index,
                                     std::string_view status_message);

ftxui::Element render_local_model_picker_panel(std::string_view current_dir,
                                               const std::vector<LocalModelEntry>& entries,
                                               int selected_index);

ftxui::Element render_session_picker_panel(const std::vector<core::session::SessionInfo>& sessions,
                                           int selected_index);

ftxui::Element render_conversation_search_panel(
    std::string_view query,
    const std::vector<ConversationSearchHit>& hits,
    int selected_index);

// ---------------------------------------------------------------------------
// Question Dialog (AskUserQuestion tool)
// ---------------------------------------------------------------------------

struct QuestionDialogOption {
    std::string label;
    std::string description;
};

struct QuestionDialogItem {
    std::string question;
    std::string header;
    std::vector<QuestionDialogOption> options;
    bool multi_select = false;
    std::string body;
};

struct QuestionDialogState {
    bool active = false;
    std::vector<QuestionDialogItem> questions;
    int current_question_index = 0;
    int selected_option = 0;
    std::vector<std::pair<std::string, std::string>> answers;  // completed answers
    std::vector<int> multi_selected;  // indices for multi-select
    bool show_other_input = false;
    std::string other_input_text;
    std::shared_ptr<std::promise<std::optional<std::vector<std::pair<std::string, std::string>>>>> promise;
};

/**
 * @brief Render the question dialog panel.
 * 
 * Mimics the kimi-cli question panel style:
 * - Cyan border with "? QUESTION" title
 * - Question text in yellow with ? prefix
 * - Options numbered [1], [2], etc.
 * - Selected option marked with →
 * - Multi-select shows [ ] / [✓] checkboxes
 * - Descriptions shown dimmed below each option
 */
ftxui::Element render_question_dialog_panel(const QuestionDialogState& state);

} // namespace tui
