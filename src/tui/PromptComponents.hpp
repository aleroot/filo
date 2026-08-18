#pragma once

#include "Autocomplete.hpp"
#include "DiffPreview.hpp"
#include "CodeBlockRunner.hpp"
#include "RewindPicker.hpp"
#include "core/llm/Models.hpp"
#include "core/scm/SourceControlProvider.hpp"
#include "core/session/SessionStore.hpp"
#include <ftxui/dom/elements.hpp>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <utility>

namespace tui {

enum class SessionPickerResource;

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

struct ModelProviderPickerRow {
    std::string name;
    std::string description;
    bool active = false;
};

struct ModelPickerRow {
    std::string id;
    std::string selector;
    std::string source_provider;
    std::string description;
    bool active = false;
    bool provider_default = false;
};

struct OptionPickerRow {
    std::string value;
    std::string label;
    std::string description;
    bool active = false;
};

struct ConversationSearchHit {
    int message_index = -1;
    std::string role;
    std::string snippet;
};

struct ThreadTab {
    std::string label;
    bool active = false;
    bool running = false;
};

using ReviewBaseRef = core::scm::BranchRef;

enum class ReviewPickerMode {
    SelectTarget,
    EnterBaseBranch,
    EnterCustomPrompt,
};

[[nodiscard]] std::string format_workspace_status_label(
    std::string_view workspace_path,
    bool sandbox_enabled);

[[nodiscard]] std::string format_runtime_status_summary(
    std::string_view provider_name,
    std::string_view model_name,
    int mcp_server_count);

/// Compact footer badge for the active provider/model, optionally including a
/// non-default effort level. Empty `effort_level` means auto/provider default
/// and is omitted so the badge stays `provider · model`. Internal value
/// `none` is shown as `off` to match the `/effort` command language.
[[nodiscard]] std::string format_model_status_badge(
    std::string_view provider_name,
    std::string_view model_name,
    std::string_view effort_level = {});

/// Return setup guidance only when the selected provider has no usable
/// configured credential. OAuth-backed Grok profiles are already ready even
/// though they intentionally have no API key.
[[nodiscard]] std::string format_provider_setup_hint(
    std::string_view provider_name,
    std::string_view api_key,
    std::string_view auth_type);

/// Compact token counters for subscription providers. Cache and reasoning
/// details are useful when no quota window is available, but become noise once
/// the footer can show the provider's authoritative utilization windows.
[[nodiscard]] std::string format_subscription_token_usage(
    const core::llm::TokenUsage& usage,
    bool has_usage_windows);

/// Lifecycle of the compact footer signal for the assistant turn.
enum class TurnActivityState {
    Idle,       ///< No turn activity to report: renders nothing.
    Active,     ///< A turn is running: animated quadrant-filling circle.
    Completed,  ///< The last turn finished successfully: static success tick.
    Failed,     ///< The last turn was cancelled or errored: static cross.
};

/// Compact footer signal for the assistant turn. While a turn runs it shows
/// the same quadrant-filling circle animation as the tool call rows; once the
/// work is finished it settles on a static outcome glyph, mirroring a
/// completed subagent: a tick on success, a cross when the turn was cancelled
/// or ended with an error. Returns an empty element when there is no turn
/// activity to report. When animation is disabled an active turn shows a
/// static dot.
ftxui::Element render_turn_activity_indicator(TurnActivityState state,
                                              bool animate,
                                              std::size_t tick);

ftxui::Element render_default_prompt_panel(ftxui::Element input_line,
                                           std::string_view input_text);

ftxui::Element render_prompt_box(ftxui::Element input_line,
                                 ftxui::Color accent);

ftxui::Element render_command_prompt_panel(const std::vector<CommandSuggestion>& suggestions,
                                           int selected_index,
                                           ftxui::Element input_line,
                                           std::string_view input_text);

ftxui::Element render_mention_prompt_panel(const std::vector<MentionSuggestion>& suggestions,
                                           int selected_index,
                                           ftxui::Element input_line,
                                           std::string_view input_text);

/// @p origin_label identifies the thread that triggered the prompt; empty
/// for the current thread, rendered when a hidden thread requests approval.
ftxui::Element render_permission_prompt_panel(std::string_view tool_name,
                                              std::string_view args_preview,
                                              const ToolDiffPreview& diff_preview,
                                              std::string_view allow_label,
                                              int selected_index,
                                              std::string_view origin_label = {});

ftxui::Element render_startup_banner_panel(std::string_view provider_name,
                                           std::string_view model_name,
                                           int mcp_server_count,
                                           std::string_view context_sources_label,
                                           std::string_view provider_setup_hint,
                                           std::string_view clock_label = {},
                                           const std::vector<ThreadTab>& thread_tabs = {},
                                           std::vector<ftxui::Box>* thread_tab_hitboxes = nullptr,
                                           ftxui::Box* context_sources_hitbox = nullptr);

ftxui::Element render_model_selection_panel(int selected_index,
                                            std::string_view manual_description,
                                            std::string_view router_description,
                                            std::string_view auto_description,
                                            std::string_view router_policy,
                                            bool router_available,
                                            bool local_model_available);

ftxui::Element render_model_provider_picker_panel(
    const std::vector<ModelProviderPickerRow>& providers,
    int selected_index,
    bool local_model_available);

ftxui::Element render_provider_model_picker_panel(
    std::string_view provider_name,
    const std::vector<ModelPickerRow>& models,
    int selected_index);

ftxui::Element render_option_selection_panel(std::string_view title,
                                             const std::vector<OptionPickerRow>& options,
                                             int selected_index,
                                             std::string_view current_value,
                                             std::string_view help_text);

ftxui::Element render_provider_selection_panel(const std::vector<std::string>& providers,
                                               int selected_index);

ftxui::Element render_authentication_recovery_panel(
    std::string_view provider_name,
    std::string_view reason,
    bool retry_safe,
    int selected_index);

ftxui::Element render_settings_panel(std::string_view scope_label,
                                     std::string_view scope_path,
                                     const std::vector<SettingsPanelRow>& rows,
                                     int selected_index,
                                     std::string_view status_message);

ftxui::Element render_local_model_picker_panel(std::string_view current_dir,
                                               const std::vector<LocalModelEntry>& entries,
                                               int selected_index);

/// Shared renderer whose explicit resource controls thread-vs-session chrome.
/// @p filtered is the currently visible catalogue.
ftxui::Element render_session_picker_panel(
    const std::vector<core::session::SessionInfo>& filtered,
    int selected_index,
    SessionPickerResource resource,
    std::string_view current_session_id = {},
    std::string_view query = {},
    bool filter_active = false,
    bool rename_active = false,
    std::string_view rename_buffer = {},
    std::string_view status_message = {},
    const std::unordered_set<std::string>& running_session_ids = {});

ftxui::Element render_prompts_picker_panel(const std::vector<std::string>& prompts,
                                           int selected_index,
                                           std::string_view status_message);

ftxui::Element render_review_picker_panel(ReviewPickerMode mode,
                                          int selected_index,
                                          std::string_view input_text,
                                          const std::vector<ReviewBaseRef>& base_refs = {},
                                          int selected_base_ref = 0);

ftxui::Element render_conversation_search_panel(
    std::string_view query,
    const std::vector<ConversationSearchHit>& hits,
    int selected_index);

ftxui::Element render_rewind_picker_panel(
    const std::vector<RewindPickerOption>& options,
    int selected_index);

ftxui::Element render_code_block_runner_panel(const CodeBlockRunnerState& state);

ftxui::Element render_stderr_panel(const std::vector<std::string>& lines);

} // namespace tui
