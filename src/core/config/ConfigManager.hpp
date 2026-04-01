#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <string_view>
#include "core/llm/routing/PolicyConfig.hpp"

namespace core::config {

enum class SettingsScope {
    User,
    Workspace,
};

enum class ManagedSettingKey {
    DefaultMode,
    DefaultApprovalMode,
    DefaultRouterPolicy,
    UiBanner,
    UiFooter,
    UiModelInfo,
    UiContextUsage,
    UiTimestamps,
    UiSpinner,
    AutoCompactThreshold,
};

struct ManagedSettings {
    std::optional<std::string> default_mode;
    std::optional<std::string> default_approval_mode;
    std::optional<std::string> default_router_policy;
    std::optional<std::string> ui_banner;
    std::optional<std::string> ui_footer;
    std::optional<std::string> ui_model_info;
    std::optional<std::string> ui_context_usage;
    std::optional<std::string> ui_timestamps;
    std::optional<std::string> ui_spinner;
    std::optional<std::string> auto_compact_threshold;

    [[nodiscard]] bool empty() const {
        return !default_mode.has_value()
            && !default_approval_mode.has_value()
            && !default_router_policy.has_value()
            && !ui_banner.has_value()
            && !ui_footer.has_value()
            && !ui_model_info.has_value()
            && !ui_context_usage.has_value()
            && !ui_timestamps.has_value()
            && !ui_spinner.has_value()
            && !auto_compact_threshold.has_value();
    }
};

// ---------------------------------------------------------------------------
// ApiType — wire protocol used by an HTTP provider.
//
// Unlike a provider name ("openai", "grok", "my-endpoint"), ApiType describes
// only the HTTP wire format.  Built-in providers have their ApiType inferred
// from their name by ProviderFactory; user-defined providers must specify it
// explicitly via the `"api_type"` field in settings.json:
//
//   "providers": {
//     "my-groq": {
//       "api_type": "openai",
//       "base_url": "https://api.groq.com/openai/v1",
//       "api_key":  "gsk_…",
//       "model":    "llama-3.3-70b-versatile"
//     }
//   }
//
// Built-in names ("openai", "grok", "claude", "gemini", "mistral", "kimi",
// "ollama") do not need an explicit api_type — ProviderFactory fills it in.
// ---------------------------------------------------------------------------
enum class ApiType {
    Unknown,       ///< Not specified; inferred from provider name by factory
    OpenAI,        ///< OpenAI Chat Completions SSE (also Grok, Mistral, etc.)
    Kimi,          ///< Kimi Code OpenAI-compatible with X-Msh-* headers
    Anthropic,     ///< Anthropic Messages named-SSE
    Gemini,        ///< Google Gemini SSE (`?alt=sse`)
    Ollama,        ///< Ollama NDJSON streaming (`/api/chat`)
    LlamaCppLocal, ///< llama.cpp in-process (not HTTP)
    DashScope,     ///< Alibaba Cloud DashScope / Qwen (OpenAI-compatible + X-DashScope-* headers)
};

[[nodiscard]] std::string_view to_string(ApiType type) noexcept;
[[nodiscard]] ApiType api_type_from_string(std::string_view s) noexcept;

// ---------------------------------------------------------------------------
// McpServerConfig — configuration for one external MCP tool server.
// ---------------------------------------------------------------------------
struct McpServerConfig {
    std::string name;
    std::string transport;   // "stdio" or "http"
    // stdio transport
    std::string command;
    std::vector<std::string> args;
    std::vector<std::string> env;  // additional env vars in KEY=VALUE form
    // http transport
    std::string url;         // e.g. "http://localhost:9999/mcp"
};

// ---------------------------------------------------------------------------
// Engine-specific configuration blocks for local inference engines.
// ---------------------------------------------------------------------------

/// llama.cpp (GGUF / CPU-GPU via ggml) — engine-specific inference settings.
struct LlamaCppConfig {
    std::optional<int>  context_size = {};   ///< llama_context_params.n_ctx
    std::optional<int>  batch_size   = {};   ///< llama_context_params.n_batch
    std::optional<int>  threads      = {};   ///< llama_context_params.n_threads
    std::optional<int>  threads_batch = {};  ///< llama_context_params.n_threads_batch
    std::optional<int>  gpu_layers   = {};   ///< llama_model_params.n_gpu_layers
    std::optional<bool> use_mmap     = {};   ///< llama_model_params.use_mmap
    std::optional<bool> use_mlock    = {};   ///< llama_model_params.use_mlock
};

// ---------------------------------------------------------------------------
// LocalModelConfig — settings common to ALL local inference engines.
// ---------------------------------------------------------------------------
struct LocalModelConfig {
    std::string model_path    = {};  ///< Model file or directory on disk
    std::string chat_template = {};  ///< Jinja2 chat-template override (empty = auto-detect)

    std::optional<float> temperature = {};
    std::optional<float> top_p       = {};
    std::optional<int>   top_k       = {};
    std::optional<int>   seed        = {};

    std::optional<LlamaCppConfig> llamacpp = {};
};

// ---------------------------------------------------------------------------
// ProviderConfig — complete configuration for one named provider entry.
//
// For built-in providers ("openai", "grok", "claude", …) the api_type and
// base_url are optional — ProviderFactory fills in sensible defaults.
// For user-defined providers both fields are required.
//
// Protocol extensions are expressed as flat fields to keep JSON simple:
//   "reasoning_effort": "low"   → injected for OpenAI-compatible reasoning models
//   "thinking_budget": 8000     → enables extended thinking (Anthropic)
//   "stream_usage": true        → appends stream_options to OpenAI requests
//   "wire_api": "responses"     → OpenAI wire API ("chat_completions" or "responses")
//   "service_tier": "priority"  → Responses API service tier override
// ---------------------------------------------------------------------------
struct ProviderConfig {
    ApiType     api_type    = ApiType::Unknown;  ///< Inferred from name when Unknown
    std::string api_key{};
    std::string model;
    std::string base_url{};         ///< Empty = use built-in default for this name
    std::string auth_type{};        ///< "" / "oauth_google" / "oauth_claude" / etc.

    // Protocol extensions (flat; applied per api_type by ProviderFactory)
    std::string reasoning_effort{}; ///< "low" / "high" — OpenAI reasoning models
    std::string wire_api{};         ///< "chat_completions" / "responses" for OpenAI APIs
    std::string service_tier{};     ///< Responses API service_tier (e.g. "priority", "flex")
    int         thinking_budget   = 0;     ///< >0 enables extended thinking (Anthropic)
    bool        stream_usage      = false; ///< Append stream_options (OpenAI)

    /// Populated only when api_type is LlamaCppLocal.
    std::optional<LocalModelConfig> local;
};

struct SubagentConfig {
    std::string description;
    std::string prompt;
    std::string provider;
    std::string model;

    std::optional<std::vector<std::string>> allowed_tools;
    std::optional<bool> use_allow_list;
    std::optional<bool> allow_task_tool;
    std::optional<bool> enabled;
    std::optional<int>  max_steps;
};

struct AppConfig {
    std::string default_provider;
    std::string default_model_selection;
    std::string default_mode;
    std::string default_approval_mode;
    std::string ui_banner;
    std::string ui_footer;
    std::string ui_model_info;
    std::string ui_context_usage;
    std::string ui_timestamps;
    std::string ui_spinner;
    int         auto_compact_threshold = 0;
    std::unordered_map<std::string, ProviderConfig> providers;
    std::unordered_map<std::string, SubagentConfig> subagents;
    std::vector<McpServerConfig> mcp_servers;
    core::llm::routing::RouterConfig router;
    bool has_router_section = false;
};

class ConfigManager {
public:
    static ConfigManager& get_instance() {
        static ConfigManager instance;
        return instance;
    }

    void load(std::optional<std::filesystem::path> working_dir = std::nullopt);

    const AppConfig& get_config() const { return config_; }

    bool persist_model_defaults(std::string_view default_provider,
                                std::string_view default_model_selection,
                                std::string_view specific_model = {},
                                std::string* error = nullptr);

    bool persist_login_profile(std::string_view login_provider,
                               std::string* error = nullptr);

    bool persist_local_provider(std::string_view model_path,
                                std::string_view model_label,
                                std::string* error = nullptr);

    bool persist_managed_setting(SettingsScope scope,
                                 ManagedSettingKey key,
                                 std::optional<std::string> value,
                                 std::optional<std::filesystem::path> working_dir = std::nullopt,
                                 std::string* error = nullptr);

    std::string get_config_dir() const;

    std::filesystem::path get_settings_path(
        SettingsScope scope,
        std::optional<std::filesystem::path> working_dir = std::nullopt) const;

    const ManagedSettings& get_settings_overlay(SettingsScope scope) const {
        return scope == SettingsScope::User ? user_settings_ : workspace_settings_;
    }

private:
    ConfigManager() = default;
    AppConfig config_;
    ManagedSettings user_settings_;
    ManagedSettings workspace_settings_;
    std::filesystem::path last_working_dir_;
};

} // namespace core::config
