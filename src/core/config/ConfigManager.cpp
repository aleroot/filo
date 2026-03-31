#include "ConfigManager.hpp"
#include <simdjson.h>
#include <cstdlib>
#include <pwd.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
#include <filesystem>
#include "core/llm/routing/PolicyLoader.hpp"
#include "core/logging/Logger.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/utils/JsonWriter.hpp"
#include <algorithm>
#include <cctype>
#include <format>
#include <stdexcept>

namespace core::config {

namespace {

namespace fs = std::filesystem;

fs::path model_defaults_overlay_path(const fs::path& config_dir) {
    return config_dir / "model_defaults.json";
}

fs::path auth_defaults_overlay_path(const fs::path& config_dir) {
    return config_dir / "auth_defaults.json";
}

fs::path user_settings_path(const fs::path& config_dir) {
    return config_dir / "settings.json";
}

fs::path workspace_settings_path(const fs::path& working_dir) {
    return working_dir / ".filo" / "settings.json";
}

std::optional<std::string>& managed_setting_slot(ManagedSettings& settings,
                                                 ManagedSettingKey key) {
    switch (key) {
        case ManagedSettingKey::DefaultMode:
            return settings.default_mode;
        case ManagedSettingKey::DefaultApprovalMode:
            return settings.default_approval_mode;
        case ManagedSettingKey::DefaultRouterPolicy:
            return settings.default_router_policy;
        case ManagedSettingKey::UiBanner:
            return settings.ui_banner;
        case ManagedSettingKey::UiFooter:
            return settings.ui_footer;
        case ManagedSettingKey::UiModelInfo:
            return settings.ui_model_info;
        case ManagedSettingKey::UiContextUsage:
            return settings.ui_context_usage;
        case ManagedSettingKey::UiTimestamps:
            return settings.ui_timestamps;
        case ManagedSettingKey::UiSpinner:
            return settings.ui_spinner;
        case ManagedSettingKey::AutoCompactThreshold:
            return settings.auto_compact_threshold;
    }
    return settings.default_mode;
}


} // namespace

[[nodiscard]] std::string_view to_string(ApiType type) noexcept {
    switch (type) {
        case ApiType::OpenAI:        return "openai";
        case ApiType::Kimi:          return "kimi";
        case ApiType::Anthropic:     return "anthropic";
        case ApiType::Gemini:        return "gemini";
        case ApiType::Ollama:        return "ollama";
        case ApiType::LlamaCppLocal: return "llamacpp";
        case ApiType::DashScope:     return "dashscope";
        case ApiType::Unknown:       return "";
    }
    return "";
}

[[nodiscard]] ApiType api_type_from_string(std::string_view s) noexcept {
    if (s == "openai")         return ApiType::OpenAI;
    if (s == "kimi")           return ApiType::Kimi;
    if (s == "anthropic")      return ApiType::Anthropic;
    if (s == "gemini")         return ApiType::Gemini;
    if (s == "ollama")         return ApiType::Ollama;
    if (s == "llamacpp")       return ApiType::LlamaCppLocal;
    if (s == "dashscope")      return ApiType::DashScope;
    if (s == "qwen")           return ApiType::DashScope;  // user-friendly alias
    return ApiType::Unknown;
}

namespace {

void apply_api_key_fallback(std::string_view name, ProviderConfig& provider) {
    if (!provider.api_key.empty()) return;

    struct Entry { const char* prefix; const char* env_var; };
    static constexpr Entry kEnvVars[] = {
        { "grok",    "XAI_API_KEY" },
        { "openai",  "OPENAI_API_KEY" },
        { "claude",  "ANTHROPIC_API_KEY" },
        { "gemini",  "GEMINI_API_KEY" },
        { "mistral", "MISTRAL_API_KEY" },
        { "kimi",    "KIMI_API_KEY" },
    };
    for (const auto& entry : kEnvVars) {
        if (name.starts_with(entry.prefix)) {
            if (const char* e = std::getenv(entry.env_var))
                provider.api_key = e;
            return;
        }
    }
}

ProviderConfig merge_provider(std::string_view name, const ProviderConfig& base, const ProviderConfig& overlay) {
    ProviderConfig merged = base;
    if (overlay.api_type != ApiType::Unknown) merged.api_type = overlay.api_type;
    if (!overlay.api_key.empty())          merged.api_key = overlay.api_key;
    if (!overlay.model.empty())            merged.model = overlay.model;
    if (!overlay.base_url.empty())         merged.base_url = overlay.base_url;
    if (!overlay.reasoning_effort.empty()) merged.reasoning_effort = overlay.reasoning_effort;
    if (!overlay.wire_api.empty())         merged.wire_api = overlay.wire_api;
    if (!overlay.service_tier.empty())     merged.service_tier = overlay.service_tier;
    if (overlay.thinking_budget > 0)       merged.thinking_budget = overlay.thinking_budget;
    if (overlay.stream_usage)              merged.stream_usage = overlay.stream_usage;
    if (!overlay.auth_type.empty())        merged.auth_type = overlay.auth_type;

    if (overlay.local.has_value()) {
        if (!merged.local.has_value()) {
            merged.local = overlay.local;
        } else {
            auto& dst = *merged.local;
            const auto& src = *overlay.local;

            // Universal fields
            if (!src.model_path.empty())    dst.model_path    = src.model_path;
            if (!src.chat_template.empty()) dst.chat_template = src.chat_template;
            if (src.temperature)            dst.temperature   = src.temperature;
            if (src.top_p)                  dst.top_p         = src.top_p;
            if (src.top_k)                  dst.top_k         = src.top_k;
            if (src.seed)                   dst.seed          = src.seed;

            // llama.cpp engine-specific
            if (src.llamacpp.has_value()) {
                if (!dst.llamacpp.has_value()) {
                    dst.llamacpp = src.llamacpp;
                } else {
                    auto& dl = *dst.llamacpp;
                    const auto& sl = *src.llamacpp;
                    if (sl.context_size)  dl.context_size  = sl.context_size;
                    if (sl.batch_size)    dl.batch_size    = sl.batch_size;
                    if (sl.threads)       dl.threads       = sl.threads;
                    if (sl.threads_batch) dl.threads_batch = sl.threads_batch;
                    if (sl.gpu_layers)    dl.gpu_layers    = sl.gpu_layers;
                    if (sl.use_mmap)      dl.use_mmap      = sl.use_mmap;
                    if (sl.use_mlock)     dl.use_mlock     = sl.use_mlock;
                }
            }
            // Future: if (src.mlx.has_value()) { … }
        }
    }

    apply_api_key_fallback(name, merged);
    return merged;
}

struct LoginProfileMapping {
    std::string provider_name;
    std::string auth_type;
    std::string default_model;
};

std::optional<LoginProfileMapping> resolve_login_profile(std::string_view login_provider) {
    std::string normalized;
    normalized.reserve(login_provider.size());
    for (const unsigned char ch : login_provider) {
        if (std::isspace(ch)) continue;
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }

    if (normalized == "claude") {
        return LoginProfileMapping{
            .provider_name = "claude",
            .auth_type = "oauth_claude",
            .default_model = "claude-sonnet-4-6",
        };
    }
    if (normalized == "openai") {
        return LoginProfileMapping{
            .provider_name = "openai",
            .auth_type = "oauth_openai",
            .default_model = "gpt-5.4",
        };
    }
    if (normalized == "openai-pkce"
        || normalized == "openai_pkce"
        || normalized == "openaipkce") {
        return LoginProfileMapping{
            .provider_name = "openai",
            .auth_type = "oauth_openai_pkce",
            .default_model = "gpt-5.4",
        };
    }
    if (normalized == "google") {
        return LoginProfileMapping{
            .provider_name = "gemini",
            .auth_type = "oauth_google",
            .default_model = "gemini-2.5-flash",
        };
    }
    if (normalized == "kimi") {
        return LoginProfileMapping{
            .provider_name = "kimi",
            .auth_type = "oauth_kimi",
            .default_model = "moonshot-v1-8k",
        };
    }
    if (normalized == "qwen") {
        return LoginProfileMapping{
            .provider_name = "qwen",
            .auth_type = "oauth_qwen",
            .default_model = "coder-model",
        };
    }

    return std::nullopt;
}

std::string serialize_login_profile_overlay(const AppConfig& overlay) {
    core::utils::JsonWriter writer(512);
    {
        auto object = writer.object();
        writer.kv_str("default_provider", overlay.default_provider);
        writer.comma();
        writer.kv_str("default_model_selection", overlay.default_model_selection);

        std::vector<std::string> provider_names;
        provider_names.reserve(overlay.providers.size());
        for (const auto& [name, provider] : overlay.providers) {
            if (!provider.auth_type.empty() || !provider.model.empty()) {
                provider_names.push_back(name);
            }
        }
        std::sort(provider_names.begin(), provider_names.end());

        if (!provider_names.empty()) {
            writer.comma();
            writer.key("providers");
            auto providers_object = writer.object();
            for (std::size_t i = 0; i < provider_names.size(); ++i) {
                const std::string& name = provider_names[i];
                const auto provider_it = overlay.providers.find(name);
                if (provider_it == overlay.providers.end()) {
                    continue;
                }
                const ProviderConfig& provider = provider_it->second;

                if (i > 0) writer.comma();
                writer.key(name);
                auto provider_object = writer.object();

                bool has_field = false;
                if (!provider.model.empty()) {
                    writer.kv_str("model", provider.model);
                    has_field = true;
                }
                if (!provider.auth_type.empty()) {
                    if (has_field) writer.comma();
                    writer.kv_str("auth_type", provider.auth_type);
                }
            }
        }
    }

    std::string payload = std::move(writer).take();
    payload.push_back('\n');
    return payload;
}

std::string normalize_subagent_name(std::string_view name) {
    std::string normalized;
    normalized.reserve(name.size());

    std::size_t start = 0;
    while (start < name.size()
           && std::isspace(static_cast<unsigned char>(name[start]))) {
        ++start;
    }
    if (start < name.size() && name[start] == '@') {
        ++start;
    }

    for (std::size_t i = start; i < name.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(name[i]);
        if (std::isspace(ch)) break;
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

SubagentConfig merge_subagent(const SubagentConfig& base, const SubagentConfig& overlay) {
    SubagentConfig merged = base;
    if (!overlay.description.empty()) merged.description = overlay.description;
    if (!overlay.prompt.empty()) merged.prompt = overlay.prompt;
    if (!overlay.provider.empty()) merged.provider = overlay.provider;
    if (!overlay.model.empty()) merged.model = overlay.model;
    if (overlay.allowed_tools.has_value()) merged.allowed_tools = overlay.allowed_tools;
    if (overlay.use_allow_list.has_value()) merged.use_allow_list = overlay.use_allow_list;
    if (overlay.allow_task_tool.has_value()) merged.allow_task_tool = overlay.allow_task_tool;
    if (overlay.enabled.has_value()) merged.enabled = overlay.enabled;
    if (overlay.max_steps.has_value()) merged.max_steps = overlay.max_steps;
    return merged;
}

AppConfig make_default_config() {
    AppConfig config;
    config.default_provider = "grok";
    config.default_model_selection = "manual";
    config.default_mode = "BUILD";
    config.default_approval_mode = "prompt";
    config.ui_banner = "show";
    config.ui_footer = "show";
    config.ui_model_info = "show";
    config.ui_context_usage = "show";
    config.ui_timestamps = "show";
    config.ui_spinner = "show";
    config.auto_compact_threshold = 50000;
    config.router = core::llm::routing::make_default_router_config();

    auto add_provider = [&](std::string name, std::string model,
                             std::string reasoning_effort = {},
                             std::string auth_type = {},
                             std::string base_url = {},
                             std::string wire_api = {},
                             std::string service_tier = {}) {
        ProviderConfig p;
        p.model            = std::move(model);
        p.reasoning_effort = std::move(reasoning_effort);
        p.auth_type        = std::move(auth_type);
        p.base_url         = std::move(base_url);
        p.wire_api         = std::move(wire_api);
        p.service_tier     = std::move(service_tier);
        apply_api_key_fallback(name, p);
        config.providers[name] = std::move(p);
    };

    add_provider("openai",         "gpt-5.4", {}, {}, {}, "responses");
    add_provider("mistral",        "devstral-small-latest");
    add_provider("grok",           "grok-code-fast-1");
    add_provider("grok-4",         "grok-4");
    add_provider("grok-4-fast",    "grok-4-fast-non-reasoning");
    add_provider("grok-reasoning", "grok-4.20-reasoning");
    add_provider("grok-fast",      "grok-4.20-non-reasoning");
    add_provider("grok-mini",      "grok-3-mini",      "high");
    add_provider("grok-mini-fast", "grok-3-mini-fast", "low");
    add_provider("claude",         "claude-sonnet-4-6");
    add_provider("gemini",         "gemini-2.5-flash");
    add_provider("kimi",           "moonshot-v1-8k");
    add_provider("kimi-32k",       "moonshot-v1-32k");
    add_provider("kimi-128k",      "moonshot-v1-128k");
    add_provider("kimi-k2-5",      "kimi-k2-5");
    add_provider("ollama",         "llama3", {}, {}, "http://localhost:11434");

    SubagentConfig general;
    general.description = "General-purpose subagent for complex multi-step tasks and broad research.";
    general.prompt =
        "You are Filo's @general worker subagent.\n"
        "Work independently on delegated tasks and use tools when needed.\n"
        "Return a concise, actionable final summary for the parent agent.";
    general.allow_task_tool = false;
    general.use_allow_list = false;
    general.enabled = true;
    general.max_steps = 12;
    config.subagents["general"] = std::move(general);

    SubagentConfig explore;
    explore.description = "Fast read-only codebase explorer for search-heavy investigations.";
    explore.prompt =
        "You are Filo's @explore worker subagent.\n"
        "Focus on finding information quickly using read/search/list tools.\n"
        "Do not edit files. Cite concrete files and findings in your final summary.";
    explore.allowed_tools = std::vector<std::string>{
        "read_file",
        "file_search",
        "grep_search",
        "list_directory",
        "get_current_time",
    };
    explore.allow_task_tool = false;
    explore.use_allow_list = true;
    explore.enabled = true;
    explore.max_steps = 10;
    config.subagents["explore"] = std::move(explore);

    return config;
}

std::string default_config_json() {
    return R"({
    "default_provider": "grok",
    "default_model_selection": "manual",
    "default_mode": "BUILD",
    "default_approval_mode": "prompt",
    "auto_compact_threshold": 50000,
    "providers": {
        "grok":           { "model": "grok-code-fast-1" },
        "grok-4":         { "model": "grok-4" },
        "grok-4-fast":    { "model": "grok-4-fast-non-reasoning" },
        "grok-reasoning": { "model": "grok-4.20-reasoning" },
        "grok-fast":      { "model": "grok-4.20-non-reasoning" },
        "grok-mini":      { "model": "grok-3-mini",      "reasoning_effort": "high" },
        "grok-mini-fast": { "model": "grok-3-mini-fast", "reasoning_effort": "low" },
        "openai":         { "model": "gpt-5.4", "wire_api": "responses" },
        "mistral":        { "model": "devstral-small-latest" },
        "claude":         { "model": "claude-sonnet-4-6" },
        "claude-thinking":{ "model": "claude-sonnet-4-6", "thinking_budget": 10000 },
        "gemini":         { "model": "gemini-2.5-flash" },
        "gemini-oauth":   { "model": "gemini-2.5-flash", "auth_type": "oauth_google" },
        "kimi":           { "model": "moonshot-v1-8k" },
        "kimi-32k":       { "model": "moonshot-v1-32k" },
        "kimi-128k":      { "model": "moonshot-v1-128k" },
        "kimi-k2-5":      { "model": "kimi-k2-5" },
        "ollama":         { "model": "llama3", "base_url": "http://localhost:11434" }
    },
    "subagents": {
        "general": {
            "description": "General-purpose subagent for complex multi-step tasks and broad research.",
            "prompt": "You are Filo's @general worker subagent.\nWork independently on delegated tasks and use tools when needed.\nReturn a concise, actionable final summary for the parent agent.",
            "allow_task_tool": false,
            "use_allow_list": false,
            "max_steps": 12,
            "enabled": true
        },
        "explore": {
            "description": "Fast read-only codebase explorer for search-heavy investigations.",
            "prompt": "You are Filo's @explore worker subagent.\nFocus on finding information quickly using read/search/list tools.\nDo not edit files. Cite concrete files and findings in your final summary.",
            "allow_task_tool": false,
            "use_allow_list": true,
            "allowed_tools": [
                "read_file",
                "file_search",
                "grep_search",
                "list_directory",
                "get_current_time"
            ],
            "max_steps": 10,
            "enabled": true
        }
    },
    "router": {
        "enabled": true,
        "default_policy": "smart-code",
        "policies": {
            "smart-code": {
                "description": "Embedded smart routing inspired by Requesty policy composition.",
                "strategy": "smart",
                "defaults": [
                    { "provider": "grok", "model": "grok-code-fast-1", "weight": 5 },
                    { "provider": "grok-mini-fast", "model": "grok-3-mini-fast", "weight": 3 },
                    { "provider": "openai", "model": "gpt-5.4", "weight": 2 }
                ],
                "rules": [
                    {
                        "name": "deep-reasoning",
                        "priority": 10,
                        "strategy": "fallback",
                        "when": {
                            "min_prompt_chars": 260,
                            "any_keywords": ["debug", "root cause", "architecture", "design", "migration", "refactor", "reasoning"]
                        },
                        "candidates": [
                            { "provider": "claude", "model": "claude-sonnet-4-6", "retries": 1 },
                            { "provider": "grok-reasoning", "model": "grok-4.20-reasoning", "retries": 1 },
                            { "provider": "openai", "model": "gpt-5.4", "retries": 1 }
                        ]
                    },
                    {
                        "name": "tool-heavy",
                        "priority": 25,
                        "strategy": "latency",
                        "when": { "needs_tool_history": true },
                        "candidates": [
                            { "provider": "grok", "model": "grok-code-fast-1", "latency_bias_ms": 15 },
                            { "provider": "gemini", "model": "gemini-2.5-flash", "latency_bias_ms": 10 },
                            { "provider": "openai", "model": "gpt-5.4", "latency_bias_ms": 20 }
                        ]
                    },
                    {
                        "name": "quick-iteration",
                        "priority": 40,
                        "strategy": "load_balance",
                        "when": { "max_prompt_chars": 220 },
                        "candidates": [
                            { "provider": "grok-mini-fast", "model": "grok-3-mini-fast", "weight": 6 },
                            { "provider": "gemini", "model": "gemini-2.5-flash", "weight": 3 },
                            { "provider": "grok", "model": "grok-code-fast-1", "weight": 1 }
                        ]
                    }
                ]
            }
        }
    }
}
)";
}

ProviderConfig parse_provider(simdjson::dom::object provider_obj) {
    ProviderConfig provider;

    std::string_view value;
    if (!provider_obj["api_type"].get(value)) provider.api_type = api_type_from_string(value);
    if (!provider_obj["api_key"].get(value)) provider.api_key = std::string(value);
    if (!provider_obj["model"].get(value)) provider.model = std::string(value);
    if (!provider_obj["base_url"].get(value)) provider.base_url = std::string(value);
    if (!provider_obj["reasoning_effort"].get(value)) {
        provider.reasoning_effort = std::string(value);
    }
    if (!provider_obj["auth_type"].get(value)) {
        provider.auth_type = std::string(value);
    }
    if (!provider_obj["wire_api"].get(value)) {
        provider.wire_api = std::string(value);
    }
    if (!provider_obj["service_tier"].get(value)) {
        provider.service_tier = std::string(value);
    }
    int64_t thinking_budget = 0;
    if (!provider_obj["thinking_budget"].get(thinking_budget)) {
        provider.thinking_budget = static_cast<int>(thinking_budget);
    }
    bool stream_usage = false;
    if (!provider_obj["stream_usage"].get(stream_usage)) {
        provider.stream_usage = stream_usage;
    }

    // Parse local-model settings.
    // A provider is "local" when its api_type is LlamaCppLocal
    // or when any local-specific key appears in the JSON object.
    // Engine-specific settings live in named sub-objects ("llamacpp", …);
    // flat top-level keys are also accepted for convenience.
    {
        const bool is_local_type = (provider.api_type == ApiType::LlamaCppLocal);

        LocalModelConfig local;
        bool has_local = is_local_type;

        // Universal local-model keys (shared across all engines)
        if (!provider_obj["model_path"].get(value)) {
            local.model_path = std::string(value); has_local = true;
        }
        if (!provider_obj["chat_template"].get(value)) {
            local.chat_template = std::string(value); has_local = true;
        }
        double fv = 0.0;
        int64_t iv = 0;
        if (!provider_obj["temperature"].get(fv)) { local.temperature = static_cast<float>(fv); has_local = true; }
        if (!provider_obj["top_p"].get(fv))       { local.top_p       = static_cast<float>(fv); has_local = true; }
        if (!provider_obj["top_k"].get(iv))        { local.top_k       = static_cast<int>(iv);  has_local = true; }
        if (!provider_obj["seed"].get(iv))         { local.seed        = static_cast<int>(iv);  has_local = true; }

        // Engine-specific sub-objects.  Each key maps to a named sub-object in
        // the JSON; flat top-level keys of the same name are also accepted for
        // convenience (e.g. "context_size": 8192 directly in the provider object).
        // Sub-object values take precedence over flat values when both are present.
        {
            LlamaCppConfig ll;
            bool has_ll = false;

            // Flat top-level convenience keys (matches README examples).
            if (!provider_obj["context_size"].get(iv))  { ll.context_size  = static_cast<int>(iv); has_ll = true; }
            if (!provider_obj["batch_size"].get(iv))    { ll.batch_size    = static_cast<int>(iv); has_ll = true; }
            if (!provider_obj["threads"].get(iv))       { ll.threads       = static_cast<int>(iv); has_ll = true; }
            if (!provider_obj["threads_batch"].get(iv)) { ll.threads_batch = static_cast<int>(iv); has_ll = true; }
            if (!provider_obj["gpu_layers"].get(iv))    { ll.gpu_layers    = static_cast<int>(iv); has_ll = true; }
            {
                bool bv = false;
                if (!provider_obj["use_mmap"].get(bv))  { ll.use_mmap = bv; has_ll = true; }
                if (!provider_obj["use_mlock"].get(bv)) { ll.use_mlock = bv; has_ll = true; }
            }

            // Named sub-object overrides flat keys when present.
            simdjson::dom::object ll_obj;
            if (provider_obj["llamacpp"].get(ll_obj) == simdjson::SUCCESS) {
                if (!ll_obj["context_size"].get(iv))  { ll.context_size  = static_cast<int>(iv); has_ll = true; }
                if (!ll_obj["batch_size"].get(iv))    { ll.batch_size    = static_cast<int>(iv); has_ll = true; }
                if (!ll_obj["threads"].get(iv))       { ll.threads       = static_cast<int>(iv); has_ll = true; }
                if (!ll_obj["threads_batch"].get(iv)) { ll.threads_batch = static_cast<int>(iv); has_ll = true; }
                if (!ll_obj["gpu_layers"].get(iv))    { ll.gpu_layers    = static_cast<int>(iv); has_ll = true; }
                bool bv = false;
                if (!ll_obj["use_mmap"].get(bv))      { ll.use_mmap      = bv; has_ll = true; }
                if (!ll_obj["use_mlock"].get(bv))     { ll.use_mlock     = bv; has_ll = true; }
            }

            if (has_ll) { local.llamacpp = std::move(ll); has_local = true; }
            // Future: simdjson::dom::object mlx_obj;
            // if (provider_obj["mlx"].get(mlx_obj) == simdjson::SUCCESS) { … local.mlx = …; }
        }

        if (has_local) provider.local = std::move(local);
    }

    return provider;
}

SubagentConfig parse_subagent(simdjson::dom::object subagent_obj) {
    SubagentConfig subagent;

    std::string_view value;
    if (!subagent_obj["description"].get(value)) subagent.description = std::string(value);
    if (!subagent_obj["prompt"].get(value)) subagent.prompt = std::string(value);
    if (!subagent_obj["provider"].get(value)) subagent.provider = std::string(value);
    if (!subagent_obj["model"].get(value)) subagent.model = std::string(value);

    bool bool_value = false;
    if (!subagent_obj["use_allow_list"].get(bool_value)) {
        subagent.use_allow_list = bool_value;
    }
    if (!subagent_obj["allow_task_tool"].get(bool_value)) {
        subagent.allow_task_tool = bool_value;
    }
    if (!subagent_obj["enabled"].get(bool_value)) {
        subagent.enabled = bool_value;
    }

    int64_t max_steps = 0;
    if (!subagent_obj["max_steps"].get(max_steps)) {
        subagent.max_steps = static_cast<int>(max_steps);
    }

    simdjson::dom::array tools_arr;
    if (!subagent_obj["allowed_tools"].get(tools_arr)) {
        std::vector<std::string> tools;
        for (simdjson::dom::element tool : tools_arr) {
            std::string_view tool_name;
            if (!tool.get(tool_name)) {
                tools.emplace_back(tool_name);
            }
        }
        subagent.allowed_tools = std::move(tools);
    }

    return subagent;
}

ManagedSettings parse_managed_settings_file(const fs::path& settings_path) {
    simdjson::padded_string json = simdjson::padded_string::load(settings_path.string());
    simdjson::dom::parser parser;
    simdjson::dom::element doc = parser.parse(json);

    ManagedSettings parsed;
    std::string_view value;
    if (!doc["default_mode"].get(value)) {
        parsed.default_mode = std::string(value);
    }
    if (!doc["default_approval_mode"].get(value)) {
        parsed.default_approval_mode = std::string(value);
    }
    if (!doc["default_router_policy"].get(value)) {
        parsed.default_router_policy = std::string(value);
    }
    if (!doc["ui_banner"].get(value)) {
        parsed.ui_banner = std::string(value);
    }
    if (!doc["ui_footer"].get(value)) {
        parsed.ui_footer = std::string(value);
    }
    if (!doc["ui_model_info"].get(value)) {
        parsed.ui_model_info = std::string(value);
    }
    if (!doc["ui_context_usage"].get(value)) {
        parsed.ui_context_usage = std::string(value);
    }
    if (!doc["ui_timestamps"].get(value)) {
        parsed.ui_timestamps = std::string(value);
    }
    if (!doc["ui_spinner"].get(value)) {
        parsed.ui_spinner = std::string(value);
    }
    if (!doc["auto_compact_threshold"].get(value)) {
        parsed.auto_compact_threshold = std::string(value);
    }
    return parsed;
}

std::string serialize_managed_settings(const ManagedSettings& settings) {
    core::utils::JsonWriter writer(256);
    {
        auto object = writer.object();
        bool needs_comma = false;
        auto append_field = [&](std::string_view key,
                                const std::optional<std::string>& value) {
            if (!value.has_value()) {
                return;
            }
            if (needs_comma) {
                writer.comma();
            }
            writer.kv_str(key, *value);
            needs_comma = true;
        };

        append_field("default_mode", settings.default_mode);
        append_field("default_approval_mode", settings.default_approval_mode);
        append_field("default_router_policy", settings.default_router_policy);
        append_field("ui_banner", settings.ui_banner);
        append_field("ui_footer", settings.ui_footer);
        append_field("ui_model_info", settings.ui_model_info);
        append_field("ui_context_usage", settings.ui_context_usage);
        append_field("ui_timestamps", settings.ui_timestamps);
        append_field("ui_spinner", settings.ui_spinner);
        append_field("auto_compact_threshold", settings.auto_compact_threshold);
    }

    std::string output = std::move(writer).take();
    output.push_back('\n');
    return output;
}

void apply_managed_settings(const ManagedSettings& settings, AppConfig& config) {
    if (settings.default_mode.has_value()) {
        config.default_mode = *settings.default_mode;
    }
    if (settings.default_approval_mode.has_value()) {
        config.default_approval_mode = *settings.default_approval_mode;
    }
    if (settings.default_router_policy.has_value()) {
        config.router.default_policy = *settings.default_router_policy;
    }
    if (settings.ui_banner.has_value()) {
        config.ui_banner = *settings.ui_banner;
    }
    if (settings.ui_footer.has_value()) {
        config.ui_footer = *settings.ui_footer;
    }
    if (settings.ui_model_info.has_value()) {
        config.ui_model_info = *settings.ui_model_info;
    }
    if (settings.ui_context_usage.has_value()) {
        config.ui_context_usage = *settings.ui_context_usage;
    }
    if (settings.ui_timestamps.has_value()) {
        config.ui_timestamps = *settings.ui_timestamps;
    }
    if (settings.ui_spinner.has_value()) {
        config.ui_spinner = *settings.ui_spinner;
    }
    if (settings.auto_compact_threshold.has_value()) {
        try {
            config.auto_compact_threshold = std::stoi(*settings.auto_compact_threshold);
        } catch (...) {}
    }
}

AppConfig parse_config_file(const fs::path& config_path) {
    simdjson::padded_string json = simdjson::padded_string::load(config_path.string());
    simdjson::dom::parser parser;
    simdjson::dom::element doc = parser.parse(json);

    AppConfig parsed;

    std::string_view value;
    if (!doc["default_provider"].get(value)) {
        parsed.default_provider = std::string(value);
    }
    if (!doc["default_model_selection"].get(value)) {
        parsed.default_model_selection = std::string(value);
    }
    if (!doc["default_mode"].get(value)) {
        parsed.default_mode = std::string(value);
    }
    if (!doc["default_approval_mode"].get(value)) {
        parsed.default_approval_mode = std::string(value);
    }
    if (!doc["ui_banner"].get(value)) {
        parsed.ui_banner = std::string(value);
    }
    if (!doc["ui_footer"].get(value)) {
        parsed.ui_footer = std::string(value);
    }
    if (!doc["ui_model_info"].get(value)) {
        parsed.ui_model_info = std::string(value);
    }
    if (!doc["ui_context_usage"].get(value)) {
        parsed.ui_context_usage = std::string(value);
    }
    if (!doc["ui_timestamps"].get(value)) {
        parsed.ui_timestamps = std::string(value);
    }
    if (!doc["ui_spinner"].get(value)) {
        parsed.ui_spinner = std::string(value);
    }
    int64_t threshold = 0;
    if (!doc["auto_compact_threshold"].get(threshold)) {
        parsed.auto_compact_threshold = static_cast<int>(threshold);
    }

    simdjson::dom::object providers_obj;
    if (!doc["providers"].get(providers_obj)) {
        for (auto field : providers_obj) {
            simdjson::dom::object provider_obj;
            if (!field.value.get(provider_obj)) {
                parsed.providers[std::string(field.key)] = parse_provider(provider_obj);
            }
        }
    }

    simdjson::dom::object subagents_obj;
    if (!doc["subagents"].get(subagents_obj)) {
        for (auto field : subagents_obj) {
            simdjson::dom::object subagent_obj;
            if (!field.value.get(subagent_obj)) {
                const std::string normalized_name =
                    normalize_subagent_name(std::string(field.key));
                if (!normalized_name.empty()) {
                    parsed.subagents[normalized_name] = parse_subagent(subagent_obj);
                }
            }
        }
    }

    simdjson::dom::object router_obj;
    if (!doc["router"].get(router_obj)) {
        parsed.has_router_section = true;
        std::string error;
        if (!core::llm::routing::parse_router_config(router_obj, parsed.router, error)) {
            throw std::runtime_error(std::string("router config invalid: ") + error);
        }
    }

    simdjson::dom::array mcp_arr;
    if (!doc["mcp_servers"].get(mcp_arr)) {
        for (simdjson::dom::element srv : mcp_arr) {
            McpServerConfig server;
            if (!srv["name"].get(value))      server.name = std::string(value);
            if (!srv["transport"].get(value)) server.transport = std::string(value);
            if (!srv["command"].get(value))   server.command = std::string(value);
            if (!srv["url"].get(value))       server.url = std::string(value);

            simdjson::dom::array args_arr;
            if (!srv["args"].get(args_arr)) {
                for (simdjson::dom::element arg : args_arr) {
                    std::string_view arg_value;
                    if (!arg.get(arg_value)) server.args.push_back(std::string(arg_value));
                }
            }

            simdjson::dom::array env_arr;
            if (!srv["env"].get(env_arr)) {
                for (simdjson::dom::element env_item : env_arr) {
                    std::string_view env_value;
                    if (!env_item.get(env_value)) server.env.push_back(std::string(env_value));
                }
            }

            if (!server.name.empty() && !server.transport.empty()) {
                parsed.mcp_servers.push_back(std::move(server));
            }
        }
    }

    return parsed;
}

void merge_into(AppConfig& base, const AppConfig& overlay) {
    if (!overlay.default_provider.empty()) {
        base.default_provider = overlay.default_provider;
    }
    if (!overlay.default_model_selection.empty()) {
        base.default_model_selection = overlay.default_model_selection;
    }
    if (!overlay.default_mode.empty()) {
        base.default_mode = overlay.default_mode;
    }
    if (!overlay.default_approval_mode.empty()) {
        base.default_approval_mode = overlay.default_approval_mode;
    }
    if (!overlay.ui_banner.empty()) {
        base.ui_banner = overlay.ui_banner;
    }
    if (!overlay.ui_footer.empty()) {
        base.ui_footer = overlay.ui_footer;
    }
    if (!overlay.ui_model_info.empty()) {
        base.ui_model_info = overlay.ui_model_info;
    }
    if (!overlay.ui_context_usage.empty()) {
        base.ui_context_usage = overlay.ui_context_usage;
    }
    if (!overlay.ui_timestamps.empty()) {
        base.ui_timestamps = overlay.ui_timestamps;
    }
    if (!overlay.ui_spinner.empty()) {
        base.ui_spinner = overlay.ui_spinner;
    }
    if (overlay.auto_compact_threshold > 0) {
        base.auto_compact_threshold = overlay.auto_compact_threshold;
    }

    for (const auto& [name, provider] : overlay.providers) {
        auto it = base.providers.find(name);
        if (it == base.providers.end()) {
            ProviderConfig merged = provider;
            apply_api_key_fallback(name, merged);
            base.providers[name] = std::move(merged);
        } else {
            it->second = merge_provider(name, it->second, provider);
        }
    }

    for (const auto& [name, subagent] : overlay.subagents) {
        auto it = base.subagents.find(name);
        if (it == base.subagents.end()) {
            base.subagents[name] = subagent;
        } else {
            it->second = merge_subagent(it->second, subagent);
        }
    }

    base.mcp_servers.insert(base.mcp_servers.end(),
                            overlay.mcp_servers.begin(),
                            overlay.mcp_servers.end());

    if (overlay.has_router_section) {
        core::llm::routing::merge_router_config(base.router, overlay.router);
        base.has_router_section = true;
    }
}

void load_optional_config(const fs::path& path, AppConfig& config) {
    if (!fs::exists(path)) return;
    try {
        merge_into(config, parse_config_file(path));
    } catch (const std::exception& e) {
        core::logging::warn(
            "Failed to parse {}: {}. Using previous configuration values.",
            path.string(),
            e.what());
    }
}

ManagedSettings load_optional_managed_settings(const fs::path& path,
                                               AppConfig& config) {
    if (!fs::exists(path)) {
        return {};
    }

    try {
        ManagedSettings settings = parse_managed_settings_file(path);
        apply_managed_settings(settings, config);
        return settings;
    } catch (const std::exception& e) {
        core::logging::warn(
            "Failed to parse {}: {}. Using previous configuration values.",
            path.string(),
            e.what());
        return {};
    }
}

bool write_text_atomic(const fs::path& path,
                       std::string_view content,
                       std::string& error_out) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
        error_out = std::format(
            "could not create config directory '{}': {}",
            path.parent_path().string(),
            ec.message());
        return false;
    }

    const fs::path tmp_path = path.parent_path() / (path.filename().string() + ".tmp");
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            error_out = std::format(
                "could not write temporary config file '{}'",
                tmp_path.string());
            return false;
        }
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        out.flush();
        if (!out) {
            error_out = std::format(
                "failed while writing temporary config file '{}'",
                tmp_path.string());
            return false;
        }
    }

    fs::rename(tmp_path, path, ec);
    if (ec) {
        // Windows cannot rename over an existing file: remove then retry.
        std::error_code remove_ec;
        fs::remove(path, remove_ec);
        ec.clear();
        fs::rename(tmp_path, path, ec);
    }

    if (ec) {
        std::error_code ignore_ec;
        fs::remove(tmp_path, ignore_ec);
        error_out = std::format(
            "could not replace config file '{}': {}",
            path.string(),
            ec.message());
        return false;
    }

    return true;
}

} // namespace

std::string ConfigManager::get_config_dir() const {
    const char* xdg_config = std::getenv("XDG_CONFIG_HOME");
    if (xdg_config && xdg_config[0] != '\0') {
        return std::string(xdg_config) + "/filo";
    }

    const char* home = std::getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        if (pw) {
            home = pw->pw_dir;
        }
    }

    if (home) {
        return std::string(home) + "/.config/filo";
    }

    return ".filo"; // fallback to current dir
}

std::filesystem::path ConfigManager::get_settings_path(
    SettingsScope scope,
    std::optional<std::filesystem::path> working_dir) const {
    const fs::path config_dir = get_config_dir();
    if (scope == SettingsScope::User) {
        return user_settings_path(config_dir);
    }

    const fs::path cwd = working_dir.value_or(
        last_working_dir_.empty() ? fs::current_path() : last_working_dir_);
    return workspace_settings_path(cwd);
}

void ConfigManager::load(std::optional<std::filesystem::path> working_dir) {
    config_ = make_default_config();
    user_settings_ = {};
    workspace_settings_ = {};

    const fs::path config_dir = get_config_dir();
    const fs::path global_config_path = config_dir / "config.json";
    const fs::path model_overlay_path = model_defaults_overlay_path(config_dir);
    const fs::path auth_overlay_path = auth_defaults_overlay_path(config_dir);
    const fs::path cwd = working_dir.value_or(fs::current_path());
    last_working_dir_ = cwd;
    const fs::path user_settings_file = user_settings_path(config_dir);
    const fs::path project_config_path = cwd / ".filo" / "config.json";
    const fs::path workspace_settings_file = workspace_settings_path(cwd);

    if (!fs::exists(global_config_path)) {
        std::error_code ec;
        fs::create_directories(config_dir, ec);
        std::ofstream out(global_config_path);
        if (out) {
            out << default_config_json();
        }
    }

    load_optional_config(global_config_path, config_);
    load_optional_config(model_overlay_path, config_);
    load_optional_config(auth_overlay_path, config_);
    user_settings_ = load_optional_managed_settings(user_settings_file, config_);
    load_optional_config(project_config_path, config_);
    workspace_settings_ = load_optional_managed_settings(workspace_settings_file, config_);
}

bool ConfigManager::persist_managed_setting(SettingsScope scope,
                                            ManagedSettingKey key,
                                            std::optional<std::string> value,
                                            std::optional<std::filesystem::path> working_dir,
                                            std::string* error) {
    const fs::path cwd = working_dir.value_or(
        last_working_dir_.empty() ? fs::current_path() : last_working_dir_);
    ManagedSettings next_settings =
        scope == SettingsScope::User ? user_settings_ : workspace_settings_;

    auto& slot = managed_setting_slot(next_settings, key);
    slot = std::move(value);

    const fs::path settings_path = get_settings_path(scope, cwd);
    if (next_settings.empty()) {
        std::error_code remove_ec;
        fs::remove(settings_path, remove_ec);
        if (remove_ec) {
            if (error) {
                *error = std::format(
                    "could not remove settings file '{}': {}",
                    settings_path.string(),
                    remove_ec.message());
            }
            return false;
        }
    } else {
        std::string write_error;
        if (!write_text_atomic(settings_path,
                               serialize_managed_settings(next_settings),
                               write_error)) {
            if (error) *error = write_error;
            return false;
        }
    }

    load(cwd);
    if (error) error->clear();
    return true;
}

bool ConfigManager::persist_model_defaults(std::string_view default_provider,
                                           std::string_view default_model_selection,
                                           std::string* error) {
    if (default_provider.empty()) {
        if (error) *error = "default provider cannot be empty";
        return false;
    }
    if (default_model_selection.empty()) {
        if (error) *error = "default model selection cannot be empty";
        return false;
    }

    const std::string provider_str(default_provider);
    const std::string selection_str(default_model_selection);
    const fs::path overlay_path = model_defaults_overlay_path(get_config_dir());

    const std::string payload = std::format(
        "{{\n"
        "    \"default_provider\": \"{}\",\n"
        "    \"default_model_selection\": \"{}\"\n"
        "}}\n",
        core::utils::escape_json_string(provider_str),
        core::utils::escape_json_string(selection_str));

    std::string write_error;
    if (!write_text_atomic(overlay_path, payload, write_error)) {
        if (error) *error = write_error;
        return false;
    }

    config_.default_provider = provider_str;
    config_.default_model_selection = selection_str;
    return true;
}

bool ConfigManager::persist_login_profile(std::string_view login_provider,
                                          std::string* error) {
    if (login_provider.empty()) {
        if (error) *error = "login provider cannot be empty";
        return false;
    }

    const auto mapping = resolve_login_profile(login_provider);
    if (!mapping.has_value()) {
        if (error) {
            *error = std::format(
                "unsupported login provider '{}'",
                std::string(login_provider));
        }
        return false;
    }

    const fs::path overlay_path = auth_defaults_overlay_path(get_config_dir());
    AppConfig overlay;
    if (fs::exists(overlay_path)) {
        try {
            overlay = parse_config_file(overlay_path);
        } catch (const std::exception& e) {
            core::logging::warn(
                "Failed to parse {}: {}. Rewriting auth login overlay.",
                overlay_path.string(),
                e.what());
        }
    }

    overlay.default_provider = mapping->provider_name;
    overlay.default_model_selection = "manual";

    ProviderConfig& provider_overlay = overlay.providers[mapping->provider_name];
    provider_overlay.auth_type = mapping->auth_type;
    if (provider_overlay.model.empty()) {
        if (const auto it = config_.providers.find(mapping->provider_name);
            it != config_.providers.end() && !it->second.model.empty()) {
            provider_overlay.model = it->second.model;
        } else {
            provider_overlay.model = mapping->default_model;
        }
    }

    std::string write_error;
    if (!write_text_atomic(overlay_path,
                           serialize_login_profile_overlay(overlay),
                           write_error)) {
        if (error) *error = write_error;
        return false;
    }

    config_.default_provider = mapping->provider_name;
    config_.default_model_selection = "manual";
    ProviderConfig& provider = config_.providers[mapping->provider_name];
    provider.auth_type = mapping->auth_type;
    if (provider.model.empty()) {
        provider.model = provider_overlay.model;
    }

    if (error) error->clear();
    return true;
}

bool ConfigManager::persist_local_provider(std::string_view model_path,
                                            std::string_view model_label,
                                            std::string* error) {
    const fs::path overlay_path = model_defaults_overlay_path(get_config_dir());

    const std::string payload = std::format(
        "{{\n"
        "    \"default_provider\": \"local\",\n"
        "    \"default_model_selection\": \"manual\",\n"
        "    \"providers\": {{\n"
        "        \"local\": {{\n"
        "            \"api_type\": \"llamacpp\",\n"
        "            \"model\": \"{}\",\n"
        "            \"model_path\": \"{}\"\n"
        "        }}\n"
        "    }}\n"
        "}}\n",
        core::utils::escape_json_string(std::string(model_label)),
        core::utils::escape_json_string(std::string(model_path)));

    std::string write_error;
    if (!write_text_atomic(overlay_path, payload, write_error)) {
        if (error) *error = write_error;
        return false;
    }

    config_.default_provider = "local";
    config_.default_model_selection = "manual";
    config_.providers["local"] = ProviderConfig{
        .api_type = ApiType::LlamaCppLocal,
        .model    = std::string(model_label),
        .local    = LocalModelConfig{
            .model_path = std::string(model_path),
        },
    };
    return true;
}

} // namespace core::config
