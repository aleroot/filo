#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/config/ConfigManager.hpp"

#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

struct ScopedEnvVar {
    std::string name;
    std::optional<std::string> old_value;

    ScopedEnvVar(std::string env_name, const std::string& value)
        : name(std::move(env_name)) {
        if (const char* existing = std::getenv(name.c_str())) {
            old_value = std::string(existing);
        }
        setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (old_value.has_value()) {
            setenv(name.c_str(), old_value->c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

fs::path make_temp_dir(const std::string& label) {
    const auto stamp = std::to_string(static_cast<long long>(std::rand()));
    fs::path path = fs::temp_directory_path() / (label + "_" + stamp);
    fs::create_directories(path);
    return path;
}

void write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << text;
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

const std::optional<std::string>& managed_overlay_value(
    const core::config::ManagedSettings& settings,
    core::config::ManagedSettingKey key) {
    switch (key) {
        case core::config::ManagedSettingKey::DefaultMode:
            return settings.default_mode;
        case core::config::ManagedSettingKey::DefaultApprovalMode:
            return settings.default_approval_mode;
        case core::config::ManagedSettingKey::DefaultRouterPolicy:
            return settings.default_router_policy;
        case core::config::ManagedSettingKey::UiBanner:
            return settings.ui_banner;
        case core::config::ManagedSettingKey::UiFooter:
            return settings.ui_footer;
        case core::config::ManagedSettingKey::UiModelInfo:
            return settings.ui_model_info;
        case core::config::ManagedSettingKey::UiContextUsage:
            return settings.ui_context_usage;
        case core::config::ManagedSettingKey::UiTimestamps:
            return settings.ui_timestamps;
        case core::config::ManagedSettingKey::UiSpinner:
            return settings.ui_spinner;
        case core::config::ManagedSettingKey::AutoCompactThreshold:
            return settings.auto_compact_threshold;
        case core::config::ManagedSettingKey::ContextCompression:
            return settings.context_compression;
    }
    return settings.default_mode;
}

std::string effective_managed_value(const core::config::AppConfig& config,
                                    core::config::ManagedSettingKey key) {
    switch (key) {
        case core::config::ManagedSettingKey::DefaultMode:
            return config.default_mode;
        case core::config::ManagedSettingKey::DefaultApprovalMode:
            return config.default_approval_mode;
        case core::config::ManagedSettingKey::DefaultRouterPolicy:
            return config.router.default_policy;
        case core::config::ManagedSettingKey::UiBanner:
            return config.ui_banner;
        case core::config::ManagedSettingKey::UiFooter:
            return config.ui_footer;
        case core::config::ManagedSettingKey::UiModelInfo:
            return config.ui_model_info;
        case core::config::ManagedSettingKey::UiContextUsage:
            return config.ui_context_usage;
        case core::config::ManagedSettingKey::UiTimestamps:
            return config.ui_timestamps;
        case core::config::ManagedSettingKey::UiSpinner:
            return config.ui_spinner;
        case core::config::ManagedSettingKey::AutoCompactThreshold:
            return std::to_string(config.auto_compact_threshold);
        case core::config::ManagedSettingKey::ContextCompression:
            return config.context_compression;
    }
    return config.default_mode;
}

} // namespace

TEST_CASE("ConfigManager prefers project config and merges provider overrides", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_merge");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path local_config = project_dir / ".filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());
    ScopedEnvVar mistral_key("MISTRAL_API_KEY", "env-mistral-key");

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_mode": "BUILD",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-4o" },
            "mistral": { "type": "mistral", "model": "devstral-small-latest" }
        }
    })");

    write_text(local_config, R"({
        "default_provider": "mistral",
        "default_mode": "debug",
        "providers": {
            "mistral": { "model": "codestral-latest" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& config = manager.get_config();

    REQUIRE(config.default_provider == "mistral");
    REQUIRE(config.default_model_selection == "manual");
    REQUIRE(config.default_mode == "debug");
    REQUIRE(config.providers.contains("openai"));
    REQUIRE(config.providers.contains("mistral"));
    REQUIRE(config.providers.at("mistral").model == "codestral-latest");
    REQUIRE(config.providers.at("mistral").api_key == "env-mistral-key");
    REQUIRE(config.subagents.contains("general"));
    REQUIRE(config.router.policies.contains("smart-code"));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager parses tool output history settings", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_context_compression");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path local_config = project_dir / ".filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(local_config, R"({
        "tool_output_token_limit": 6144,
        "context_compression": "light"
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& config = manager.get_config();

    REQUIRE(config.tool_output_token_limit == 6144);
    REQUIRE(config.context_compression == "light");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager merges subagent overrides and custom profiles", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_subagents");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path local_config = project_dir / ".filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "subagents": {
            "general": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "max_steps": 18
            },
            "explore": {
                "use_allow_list": true,
                "allowed_tools": ["read_file", "file_search"]
            }
        }
    })");

    write_text(local_config, R"({
        "subagents": {
            "general": {
                "model": "gpt-4o",
                "max_steps": 6
            },
            "explore": {
                "enabled": false
            },
            "analysis": {
                "description": "Custom analysis worker",
                "prompt": "Investigate and summarize.",
                "response_format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                        "additionalProperties": false
                    }
                },
                "max_steps": 4
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& config = manager.get_config();

    REQUIRE(config.subagents.contains("general"));
    REQUIRE(config.subagents.contains("explore"));
    REQUIRE(config.subagents.contains("analysis"));

    const auto& general = config.subagents.at("general");
    REQUIRE(general.provider == "openai");
    REQUIRE(general.model == "gpt-4o");
    REQUIRE(general.max_steps.has_value());
    REQUIRE(general.max_steps.value() == 6);

    const auto& explore = config.subagents.at("explore");
    REQUIRE(explore.enabled.has_value());
    REQUIRE_FALSE(explore.enabled.value());
    REQUIRE(explore.allowed_tools.has_value());
    REQUIRE(explore.allowed_tools->size() == 2);

    const auto& analysis = config.subagents.at("analysis");
    REQUIRE(analysis.description == "Custom analysis worker");
    REQUIRE(analysis.prompt == "Investigate and summarize.");
    REQUIRE(analysis.max_steps.has_value());
    REQUIRE(analysis.max_steps.value() == 4);
    REQUIRE(analysis.response_format.has_value());
    REQUIRE(analysis.response_format->type == core::llm::ResponseFormat::Type::JsonSchema);
    REQUIRE(analysis.response_format->schema.contains("summary"));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager applies active profile overlays with inheritance", "[config][profiles]") {
    const fs::path sandbox = make_temp_dir("filo_config_profiles_apply");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path profile_defaults = xdg_home / "filo" / "profile_defaults.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "default_mode": "BUILD",
        "default_approval_mode": "prompt",
        "providers": {
            "openai": { "model": "gpt-4o" },
            "grok": { "model": "grok-code-fast-1" },
            "mistral": { "model": "devstral-small-latest" }
        },
        "profiles": {
            "work": {
                "description": "Work defaults",
                "default_provider": "mistral",
                "default_mode": "DEBUG",
                "providers": {
                    "mistral": { "model": "codestral-latest" }
                }
            },
            "oss": {
                "extends_from": ["work"],
                "default_provider": "grok",
                "default_approval_mode": "yolo"
            }
        }
    })");

    write_text(profile_defaults, R"({
        "active_profile": "oss"
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& config = manager.get_config();

    REQUIRE(manager.get_active_profile() == "oss");
    REQUIRE(manager.get_profiles().contains("work"));
    REQUIRE(manager.get_profiles().contains("oss"));
    REQUIRE(config.default_provider == "grok");
    REQUIRE(config.default_mode == "DEBUG");
    REQUIRE(config.default_approval_mode == "yolo");
    REQUIRE(config.providers.at("mistral").model == "codestral-latest");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists active profile selection and can clear it", "[config][profiles]") {
    const fs::path sandbox = make_temp_dir("filo_config_profiles_persist");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path profile_defaults = xdg_home / "filo" / "profile_defaults.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "providers": {
            "openai": { "model": "gpt-4o" },
            "grok": { "model": "grok-code-fast-1" }
        },
        "profiles": {
            "work": {
                "default_provider": "openai",
                "default_mode": "BUILD"
            },
            "travel": {
                "default_provider": "grok",
                "default_mode": "RESEARCH"
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    REQUIRE(manager.persist_active_profile(std::string("work"), project_dir, &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_active_profile() == "work");
    REQUIRE(fs::exists(profile_defaults));
    REQUIRE(read_text(profile_defaults).find("\"active_profile\": \"work\"") != std::string::npos);

    REQUIRE_FALSE(manager.persist_active_profile(std::string("unknown"), project_dir, &error));
    REQUIRE(error.find("not defined") != std::string::npos);

    REQUIRE(manager.persist_active_profile(std::nullopt, project_dir, &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_active_profile().empty());
    REQUIRE_FALSE(fs::exists(profile_defaults));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager rejects persisting unresolved profile inheritance", "[config][profiles]") {
    const fs::path sandbox = make_temp_dir("filo_config_profiles_invalid");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path profile_defaults = xdg_home / "filo" / "profile_defaults.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "providers": {
            "openai": { "model": "gpt-4o" }
        },
        "profiles": {
            "dangling": {
                "extends_from": ["missing-parent"],
                "default_mode": "RESEARCH"
            },
            "cycle-a": {
                "extends_from": ["cycle-b"],
                "default_mode": "BUILD"
            },
            "cycle-b": {
                "extends_from": ["cycle-a"],
                "default_mode": "DEBUG"
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    REQUIRE_FALSE(manager.persist_active_profile(std::string("dangling"), project_dir, &error));
    REQUIRE(error.find("could not be applied") != std::string::npos);
    REQUIRE(error.find("not defined") != std::string::npos);
    REQUIRE_FALSE(fs::exists(profile_defaults));

    REQUIRE_FALSE(manager.persist_active_profile(std::string("cycle-a"), project_dir, &error));
    REQUIRE(error.find("could not be applied") != std::string::npos);
    REQUIRE(error.find("cycle") != std::string::npos);
    REQUIRE_FALSE(fs::exists(profile_defaults));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager defaults leave openai wire_api unset", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_defaults_wire_api");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";

    fs::create_directories(project_dir);
    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    const auto& config = manager.get_config();
    REQUIRE(config.providers.contains("openai"));
    REQUIRE(config.providers.at("openai").wire_api.empty());

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager parses provider wire_api overrides", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_wire_api");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path local_config = project_dir / ".filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "providers": {
            "openai": {
                "type": "openai",
                "model": "gpt-5",
                "wire_api": "responses",
                "service_tier": "priority"
            }
        }
    })");

    write_text(local_config, R"({
        "providers": {
            "openai": {
                "wire_api": "chat_completions",
                "service_tier": "flex"
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& provider = manager.get_config().providers.at("openai");

    REQUIRE(provider.model == "gpt-5");
    REQUIRE(provider.wire_api == "chat_completions");
    REQUIRE(provider.service_tier == "flex");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager parses embedded llama.cpp provider settings", "[config][llamacpp]") {
    const fs::path sandbox = make_temp_dir("filo_config_llamacpp");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path local_config = project_dir / ".filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    // Global config: use the preferred nested "llamacpp" sub-object format.
    write_text(global_config, R"({
        "providers": {
            "local-code": {
                "api_type": "llamacpp",
                "model": "qwen2.5-coder",
                "model_path": "/models/qwen2.5-coder.gguf",
                "temperature": 0.15,
                "top_p": 0.9,
                "llamacpp": {
                    "context_size": 8192,
                    "threads": 8,
                    "gpu_layers": 24,
                    "use_mmap": true
                }
            }
        }
    })");

    // Workspace overlay: also uses the nested format; verifies overlay merging.
    write_text(local_config, R"({
        "providers": {
            "local-code": {
                "chat_template": "chatml",
                "llamacpp": {
                    "threads": 12,
                    "batch_size": 1024
                }
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& provider = manager.get_config().providers.at("local-code");

    REQUIRE(provider.api_type == core::config::ApiType::LlamaCppLocal);
    REQUIRE(provider.model == "qwen2.5-coder");
    REQUIRE(provider.local.has_value());
    const auto& local = *provider.local;
    REQUIRE(local.model_path == "/models/qwen2.5-coder.gguf");
    REQUIRE(local.chat_template == "chatml");
    REQUIRE(local.temperature.has_value());
    REQUIRE(local.temperature.value() == Catch::Approx(0.15f));
    REQUIRE(local.top_p.has_value());
    REQUIRE(local.top_p.value() == Catch::Approx(0.9f));
    REQUIRE(local.llamacpp.has_value());
    const auto& ll = *local.llamacpp;
    REQUIRE(ll.context_size == 8192);
    REQUIRE(ll.batch_size == 1024);   // merged from workspace overlay
    REQUIRE(ll.threads == 12);        // overridden by workspace overlay
    REQUIRE(ll.gpu_layers == 24);
    REQUIRE(ll.use_mmap == true);

    fs::remove_all(sandbox);
}


TEST_CASE("ConfigManager resets state on repeated loads instead of duplicating MCP servers", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_reset");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "providers": {
            "openai": { "type": "openai", "model": "gpt-4o" }
        },
        "mcp_servers": [
            { "name": "fetch", "transport": "stdio", "command": "uvx", "args": ["mcp-server-fetch"] }
        ]
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    manager.load(project_dir);

    const auto& config = manager.get_config();
    REQUIRE(config.mcp_servers.size() == 1);
    REQUIRE(config.mcp_servers[0].name == "fetch");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager merges managed user and workspace settings with expected precedence",
          "[config]") {
    const fs::path sandbox = make_temp_dir("filo_managed_settings_precedence");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path model_defaults = xdg_home / "filo" / "model_defaults.json";
    const fs::path user_settings = xdg_home / "filo" / "settings.json";
    const fs::path local_config = project_dir / ".filo" / "config.json";
    const fs::path local_settings = project_dir / ".filo" / "settings.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "default_mode": "BUILD",
        "default_approval_mode": "prompt",
        "router": {
            "enabled": true,
            "default_policy": "fast",
            "policies": {
                "fast": {
                    "defaults": [{ "provider": "openai", "model": "gpt-4o" }]
                },
                "deep": {
                    "defaults": [{ "provider": "grok", "model": "grok-code-fast-1" }]
                }
            }
        },
        "providers": {
            "openai": { "type": "openai", "model": "gpt-4o" },
            "grok": { "type": "grok", "model": "grok-code-fast-1" },
            "mistral": { "type": "mistral", "model": "devstral-small-latest" }
        }
    })");

    write_text(model_defaults, R"({
        "default_provider": "grok",
        "default_model_selection": "manual"
    })");

    write_text(user_settings, R"({
        "default_approval_mode": "yolo",
        "ui_footer": "hide",
        "context_compression": "light"
    })");

    write_text(local_config, R"({
        "default_mode": "DEBUG"
    })");

    write_text(local_settings, R"({
        "default_router_policy": "deep",
        "ui_banner": "hide",
        "context_compression": "ultra"
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& config = manager.get_config();

    REQUIRE(config.default_provider == "grok");
    REQUIRE(config.default_model_selection == "manual");
    REQUIRE(config.default_mode == "DEBUG");
    REQUIRE(config.default_approval_mode == "yolo");
    REQUIRE(config.router.default_policy == "deep");
    REQUIRE(config.ui_footer == "hide");
    REQUIRE(config.ui_banner == "hide");
    REQUIRE(config.context_compression == "ultra");
    REQUIRE(manager.get_settings_overlay(core::config::SettingsScope::User).ui_footer
            == std::optional<std::string>{"hide"});
    REQUIRE(manager.get_settings_overlay(core::config::SettingsScope::User).context_compression
            == std::optional<std::string>{"light"});
    REQUIRE(manager.get_settings_overlay(core::config::SettingsScope::Workspace).default_router_policy
            == std::optional<std::string>{"deep"});
    REQUIRE(manager.get_settings_overlay(core::config::SettingsScope::Workspace).context_compression
            == std::optional<std::string>{"ultra"});

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager writes Grok-first defaults for a fresh install", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_defaults");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& config = manager.get_config();

    REQUIRE(config.default_provider == "grok");
    REQUIRE(config.default_model_selection == "manual");
    REQUIRE(config.default_approval_mode == "prompt");
    REQUIRE(config.ui_banner == "show");
    REQUIRE(config.ui_footer == "show");
    REQUIRE(config.ui_model_info == "show");
    REQUIRE(config.ui_context_usage == "show");
    REQUIRE(config.ui_timestamps == "show");
    REQUIRE(config.ui_spinner == "show");
    REQUIRE(config.tool_output_token_limit == 3072);
    REQUIRE(config.providers.contains("grok"));
    REQUIRE(config.providers.at("grok").model == "grok-code-fast-1");
    REQUIRE(config.providers.contains("grok-4-5"));
    REQUIRE(config.providers.at("grok-4-5").model == "grok-4.5");
    REQUIRE(config.providers.at("grok-4-5").wire_api == "responses");
    REQUIRE(config.providers.contains("grok-reasoning"));
    REQUIRE(config.providers.at("grok-reasoning").model == "grok-4.5");
    REQUIRE(config.providers.at("grok-reasoning").wire_api == "responses");
    REQUIRE(config.providers.contains("openai"));
    REQUIRE(config.providers.at("openai").model == "gpt-5.6-sol");
    REQUIRE(config.providers.contains("mistral"));
    REQUIRE(config.providers.at("mistral").model == "mistral-vibe-cli-latest");
    REQUIRE(config.providers.at("mistral").reasoning_effort == "high");
    REQUIRE(config.providers.contains("kimi"));
    REQUIRE(config.providers.at("kimi").model == "kimi-k3");
    REQUIRE(config.providers.at("kimi").reasoning_effort == "max");
    REQUIRE(config.providers.contains("kimi-code"));
    REQUIRE(config.providers.at("kimi-code").model == "k3");
    REQUIRE(config.providers.at("kimi-code").base_url == "https://api.kimi.com/coding/v1");
    REQUIRE(config.providers.contains("kimi-code-fast"));
    REQUIRE(config.providers.at("kimi-code-fast").model == "kimi-for-coding-highspeed");
    REQUIRE(config.providers.contains("kimi-k2-6"));
    REQUIRE(config.providers.at("kimi-k2-6").model == "kimi-k2.6");
    REQUIRE(config.providers.contains("kimi-k2-5"));
    REQUIRE(config.providers.at("kimi-k2-5").model == "kimi-k2.5");
    REQUIRE(config.providers.contains("kimi-for-coding"));
    REQUIRE(config.providers.at("kimi-for-coding").model == "kimi-for-coding");
    REQUIRE(config.providers.at("kimi-for-coding").base_url == "https://api.kimi.com/coding/v1");
    REQUIRE(config.providers.contains("claude"));
    REQUIRE_FALSE(config.providers.contains("claude-fable"));
    REQUIRE_FALSE(config.providers.contains("claude-haiku"));
    REQUIRE_FALSE(config.providers.contains("claude-oauth"));
    REQUIRE(config.subagents.contains("general"));
    REQUIRE(config.subagents.contains("explore"));
    REQUIRE(config.subagents.at("general").max_steps.has_value());
    REQUIRE(config.subagents.at("general").max_steps.value() == 12);
    REQUIRE(config.subagents.at("explore").use_allow_list.has_value());
    REQUIRE(config.subagents.at("explore").use_allow_list.value());
    REQUIRE(config.router.enabled);
    REQUIRE(config.router.policies.contains("smart-code"));
    REQUIRE(fs::exists(xdg_home / "filo" / "config.json"));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persist_login_profile('kimi') selects oauth_kimi and K3 default",
          "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_login_kimi_profile");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-5.4" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    REQUIRE(manager.persist_login_profile("kimi", &error));
    REQUIRE(error.empty());

    const auto& config = manager.get_config();
    REQUIRE(config.default_provider == "kimi");
    REQUIRE(config.default_model_selection == "manual");
    REQUIRE(config.providers.contains("kimi"));
    REQUIRE(config.providers.at("kimi").auth_type == "oauth_kimi");
    REQUIRE(config.providers.at("kimi").model == "k3");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persist_login_profile('qwen') selects Token Plan and preserves its key",
          "[config][qwen]") {
    const fs::path sandbox = make_temp_dir("filo_config_login_qwen_profile");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path auth_overlay = xdg_home / "filo" / "auth_defaults.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-5.4" }
        }
    })");
    write_text(auth_overlay, R"({
        "providers": {
            "qwen-token-plan": {
                "api_key": "test-token-plan-key"
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    REQUIRE(manager.persist_login_profile("qwen", &error));
    REQUIRE(error.empty());

    const auto& config = manager.get_config();
    REQUIRE(config.default_provider == "qwen-token-plan");
    REQUIRE(config.default_model_selection == "manual");
    REQUIRE(config.providers.contains("qwen-token-plan"));
    REQUIRE(config.providers.at("qwen-token-plan").auth_type.empty());
    REQUIRE(config.providers.at("qwen-token-plan").model.empty());
    REQUIRE(config.providers.at("qwen-token-plan").api_key == "test-token-plan-key");

    manager.load(project_dir);
    const auto& reloaded = manager.get_config();
    REQUIRE(reloaded.default_provider == "qwen-token-plan");
    REQUIRE(reloaded.providers.at("qwen-token-plan").model.empty());
    REQUIRE(reloaded.providers.at("qwen-token-plan").api_key == "test-token-plan-key");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists /model defaults across reloads", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_model_persist");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-4o" },
            "grok": { "type": "grok", "model": "grok-code-fast-1" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    REQUIRE(manager.persist_model_defaults("grok", "manual", {}, &error));
    REQUIRE(error.empty());

    manager.load(project_dir);
    const auto& config = manager.get_config();
    REQUIRE(config.default_provider == "grok");
    REQUIRE(config.default_model_selection == "manual");
    REQUIRE(fs::exists(xdg_home / "filo" / "model_defaults.json"));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists login profiles and selects the authenticated provider",
          "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_login_profile");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path auth_overlay = xdg_home / "filo" / "auth_defaults.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "grok",
        "default_model_selection": "manual",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-5.4" },
            "claude": { "type": "claude", "model": "claude-sonnet-4-6" },
            "grok":   { "type": "grok",   "model": "grok-code-fast-1" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    REQUIRE(manager.persist_login_profile("claude", &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().default_provider == "claude");
    REQUIRE(manager.get_config().default_model_selection == "manual");
    REQUIRE(manager.get_config().providers.at("claude").auth_type == "oauth_claude");

    REQUIRE(manager.persist_login_profile("openai", &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().default_provider == "openai");
    REQUIRE(manager.get_config().default_model_selection == "manual");
    REQUIRE(manager.get_config().providers.at("openai").auth_type == "oauth_openai_pkce");
    REQUIRE(manager.get_config().providers.at("openai").model == "gpt-5.4");

    REQUIRE(manager.persist_login_profile("openai-pkce", &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().providers.at("openai").auth_type == "oauth_openai_pkce");
    REQUIRE(manager.get_config().providers.at("openai").model == "gpt-5.4");

    REQUIRE(manager.persist_login_profile("x.ai", &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().default_provider == "grok");
    REQUIRE(manager.get_config().providers.at("grok").auth_type == "oauth_xai");
    REQUIRE(manager.get_config().providers.at("grok").model == "grok-build");

    write_text(auth_overlay, R"({
        "default_provider": "zai",
        "default_model_selection": "manual",
        "providers": {
            "zai": {
                "model": "glm-5.1",
                "api_key": "test-zai-key"
            },
            "zai-coding": {
                "model": "glm-5.2",
                "api_key": "test-zai-key"
            }
        }
    })");

    REQUIRE(manager.persist_login_profile("zai", &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().default_provider == "zai");
    REQUIRE(manager.get_config().providers.at("zai").model == "glm-5.1");
    REQUIRE(manager.get_config().providers.at("zai").api_key == "test-zai-key");
    REQUIRE(manager.get_config().providers.at("zai-coding").model == "glm-5.2");
    REQUIRE(manager.get_config().providers.at("zai-coding").api_key == "test-zai-key");
    REQUIRE(read_text(auth_overlay).find(R"("api_key":"test-zai-key")") != std::string::npos);

    manager.load(project_dir);
    const auto& reloaded = manager.get_config();
    REQUIRE(fs::exists(auth_overlay));
    REQUIRE(reloaded.default_provider == "zai");
    REQUIRE(reloaded.providers.at("zai").api_key == "test-zai-key");
    REQUIRE(reloaded.providers.at("zai-coding").api_key == "test-zai-key");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists local provider overlay using api_type=llamacpp",
          "[config][llamacpp]") {
    const fs::path sandbox = make_temp_dir("filo_config_local_persist");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path model_defaults = xdg_home / "filo" / "model_defaults.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-4o" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    REQUIRE(manager.persist_local_provider("/models/local.gguf", "qwen-local", &error));
    REQUIRE(error.empty());

    REQUIRE(fs::exists(model_defaults));
    const std::string overlay = read_text(model_defaults);
    REQUIRE(overlay.find("\"api_type\": \"llamacpp\"") != std::string::npos);

    manager.load(project_dir);
    const auto& config = manager.get_config();
    REQUIRE(config.default_provider == "local");
    REQUIRE(config.providers.contains("local"));

    const auto& local_provider = config.providers.at("local");
    REQUIRE(local_provider.api_type == core::config::ApiType::LlamaCppLocal);
    REQUIRE(local_provider.model == "qwen-local");
    REQUIRE(local_provider.local.has_value());
    REQUIRE(local_provider.local->model_path == "/models/local.gguf");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists model overlay preferences over project-level defaults",
          "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_model_overlay_priority");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path local_config = project_dir / ".filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-4o" },
            "grok": { "type": "grok", "model": "grok-code-fast-1" }
        }
    })");

    write_text(local_config, R"({
        "default_provider": "openai",
        "default_model_selection": "router",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-4o" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    // User explicitly switches to grok/manual - this should persist
    REQUIRE(manager.persist_model_defaults("grok", "manual", {}, &error));
    REQUIRE(error.empty());

    manager.load(project_dir);
    const auto& config = manager.get_config();
    // User's model choice should take precedence over project defaults
    REQUIRE(config.default_provider == "grok");
    REQUIRE(config.default_model_selection == "manual");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists specific model flavour across reloads", "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_model_flavour_persist");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "kimi",
        "default_model_selection": "manual",
        "providers": {
            "kimi": { "api_type": "kimi", "api_key": "test-key", "model": "kimi-default" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    // Verify initial model is the default
    {
        const auto config = manager.get_config();
        REQUIRE(config.default_provider == "kimi");
        REQUIRE(config.providers.at("kimi").model == "kimi-default");
    }

    // User switches to a specific model flavour using /model kimi kimi-for-coding
    std::string error;
    REQUIRE(manager.persist_model_defaults("kimi", "manual", "kimi-for-coding", &error));
    REQUIRE(error.empty());

    // Verify in-memory config is updated
    {
        const auto config = manager.get_config();
        REQUIRE(config.providers.at("kimi").model == "kimi-for-coding");
    }

    // Simulate application restart by reloading config
    manager.load(project_dir);

    // Verify the specific model flavour is persisted
    {
        const auto config = manager.get_config();
        REQUIRE(config.default_provider == "kimi");
        REQUIRE(config.default_model_selection == "manual");
        REQUIRE(config.providers.at("kimi").model == "kimi-for-coding");
    }

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists Claude Fable as a Claude model selection",
          "[config]") {
    const fs::path sandbox = make_temp_dir("filo_config_claude_fable_model");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "claude",
        "default_model_selection": "manual",
        "providers": {
            "claude": { "model": "claude-sonnet-5", "auth_type": "oauth_claude" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    {
        const auto config = manager.get_config();
        REQUIRE(config.default_provider == "claude");
        REQUIRE(config.default_model_selection == "manual");
        REQUIRE(config.providers.at("claude").model == "claude-sonnet-5");
        REQUIRE(config.providers.at("claude").auth_type == "oauth_claude");
        REQUIRE_FALSE(config.providers.contains("claude-fable"));
    }

    std::string error;
    REQUIRE(manager.persist_model_defaults("claude", "manual", "claude-fable-5", &error));
    REQUIRE(error.empty());

    manager.load(project_dir);
    {
        const auto config = manager.get_config();
        REQUIRE(config.default_provider == "claude");
        REQUIRE(config.providers.at("claude").model == "claude-fable-5");
        REQUIRE_FALSE(config.providers.contains("claude-fable"));
    }

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists managed settings and reset removes empty settings file",
          "[config]") {
    const fs::path sandbox = make_temp_dir("filo_managed_settings_persist");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path user_settings = xdg_home / "filo" / "settings.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "default_provider": "openai",
        "default_model_selection": "manual",
        "providers": {
            "openai": { "type": "openai", "model": "gpt-4o" },
            "grok": { "type": "grok", "model": "grok-code-fast-1" },
            "mistral": { "type": "mistral", "model": "devstral-small-latest" }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    REQUIRE(manager.persist_managed_setting(core::config::SettingsScope::User,
                                            core::config::ManagedSettingKey::UiFooter,
                                            std::string("hide"),
                                            project_dir,
                                            &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().ui_footer == "hide");
    REQUIRE(fs::exists(user_settings));

    REQUIRE(manager.persist_managed_setting(core::config::SettingsScope::User,
                                            core::config::ManagedSettingKey::ContextCompression,
                                            std::string("ultra"),
                                            project_dir,
                                            &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().context_compression == "ultra");
    const std::string persisted_settings = read_text(user_settings);
    REQUIRE(persisted_settings.find("\"context_compression\"") != std::string::npos);
    REQUIRE(persisted_settings.find("\"ultra\"") != std::string::npos);

    REQUIRE(manager.persist_managed_setting(core::config::SettingsScope::User,
                                            core::config::ManagedSettingKey::UiFooter,
                                            std::nullopt,
                                            project_dir,
                                            &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().ui_footer == "show");
    REQUIRE(fs::exists(user_settings));

    REQUIRE(manager.persist_managed_setting(core::config::SettingsScope::User,
                                            core::config::ManagedSettingKey::ContextCompression,
                                            std::nullopt,
                                            project_dir,
                                            &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().context_compression == "off");
    REQUIRE_FALSE(fs::exists(user_settings));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager managed settings table covers every persisted setting",
          "[config]") {
    struct ManagedSettingCase {
        core::config::ManagedSettingKey key;
        const char* json_key;
        const char* value;
    };

    static constexpr std::array kCases{
        ManagedSettingCase{
            core::config::ManagedSettingKey::DefaultMode,
            "default_mode",
            "REVIEW",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::DefaultApprovalMode,
            "default_approval_mode",
            "yolo",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::DefaultRouterPolicy,
            "default_router_policy",
            "deep",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::UiBanner,
            "ui_banner",
            "hide",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::UiFooter,
            "ui_footer",
            "hide",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::UiModelInfo,
            "ui_model_info",
            "hide",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::UiContextUsage,
            "ui_context_usage",
            "hide",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::UiTimestamps,
            "ui_timestamps",
            "hide",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::UiSpinner,
            "ui_spinner",
            "hide",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::AutoCompactThreshold,
            "auto_compact_threshold",
            "12345",
        },
        ManagedSettingCase{
            core::config::ManagedSettingKey::ContextCompression,
            "context_compression",
            "light",
        },
    };

    const fs::path sandbox = make_temp_dir("filo_managed_settings_table");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path user_settings = xdg_home / "filo" / "settings.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    std::string error;
    for (const auto& item : kCases) {
        const std::string baseline =
            effective_managed_value(manager.get_config(), item.key);

        REQUIRE(manager.persist_managed_setting(core::config::SettingsScope::User,
                                                item.key,
                                                std::string(item.value),
                                                project_dir,
                                                &error));
        REQUIRE(error.empty());
        REQUIRE(managed_overlay_value(
                    manager.get_settings_overlay(core::config::SettingsScope::User),
                    item.key)
                == std::optional<std::string>{item.value});
        REQUIRE(effective_managed_value(manager.get_config(), item.key) == item.value);

        const std::string persisted_settings = read_text(user_settings);
        REQUIRE(persisted_settings.find(std::format("\"{}\"", item.json_key))
                != std::string::npos);
        REQUIRE(persisted_settings.find(std::format("\"{}\"", item.value))
                != std::string::npos);

        REQUIRE(manager.persist_managed_setting(core::config::SettingsScope::User,
                                                item.key,
                                                std::nullopt,
                                                project_dir,
                                                &error));
        REQUIRE(error.empty());
        REQUIRE(manager.get_settings_overlay(core::config::SettingsScope::User).empty());
        REQUIRE(effective_managed_value(manager.get_config(), item.key) == baseline);
        REQUIRE_FALSE(fs::exists(user_settings));
    }

    for (const auto& item : kCases) {
        REQUIRE(manager.persist_managed_setting(core::config::SettingsScope::User,
                                                item.key,
                                                std::string(item.value),
                                                project_dir,
                                                &error));
        REQUIRE(error.empty());
    }

    const std::string complete_settings = read_text(user_settings);
    for (const auto& item : kCases) {
        REQUIRE(complete_settings.find(std::format("\"{}\"", item.json_key))
                != std::string::npos);
        REQUIRE(managed_overlay_value(
                    manager.get_settings_overlay(core::config::SettingsScope::User),
                    item.key)
                == std::optional<std::string>{item.value});
        REQUIRE(effective_managed_value(manager.get_config(), item.key) == item.value);
    }

    manager.load(project_dir);
    for (const auto& item : kCases) {
        REQUIRE(managed_overlay_value(
                    manager.get_settings_overlay(core::config::SettingsScope::User),
                    item.key)
                == std::optional<std::string>{item.value});
        REQUIRE(effective_managed_value(manager.get_config(), item.key) == item.value);
    }

    for (const auto& item : kCases) {
        REQUIRE(manager.persist_managed_setting(core::config::SettingsScope::User,
                                                item.key,
                                                std::nullopt,
                                                project_dir,
                                                &error));
        REQUIRE(error.empty());
    }
    REQUIRE(manager.get_settings_overlay(core::config::SettingsScope::User).empty());
    REQUIRE_FALSE(fs::exists(user_settings));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager parses flat top-level llama.cpp fields (README format)", "[config][llamacpp]") {
    const fs::path sandbox = make_temp_dir("filo_config_llamacpp_flat");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    // Flat format exactly as shown in the README examples.
    write_text(global_config, R"({
        "providers": {
            "local-deepseek": {
                "api_type": "llamacpp",
                "model": "deepseek-coder-6.7b-instruct",
                "model_path": "/models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
                "context_size": 8192,
                "threads": 8,
                "gpu_layers": 35,
                "temperature": 0.1,
                "top_p": 0.95
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& provider = manager.get_config().providers.at("local-deepseek");

    REQUIRE(provider.api_type == core::config::ApiType::LlamaCppLocal);
    REQUIRE(provider.model == "deepseek-coder-6.7b-instruct");
    REQUIRE(provider.local.has_value());
    const auto& local = *provider.local;
    REQUIRE(local.model_path == "/models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf");
    REQUIRE(local.temperature.has_value());
    REQUIRE(local.temperature.value() == Catch::Approx(0.1f));
    REQUIRE(local.top_p.has_value());
    REQUIRE(local.top_p.value() == Catch::Approx(0.95f));
    REQUIRE(local.llamacpp.has_value());
    const auto& ll = *local.llamacpp;
    REQUIRE(ll.context_size == 8192);
    REQUIRE(ll.threads == 8);
    REQUIRE(ll.gpu_layers == 35);

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager parses router guardrails from config files", "[config][router][guardrails]") {
    const fs::path sandbox = make_temp_dir("filo_config_router_guardrails");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";
    const fs::path local_config = project_dir / ".filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "router": {
            "enabled": true,
            "default_policy": "smart-code",
            "guardrails": {
                "max_session_cost_usd": 5.0,
                "min_requests_remaining_ratio": 0.2,
                "min_tokens_remaining_ratio": 0.2,
                "min_window_remaining_ratio": 0.2
            },
            "policies": {
                "smart-code": {
                    "strategy": "fallback",
                    "defaults": [{ "provider": "openai", "model": "gpt-4o" }]
                }
            }
        }
    })");

    // Workspace overlay replaces the guardrails object with stricter limits.
    write_text(local_config, R"({
        "router": {
            "guardrails": {
                "max_session_cost_usd": 2.5,
                "min_requests_remaining_ratio": 0.3,
                "min_tokens_remaining_ratio": 0.3,
                "min_window_remaining_ratio": 0.3,
                "enforce_on_local": true
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& cfg = manager.get_config();

    REQUIRE(cfg.router.guardrails.has_value());
    REQUIRE(cfg.router.guardrails->max_session_cost_usd == Catch::Approx(2.5));
    REQUIRE(cfg.router.guardrails->min_requests_remaining_ratio == Catch::Approx(0.3f));
    REQUIRE(cfg.router.guardrails->min_tokens_remaining_ratio == Catch::Approx(0.3f));
    REQUIRE(cfg.router.guardrails->min_window_remaining_ratio == Catch::Approx(0.3f));
    REQUIRE(cfg.router.guardrails->enforce_on_local);

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager parses router spend_limits alias", "[config][router][guardrails]") {
    const fs::path sandbox = make_temp_dir("filo_config_router_spend_limits");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path global_config = xdg_home / "filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(global_config, R"({
        "router": {
            "enabled": true,
            "default_policy": "smart-code",
            "spend_limits": {
                "max_session_cost_usd": 7.5,
                "min_tokens_remaining_ratio": 0.25
            },
            "policies": {
                "smart-code": {
                    "strategy": "fallback",
                    "defaults": [{ "provider": "openai", "model": "gpt-4o" }]
                }
            }
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& cfg = manager.get_config();

    REQUIRE(cfg.router.guardrails.has_value());
    REQUIRE(cfg.router.guardrails->max_session_cost_usd == Catch::Approx(7.5));
    REQUIRE(cfg.router.guardrails->min_tokens_remaining_ratio == Catch::Approx(0.25f));

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager parses tool policies and hooks", "[config][tools][hooks]") {
    const fs::path sandbox = make_temp_dir("filo_config_tools_hooks");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";
    const fs::path local_config = project_dir / ".filo" / "config.json";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    write_text(local_config, R"({
        "tools": {
            "*": {
                "denied_paths": ["secrets"]
            },
            "shell": {
                "allowed_commands": ["git status", "printf"],
                "trusted_urls": ["https://example.com"]
            }
        },
        "hooks": {
            "user_prompt_submit": [
                { "name": "log-submit", "command": "printf submit", "timeout_seconds": 3 }
            ],
            "pre_tool_use": [
                "printf pre-tool"
            ]
        }
    })");

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);
    const auto& config = manager.get_config();

    REQUIRE(config.tool_policies.contains("*"));
    REQUIRE(config.tool_policies.contains("shell"));
    REQUIRE(config.tool_policies.at("*").denied_paths.has_value());
    REQUIRE(config.tool_policies.at("*").denied_paths->front() == "secrets");
    REQUIRE(config.tool_policies.at("shell").allowed_commands.has_value());
    REQUIRE(config.tool_policies.at("shell").allowed_commands->size() == 2);
    REQUIRE(config.tool_policies.at("shell").trusted_urls.has_value());
    REQUIRE(config.tool_policies.at("shell").trusted_urls->front() == "https://example.com");
    REQUIRE(config.hooks.user_prompt_submit.size() == 1);
    REQUIRE(config.hooks.user_prompt_submit.front().name == "log-submit");
    REQUIRE(config.hooks.pre_tool_use.size() == 1);
    REQUIRE(config.hooks.pre_tool_use.front().command == "printf pre-tool");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists MCP server overlays", "[config][mcp]") {
    const fs::path sandbox = make_temp_dir("filo_config_mcp_overlay");
    const fs::path xdg_home = sandbox / "xdg";
    const fs::path project_dir = sandbox / "project";

    ScopedEnvVar xdg("XDG_CONFIG_HOME", xdg_home.string());

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project_dir);

    core::config::McpServerConfig server;
    server.name = "docs";
    server.transport = "http";
    server.url = "http://localhost:8123/mcp";

    std::string error;
    REQUIRE(manager.persist_mcp_server(
        server,
        core::config::SettingsScope::Workspace,
        project_dir,
        &error));
    REQUIRE(error.empty());

    const auto overlay_path = manager.get_mcp_overlay_path(
        core::config::SettingsScope::Workspace,
        project_dir);
    REQUIRE(fs::exists(overlay_path));
    REQUIRE(manager.get_config().mcp_servers.size() == 1);
    CHECK(manager.get_config().mcp_servers.front().name == "docs");

    REQUIRE(manager.remove_mcp_server(
        "docs",
        core::config::SettingsScope::Workspace,
        project_dir,
        &error));
    REQUIRE(error.empty());
    CHECK(manager.get_config().mcp_servers.empty());

    fs::remove_all(sandbox);
}
