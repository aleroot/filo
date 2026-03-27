#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/config/ConfigManager.hpp"

#include <cstdlib>
#include <filesystem>
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
        "ui_footer": "hide"
    })");

    write_text(local_config, R"({
        "default_mode": "DEBUG"
    })");

    write_text(local_settings, R"({
        "default_router_policy": "deep",
        "ui_banner": "hide"
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
    REQUIRE(manager.get_settings_overlay(core::config::SettingsScope::User).ui_footer
            == std::optional<std::string>{"hide"});
    REQUIRE(manager.get_settings_overlay(core::config::SettingsScope::Workspace).default_router_policy
            == std::optional<std::string>{"deep"});

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
    REQUIRE(config.providers.contains("grok"));
    REQUIRE(config.providers.at("grok").model == "grok-code-fast-1");
    REQUIRE(config.providers.contains("grok-reasoning"));
    REQUIRE(config.providers.at("grok-reasoning").model == "grok-4.20-reasoning");
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
    REQUIRE(manager.persist_model_defaults("grok", "manual", &error));
    REQUIRE(error.empty());

    manager.load(project_dir);
    const auto& config = manager.get_config();
    REQUIRE(config.default_provider == "grok");
    REQUIRE(config.default_model_selection == "manual");
    REQUIRE(fs::exists(xdg_home / "filo" / "model_defaults.json"));

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

TEST_CASE("ConfigManager keeps project-level model defaults higher priority than persisted overlay",
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
    REQUIRE(manager.persist_model_defaults("grok", "manual", &error));
    REQUIRE(error.empty());

    manager.load(project_dir);
    const auto& config = manager.get_config();
    REQUIRE(config.default_provider == "openai");
    REQUIRE(config.default_model_selection == "router");

    fs::remove_all(sandbox);
}

TEST_CASE("ConfigManager persists managed UI settings and reset removes empty settings file",
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
                                            core::config::ManagedSettingKey::UiFooter,
                                            std::nullopt,
                                            project_dir,
                                            &error));
    REQUIRE(error.empty());
    REQUIRE(manager.get_config().ui_footer == "show");
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
