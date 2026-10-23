#include "ConfigDocument.hpp"
#include "ConfigSource.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

namespace core::config {
namespace {
using core::utils::JsonValue;

std::string profile_name(std::string_view name) {
    return core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_view(name));
}

JsonValue canonical_document(JsonValue document) {
    if (const auto* profiles = document.find("profiles")) {
        auto normalized = JsonValue::object();
        for (const auto& [name, profile] : profiles->members()) normalized.set(profile_name(name), profile);
        document.set("profiles", std::move(normalized));
    }
    return document;
}

JsonValue settings_values(const JsonValue& settings) {
    auto values = settings;
    // settings.json writes managed numbers as strings.
    for (const std::string_view key :
         {"auto_compact_threshold", "prompt_cache_guard_tokens", "prompt_cache_notice_seconds"}) {
        if (const auto* number = settings.find(key); number && number->is_string()) {
            values.set(std::string(key), JsonValue::integer(std::stoll(std::string(number->as_string()))));
        }
    }
    if (const auto* policy = settings.find("default_router_policy")) {
        auto router = JsonValue::object();
        router.set("default_policy", *policy);
        values.set("router", std::move(router));
    }
    return values;
}

} // namespace

void ConfigDocument::validate(const core::utils::JsonValue& root) {
    using core::utils::JsonValue;
    if (!root.is_object()) throw std::runtime_error("configuration root must be an object");
    const auto check = [](const JsonValue& object, std::initializer_list<std::string_view> keys,
                           const auto& valid, std::string_view type) {
        for (const auto key : keys) {
            if (const auto* value = object.find(key); value && !valid(*value)) {
                throw std::runtime_error(std::string(key) + " must be " + std::string(type));
            }
        }
    };
    const auto string = [](const JsonValue& v) { return v.is_string(); };
    const auto object = [](const JsonValue& v) { return v.is_object(); };
    const auto boolean = [](const JsonValue& v) { return v.is_bool(); };
    const auto integer = [](const JsonValue& v) { return v.as_integer().has_value(); };
    const auto strings = [](const JsonValue& v) {
        return v.is_array() && std::ranges::all_of(v.items(), [](const auto& item) { return item.is_string(); });
    };
    check(root, {"default_provider", "default_model_selection", "default_mode", "default_approval_mode",
                 "prompt_editor", "diff_comparer", "ui_banner", "ui_footer", "ui_model_info",
                 "ui_context_usage", "ui_timestamps", "ui_spinner", "ui_reasoning", "ui_theme",
                 "context_compression", "steering_mode", "prompt_cache_keep_warm", "active_profile",
                 "default_router_policy"}, string, "a string");
    check(root, {"tool_recovery", "strict_tool_schemas"}, boolean, "a boolean");
    check(root, {"tool_output_token_limit"}, integer, "an integer");
    check(root, {"auto_compact_threshold", "prompt_cache_guard_tokens", "prompt_cache_notice_seconds"},
          [&](const JsonValue& v) {
        // settings.json writes this managed preference as a string.
        return integer(v) || (v.is_string() && !v.as_string().empty()
            && v.as_string().find_first_not_of("0123456789") == std::string_view::npos);
    }, "an integer or a numeric settings string");
    check(root, {"providers", "subagents", "tools", "tool_policies", "hooks", "router", "profiles"}, object, "an object");
    if (const auto* providers = root.find("providers")) {
        for (const auto& [name, provider] : providers->members()) {
            if (!provider.is_object()) throw std::runtime_error("provider must be an object: " + name);
            check(provider, {"api_type", "api_key", "model", "base_url", "auth_type", "reasoning_effort",
                              "wire_api", "service_tier", "prompt_cache_ttl", "model_path", "chat_template"}, string, "a string");
            check(provider, {"stream_usage", "use_mmap", "use_mlock"}, boolean, "a boolean");
            check(provider, {"thinking_budget", "context_window", "top_k", "seed", "context_size", "batch_size",
                              "threads", "threads_batch", "gpu_layers"}, integer, "an integer");
            check(provider, {"temperature", "top_p"}, [](const auto& v) { return v.is_number(); }, "a number");
            check(provider, {"llamacpp"}, object, "an object");
            if (const auto* local = provider.find("llamacpp")) {
                check(*local, {"context_size", "batch_size", "threads", "threads_batch", "gpu_layers"}, integer, "an integer");
                check(*local, {"use_mmap", "use_mlock"}, boolean, "a boolean");
            }
        }
    }
    if (const auto* servers = root.find("mcp_servers")) {
        if (!servers->is_array()) throw std::runtime_error("mcp_servers must be an array");
        std::set<std::string> names;
        for (const auto& server : servers->items()) {
            const auto* name = server.find("name");
            if (!server.is_object() || !name || !name->is_string() || name->as_string().empty()) {
                throw std::runtime_error("MCP server needs a nonempty name");
            }
            if (!names.emplace(name->as_string()).second) throw std::runtime_error("duplicate MCP server name");
            check(server, {"transport", "command", "url", "auth", "oauth_client_id", "oauth_client_secret"}, string, "a string");
            check(server, {"args", "env"}, strings, "an array of strings");
            check(server, {"request_timeout_ms", "timeout_ms"}, integer, "an integer");
            check(server, {"oauth_scopes"}, [&](const auto& v) { return string(v) || strings(v); }, "a string or string array");
            check(server, {"headers"}, [&](const auto& v) {
                return strings(v) || (v.is_object() && std::ranges::all_of(v.members(), [&](const auto& field) { return string(field.second); }));
            }, "an object of strings or a string array");
        }
    }
    for (const auto section : {"profiles", "subagents", "tools", "tool_policies"}) {
        if (const auto* values = root.find(section)) {
            for (const auto& [name, value] : values->members()) {
                if (!value.is_object()) throw std::runtime_error(std::string(section) + "." + name + " must be an object");
                if (std::string_view(section) == "profiles") {
                    validate(value);
                    check(value, {"extends", "extends_from"}, [&](const auto& v) { return string(v) || strings(v); }, "a string or string array");
                }
            }
        }
    }
}

ConfigDocument ConfigDocument::load(const ConfigSource& source, const std::filesystem::path& workspace) {
    source.verify_available();
    ConfigDocument result{{}, JsonValue::object(), JsonValue::object(), {}};
    const auto append_layers = [&](const auto& documents, JsonValue& managed_settings) {
        for (const auto file : {"config.json", "auth_defaults.json", "settings.json", "mcp_servers.json", "profile_defaults.json"}) {
            const auto found = documents.find(file);
            if (found == documents.end()) continue;
            validate(found->second);
            const auto document = canonical_document(found->second);
            result.layers.push_back(std::string_view(file) == "settings.json" ? settings_values(document) : document);
            if (std::string_view(file) == "settings.json") {
                for (const auto& [name, value] : document.members()) managed_settings.set(name, value);
            }
        }
        if (const auto models = documents.find("model_defaults.json"); models != documents.end()) {
            validate(models->second);
            result.model_defaults.push_back(models->second);
        }
    };
    const auto saved_documents = [](const auto& directory) {
        ConfigSource::Documents documents;
        for (const auto file : {"config.json", "auth_defaults.json", "settings.json", "mcp_servers.json",
                                "profile_defaults.json", "model_defaults.json"}) {
            if (auto document = ConfigSource::read_document(directory / file)) documents.emplace(file, std::move(*document));
        }
        return documents;
    };
    // An explicitly supplied default directory has already been loaded by
    // the normal fallback sequence. Only its isolated saved state is new.
    if (source.input_path() != source.fallback_directory()) append_layers(source.documents(), result.user_settings);
    append_layers(saved_documents(source.state_directory()), result.user_settings);
    append_layers(saved_documents(source.workspace_state_directory(workspace)), result.workspace_settings);
    return result;
}

} // namespace core::config
