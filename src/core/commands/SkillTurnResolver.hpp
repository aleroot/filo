#pragma once

#include "../agent/Agent.hpp"
#include "../config/ConfigManager.hpp"
#include "../llm/ModelRegistry.hpp"
#include "../llm/ProviderManager.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::commands {

struct SkillTurnResolution {
    core::agent::Agent::TurnCallbacks callbacks;
    std::string warning;
};

namespace detail {

inline std::string trim_copy(std::string_view value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(start, end - start + 1));
}

inline std::string lower_ascii_copy(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    for (const unsigned char ch : input) {
        output.push_back(static_cast<char>(std::tolower(ch)));
    }
    return output;
}

inline std::string join_items(const std::vector<std::string>& values) {
    std::string output;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            output += ", ";
        }
        output += values[i];
    }
    return output;
}

inline std::string provider_family_key(
    std::string_view provider_name,
    const core::config::ProviderConfig& provider_config) {
    const std::string lowered = lower_ascii_copy(trim_copy(provider_name));

    switch (provider_config.api_type) {
        case core::config::ApiType::Anthropic:
            return "anthropic";
        case core::config::ApiType::Gemini:
            return "gemini";
        case core::config::ApiType::Kimi:
            return "kimi";
        case core::config::ApiType::Ollama:
            return "ollama";
        case core::config::ApiType::DashScope:
            return "qwen";
        case core::config::ApiType::LlamaCppLocal:
            return "local";
        case core::config::ApiType::OpenAI:
            if (lowered.starts_with("grok")) return "grok";
            if (lowered.starts_with("mistral")) return "mistral";
            return "openai";
        case core::config::ApiType::Unknown:
            break;
    }

    if (lowered.starts_with("claude") || lowered.starts_with("anthropic")) return "anthropic";
    if (lowered.starts_with("grok")) return "grok";
    if (lowered.starts_with("openai")) return "openai";
    if (lowered.starts_with("gemini")) return "gemini";
    if (lowered.starts_with("kimi")) return "kimi";
    if (lowered.starts_with("ollama")) return "ollama";
    if (lowered.starts_with("qwen") || lowered.starts_with("dashscope")) return "qwen";
    if (lowered == "local" || lowered.starts_with("llama")) return "local";
    if (lowered.starts_with("mistral")) return "mistral";
    return {};
}

inline bool try_select_provider(
    std::string_view provider_name,
    std::string_view requested_model,
    std::string_view original_hint,
    SkillTurnResolution& resolution) {
    const auto& config = core::config::ConfigManager::get_instance().get_config();
    const auto provider_it = config.providers.find(std::string(provider_name));
    if (provider_it == config.providers.end()) {
        resolution.warning = std::format(
            "Skill model '{}' references unavailable provider '{}'; using the current model.",
            original_hint,
            provider_name);
        return false;
    }

    const std::string effective_model = requested_model.empty()
        ? provider_it->second.model
        : trim_copy(requested_model);
    if (effective_model.empty()) {
        resolution.warning = std::format(
            "Skill model '{}' resolved to provider '{}' without a concrete model; using the current model.",
            original_hint,
            provider_name);
        return false;
    }

    try {
        resolution.callbacks.provider_override =
            core::llm::ProviderManager::get_instance().get_provider(std::string(provider_name));
        resolution.callbacks.model_override = effective_model;
        resolution.warning.clear();
        return true;
    } catch (const std::exception& e) {
        resolution.warning = std::format(
            "Skill model '{}' could not load provider '{}': {}; using the current model.",
            original_hint,
            provider_name,
            e.what());
        return false;
    }
}

inline bool try_select_from_candidates(
    const std::vector<std::string>& candidates,
    std::string_view requested_model,
    std::string_view original_hint,
    std::optional<std::string_view> preferred_provider,
    SkillTurnResolution& resolution) {
    if (candidates.empty()) {
        return false;
    }

    if (preferred_provider.has_value()) {
        if (const auto preferred_it = std::ranges::find(candidates, std::string(*preferred_provider));
            preferred_it != candidates.end()) {
            return try_select_provider(*preferred_it, requested_model, original_hint, resolution);
        }
    }

    const auto& config = core::config::ConfigManager::get_instance().get_config();
    if (!config.default_provider.empty()) {
        if (const auto default_it = std::ranges::find(candidates, config.default_provider);
            default_it != candidates.end()) {
            return try_select_provider(*default_it, requested_model, original_hint, resolution);
        }
    }

    if (candidates.size() == 1) {
        return try_select_provider(candidates.front(), requested_model, original_hint, resolution);
    }

    resolution.warning = std::format(
        "Skill model '{}' matches multiple configured providers ({}); using the current model.",
        original_hint,
        join_items(candidates));
    return false;
}

inline SkillTurnResolution resolve_skill_turn(
    std::string_view model_hint,
    const std::vector<std::string>& allowed_tools,
    std::optional<std::string_view> preferred_provider = std::nullopt) {
    SkillTurnResolution resolution;
    resolution.callbacks.allowed_tools = allowed_tools;

    const std::string trimmed_hint = trim_copy(model_hint);
    if (trimmed_hint.empty()) {
        return resolution;
    }

    const auto& config = core::config::ConfigManager::get_instance().get_config();

    const std::size_t slash = trimmed_hint.find('/');
    if (slash != std::string::npos) {
        const std::string provider_name = trim_copy(std::string_view(trimmed_hint).substr(0, slash));
        const std::string model_name = trim_copy(std::string_view(trimmed_hint).substr(slash + 1));
        if (!provider_name.empty()) {
            if (try_select_provider(provider_name, model_name, trimmed_hint, resolution)) {
                return resolution;
            }
            return resolution;
        }
    }

    if (config.providers.contains(trimmed_hint)) {
        try_select_provider(trimmed_hint, {}, trimmed_hint, resolution);
        return resolution;
    }

    std::vector<std::string> exact_model_matches;
    for (const auto& [provider_name, provider_config] : config.providers) {
        if (provider_config.model == trimmed_hint) {
            exact_model_matches.push_back(provider_name);
        }
    }
    if (try_select_from_candidates(
            exact_model_matches,
            trimmed_hint,
            trimmed_hint,
            preferred_provider,
            resolution)) {
        return resolution;
    }
    if (!resolution.warning.empty()) {
        return resolution;
    }

    if (const auto model_info = core::llm::ModelRegistry::instance().get_info(trimmed_hint);
        model_info.has_value()) {
        std::vector<std::string> family_matches;
        for (const auto& [provider_name, provider_config] : config.providers) {
            if (provider_family_key(provider_name, provider_config) == model_info->provider) {
                family_matches.push_back(provider_name);
            }
        }

        if (try_select_from_candidates(
                family_matches,
                trimmed_hint,
                trimmed_hint,
                preferred_provider,
                resolution)) {
            return resolution;
        }
        if (!resolution.warning.empty()) {
            return resolution;
        }

        resolution.warning = std::format(
            "Skill model '{}' is not available in the configured '{}' providers; using the current model.",
            trimmed_hint,
            model_info->provider);
        return resolution;
    }

    resolution.warning = std::format(
        "Skill model '{}' is not available; using the current model.",
        trimmed_hint);
    return resolution;
}

} // namespace detail

} // namespace core::commands
