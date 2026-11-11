#pragma once

#include "../llm/MimoModelTraits.hpp"
#include "../llm/ModelRegistry.hpp"
#include "../llm/Models.hpp"
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string_view>

namespace core::budget {

[[nodiscard]] inline bool has_1m_context_suffix(std::string_view model) noexcept {
    std::size_t end = model.size();
    while (end > 0
           && std::isspace(static_cast<unsigned char>(model[end - 1]))) {
        --end;
    }
    if (end < 4) return false;
    const std::size_t base = end - 4;
    return model[base] == '['
        && model[base + 1] == '1'
        && (model[base + 2] == 'm' || model[base + 2] == 'M')
        && model[base + 3] == ']';
}

[[nodiscard]] inline int64_t context_window_for_model(std::string_view model) noexcept {
    if (has_1m_context_suffix(model)) return 1'000'000;

    if (model.find("grok-code-fast-1") != std::string_view::npos) return   131'072;
    if (model.find("grok-4.1")         != std::string_view::npos) return 2'097'152;
    if (model.find("grok-4.7")         != std::string_view::npos) return   500'000;
    if (model.find("grok-4.6")         != std::string_view::npos) return   500'000;
    if (model.find("grok-4.5")         != std::string_view::npos) return   500'000;
    if (model.find("grok-4")           != std::string_view::npos) return   256'000;
    if (model.find("grok-3-mini")      != std::string_view::npos) return   131'072;
    if (model.find("grok-3")           != std::string_view::npos) return   131'072;
    if (model.find("grok-2")           != std::string_view::npos) return   131'072;
    if (model.find("moonshot-v1-8k")   != std::string_view::npos) return     8'192;
    if (model.find("moonshot-v1-32k")  != std::string_view::npos) return    32'768;
    if (model.find("moonshot-v1-128k") != std::string_view::npos) return   128'000;
    if (model == "k3-256k") return 262'144;
    if (model == "k3" || model.find("kimi-k3") != std::string_view::npos) return 1'048'576;
    if (model.find("kimi-k2-0711-preview") != std::string_view::npos) return 128'000;
    if (model.find("kimi-k2.6")        != std::string_view::npos) return   256'000;
    if (model.find("kimi-k2-6")        != std::string_view::npos) return   256'000;
    if (model.find("kimi-k2.5")        != std::string_view::npos) return   256'000;
    if (model.find("kimi-k2-5")        != std::string_view::npos) return   256'000;
    if (model.find("kimi-for-coding")  != std::string_view::npos) return   256'000;
    if (model.find("kimi-k2")          != std::string_view::npos) return   256'000;
    if (model.find("kimi-for-university") != std::string_view::npos) return 256'000;
    if (model.find("kimi-k1.5")        != std::string_view::npos) return   256'000;
    if (model.find("kimi-")            != std::string_view::npos) return   128'000;
    if (model.find("claude-fable-5")   != std::string_view::npos) return 1'000'000;
    if (model.find("claude-opus-5")    != std::string_view::npos) return 1'000'000;
    if (model.find("claude-sonnet-5")  != std::string_view::npos) return 1'000'000;
    if (model.find("fable")            != std::string_view::npos) return 1'000'000;
    if (model == "sonnet") return 1'000'000;
    if (model.find("claude-haiku-4-5") != std::string_view::npos) return   200'000;
    if (model == "haiku" || model.find("haiku-5-5") != std::string_view::npos
        || model.find("haiku-5.5") != std::string_view::npos) return 1'000'000;
    if (model.find("claude-opus-4-8")  != std::string_view::npos) return   200'000;
    if (model.find("claude-sonnet-4-6") != std::string_view::npos) return 1'000'000;
    if (model.find("claude")           != std::string_view::npos) return   200'000;
    if (model.find("gpt-5.6")          != std::string_view::npos) return 1'050'000;
    if (model.find("gpt-5.4")          != std::string_view::npos) return   200'000;
    if (model.find("gpt-5")            != std::string_view::npos) return   200'000;
    if (model.find("gpt-4o")           != std::string_view::npos) return   128'000;
    if (model.find("gpt-4-turbo")      != std::string_view::npos) return   128'000;
    if (model.find("gpt-4")            != std::string_view::npos) return     8'192;
    if (model.find("gpt-3.5")          != std::string_view::npos) return    16'385;
    if (model.find("mimo-")            != std::string_view::npos) return 1'048'576;
    if (model == "mistral-vibe-cli-latest"
        || model.find("mistral-medium-3.5") != std::string_view::npos
        || model.find("mistral-medium-3-5") != std::string_view::npos
        || model == "mistral-medium-latest") return 256'000;
    if (model.find("mistral-small-2603") != std::string_view::npos
        || model == "mistral-small-latest") return 256'000;
    if (model.find("mistral-large-2512") != std::string_view::npos
        || model == "mistral-large-latest") return 256'000;
    if (model.find("ministral-8b-2512")  != std::string_view::npos
        || model == "ministral-8b-latest") return 256'000;
    if (model.find("codestral-2508")     != std::string_view::npos
        || model == "codestral-latest") return 128'000;
    if (model.find("gemini-3.1")       != std::string_view::npos) return 1'048'576;
    if (model.find("gemini-3")         != std::string_view::npos) return 1'048'576;
    if (model.find("gemini-2.5")       != std::string_view::npos) return 1'048'576;
    if (model.find("gemini-2.0")       != std::string_view::npos) return 1'048'576;
    if (model.find("gemini-1.5")       != std::string_view::npos) return 2'097'152;
    if (model.find("gemini-flash-latest") != std::string_view::npos) return 1'048'576;
    if (model.find("gemini-flash-lite-latest") != std::string_view::npos) return 1'048'576;
    if (model.find("gemini-pro-latest") != std::string_view::npos) return 1'048'576;
    if (model.find("auto-gemini-3")    != std::string_view::npos) return 1'048'576;
    if (model.find("auto-gemini-2.5")  != std::string_view::npos) return 1'048'576;
    try {
        if (const auto info = core::llm::ModelRegistry::instance().lookup(model); info && info->context_window > 0) return info->context_window;
    } catch (...) {}
    return 128'000;
}

struct ModelRates {
    double input_per_m  = 2.00;
    double output_per_m = 8.00;
};

/// Registry entry whose pricing is maintained as the authoritative rate card,
/// cache-read rates included. Other vendors fall back to the table below.
[[nodiscard]] inline std::shared_ptr<const core::llm::ModelInfo> registry_priced_model(
    std::string_view model) noexcept {
    try {
        auto info = core::llm::ModelRegistry::instance().lookup(model);
        if (info && (info->provider == "anthropic" || info->provider == "mimo")) {
            return info;
        }
    } catch (...) {}
    return nullptr;
}

enum class RateMatch { Contains, Exact };

struct RateRow {
    std::string_view pattern;
    RateMatch match;
    ModelRates rates;
};

/// Rates for ids the registry does not price. The first matching row wins, so a
/// more specific pattern must come before any pattern it contains
/// (`fable-5-2` before `fable-5`, `grok-3-mini` before `grok-3`).
inline constexpr RateRow kFallbackRates[] = {
    {"grok-code-fast-1", RateMatch::Contains, { 0.20,  1.50}},
    {"grok-4-fast",      RateMatch::Contains, { 0.20,  0.50}},
    {"grok-4.1-fast",    RateMatch::Contains, { 0.20,  0.50}},
    {"grok-4.6",         RateMatch::Contains, { 2.00,  6.00}},
    {"grok-4.7",         RateMatch::Contains, { 2.00,  6.00}},
    {"grok-4.5",         RateMatch::Contains, { 2.00,  6.00}},
    {"grok-3-mini",      RateMatch::Contains, { 0.30,  0.50}},
    {"grok-3",           RateMatch::Contains, { 3.00, 15.00}},
    {"grok-2",           RateMatch::Contains, { 2.00, 10.00}},
    {"grok-4",           RateMatch::Contains, { 3.00, 15.00}},
    {"fable-5-2",        RateMatch::Contains, { 8.00, 40.00}},
    {"fable-5",          RateMatch::Contains, {10.00, 50.00}},
    {"opus-5-5",         RateMatch::Contains, { 4.00, 20.00}},
    {"opus-5.5",         RateMatch::Contains, { 4.00, 20.00}},
    {"claude-opus-5",    RateMatch::Contains, { 5.00, 25.00}},
    {"claude-sonnet-5",  RateMatch::Contains, { 2.00, 10.00}},
    // Bare Fable aliases name Fable 5.2.
    {"fable",            RateMatch::Contains, { 8.00, 40.00}},
    {"sonnet",           RateMatch::Exact,    { 2.00, 10.00}},
    {"claude-haiku-4-5", RateMatch::Contains, { 1.00,  5.00}},
    {"claude-opus-4-8",  RateMatch::Contains, { 5.00, 25.00}},
    {"claude-sonnet-4-6", RateMatch::Contains, { 3.00, 15.00}},
    {"opus",             RateMatch::Contains, { 5.00, 25.00}},
    {"sonnet",           RateMatch::Contains, { 3.00, 15.00}},
    {"haiku",            RateMatch::Contains, { 1.00,  5.00}},
    {"gpt-5.6-sol",      RateMatch::Contains, { 5.00, 30.00}},
    {"gpt-5.6",          RateMatch::Exact,    { 5.00, 30.00}},
    {"gpt-5.6-terra",    RateMatch::Contains, { 2.50, 15.00}},
    {"gpt-5.6-luna",     RateMatch::Contains, { 1.00,  6.00}},
    {"gpt-5.4",          RateMatch::Contains, { 2.50, 10.00}},
    {"gpt-5",            RateMatch::Contains, { 2.50, 10.00}},
    {"gpt-4o",           RateMatch::Contains, { 2.50, 10.00}},
    {"gpt-4",            RateMatch::Contains, {30.00, 60.00}},
    {"gpt-3.5",          RateMatch::Contains, { 0.50,  1.50}},
    {"mimo-",            RateMatch::Contains, { 0.14,  0.28}},
    {"mistral-vibe-cli-latest", RateMatch::Exact,    { 1.50, 7.50}},
    {"mistral-medium-3.5",      RateMatch::Contains, { 1.50, 7.50}},
    {"mistral-medium-3-5",      RateMatch::Contains, { 1.50, 7.50}},
    {"mistral-medium-latest",   RateMatch::Exact,    { 1.50, 7.50}},
    {"mistral-small-2603",      RateMatch::Contains, { 0.15, 0.60}},
    {"mistral-small-latest",    RateMatch::Exact,    { 0.15, 0.60}},
    {"mistral-large-2512",      RateMatch::Contains, { 0.50, 1.50}},
    {"mistral-large-latest",    RateMatch::Exact,    { 0.50, 1.50}},
    {"ministral-8b-2512",       RateMatch::Contains, { 0.15, 0.15}},
    {"ministral-8b-latest",     RateMatch::Exact,    { 0.15, 0.15}},
    {"codestral-2508",          RateMatch::Contains, { 0.30, 0.90}},
    {"codestral-latest",        RateMatch::Exact,    { 0.30, 0.90}},
    {"gemini-3.1",               RateMatch::Contains, { 1.25, 10.00}},
    {"gemini-3",                 RateMatch::Contains, { 1.25, 10.00}},
    {"gemini-2.5",               RateMatch::Contains, { 1.25,  5.00}},
    {"gemini-2.0",               RateMatch::Contains, { 0.10,  0.40}},
    {"gemini-1.5",               RateMatch::Contains, { 0.075, 0.30}},
    {"gemini-pro-latest",        RateMatch::Contains, { 1.25,  5.00}},
    {"gemini-flash-latest",      RateMatch::Contains, { 0.30,  2.50}},
    {"gemini-flash-lite-latest", RateMatch::Contains, { 0.10,  0.40}},
    {"auto-gemini-3",            RateMatch::Contains, { 1.25, 10.00}},
    {"auto-gemini-2.5",          RateMatch::Contains, { 1.25,  5.00}},
};

[[nodiscard]] inline ModelRates rates_for_model(std::string_view model) noexcept {
    if (const auto info = registry_priced_model(model)) {
        return {info->pricing.input_per_mtok, info->pricing.output_per_mtok};
    }
    using enum RateMatch;
    for (const auto& row : kFallbackRates) {
        const bool hit = row.match == Exact ? model == row.pattern
                                            : model.find(row.pattern) != std::string_view::npos;
        if (hit) return row.rates;
    }
    return { 2.00, 8.00 };
}

/// Both budget guardrails and session stats use the same cache-aware estimate.
[[nodiscard]] inline double estimate_usage_cost(const core::llm::TokenUsage& usage,
                                                std::string_view model) noexcept {
    if (const auto info = registry_priced_model(model)) return info->estimate_cost(usage);
    const auto rates = rates_for_model(model);
    return (static_cast<double>(usage.prompt_tokens) * rates.input_per_m
        + static_cast<double>(usage.completion_tokens) * rates.output_per_m) / 1'000'000.0;
}

/// Credits a plan that meters in credits rather than currency charges for one
/// call; zero for a model no such plan serves. Only the MiMo Token Plan does.
[[nodiscard]] inline double estimate_plan_credits(
    const core::llm::TokenUsage& usage,
    std::string_view model,
    std::chrono::system_clock::time_point at) noexcept {
    return core::llm::mimo_token_plan_credits(usage, model, at);
}

} // namespace core::budget
