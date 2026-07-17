#include "ModelEffort.hpp"

#include <array>

namespace core::llm {

namespace {

// ── Allocation-free ASCII case-insensitive matching ──────────────────────────

[[nodiscard]] constexpr char lower(char ch) noexcept {
    return (ch >= 'A' && ch <= 'Z') ? static_cast<char>(ch + ('a' - 'A')) : ch;
}

[[nodiscard]] constexpr bool iequals(std::string_view value, std::string_view target) noexcept {
    if (value.size() != target.size()) return false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (lower(value[i]) != lower(target[i])) return false;
    }
    return true;
}

[[nodiscard]] constexpr bool istarts_with(std::string_view value,
                                          std::string_view prefix) noexcept {
    return value.size() >= prefix.size()
        && iequals(value.substr(0, prefix.size()), prefix);
}

[[nodiscard]] constexpr bool icontains(std::string_view value,
                                       std::string_view needle) noexcept {
    if (needle.empty()) return true;
    if (value.size() < needle.size()) return false;
    for (std::size_t i = 0; i + needle.size() <= value.size(); ++i) {
        if (iequals(value.substr(i, needle.size()), needle)) return true;
    }
    return false;
}

// ── Anthropic effort families ────────────────────────────────────────────────
//
// One table instead of three hand-maintained lists: a family either takes the
// effort control or it does not, and per-level acceptance is a property of the
// family. Adding a model family is a one-line change.

struct AnthropicEffortFamily {
    std::string_view token;   ///< Substring of the model id.
    bool accepts_max;
    bool accepts_xhigh;
};

constexpr std::array<AnthropicEffortFamily, 6> kAnthropicEffortFamilies{{
    {.token = "fable",      .accepts_max = true, .accepts_xhigh = true},
    {.token = "mythos",     .accepts_max = true, .accepts_xhigh = true},
    {.token = "sonnet-5",   .accepts_max = true, .accepts_xhigh = true},
    {.token = "opus-4-8",   .accepts_max = true, .accepts_xhigh = true},
    {.token = "opus-4-7",   .accepts_max = true, .accepts_xhigh = true},
    {.token = "sonnet-4-6", .accepts_max = true, .accepts_xhigh = false},
}};

[[nodiscard]] constexpr const AnthropicEffortFamily*
find_anthropic_effort_family(std::string_view model) noexcept {
    for (const auto& family : kAnthropicEffortFamilies) {
        if (icontains(model, family.token)) return &family;
    }
    return nullptr;
}

} // namespace

bool openai_model_supports_reasoning_effort(std::string_view model) noexcept {
    // Conservative allow-list to avoid sending vendor-specific fields to
    // unrelated OpenAI-compatible providers/models.
    return istarts_with(model, "gpt-5")
        || istarts_with(model, "o1")
        || istarts_with(model, "o3")
        || istarts_with(model, "o4");
}

bool openai_model_supports_max_effort(std::string_view model) noexcept {
    return istarts_with(model, "gpt-5.6");
}

bool mistral_model_supports_reasoning_effort(std::string_view model) noexcept {
    return iequals(model, "mistral-vibe-cli-latest")
        || iequals(model, "mistral-medium-3.5")
        || iequals(model, "mistral-medium-3-5")
        || iequals(model, "mistral-medium-latest")
        || iequals(model, "mistral-small-2603")
        || iequals(model, "mistral-small-latest");
}

bool kimi_model_is_k3(std::string_view model) noexcept {
    return iequals(model, "k3") || iequals(model, "kimi-k3");
}

bool kimi_model_supports_thinking(std::string_view model) noexcept {
    return kimi_model_is_k3(model)
        || iequals(model, "kimi-for-coding")
        || iequals(model, "kimi-for-coding-highspeed")
        || istarts_with(model, "kimi-k2")
        || icontains(model, "thinking");
}

bool kimi_model_requires_thinking(std::string_view model) noexcept {
    return iequals(model, "kimi-k2.7-code")
        || iequals(model, "kimi-k2.7-code-highspeed")
        || iequals(model, "kimi-for-coding")
        || iequals(model, "kimi-for-coding-highspeed");
}

bool anthropic_model_supports_effort(std::string_view model) noexcept {
    return find_anthropic_effort_family(model) != nullptr;
}

bool anthropic_model_supports_max_effort(std::string_view model) noexcept {
    const auto* family = find_anthropic_effort_family(model);
    return family != nullptr && family->accepts_max;
}

bool anthropic_model_supports_xhigh_effort(std::string_view model) noexcept {
    const auto* family = find_anthropic_effort_family(model);
    return family != nullptr && family->accepts_xhigh;
}

bool model_supports_max_effort(std::string_view model) noexcept {
    return openai_model_supports_max_effort(model)
        || mistral_model_supports_reasoning_effort(model)
        || anthropic_model_supports_max_effort(model)
        || kimi_model_is_k3(model);
}

} // namespace core::llm
