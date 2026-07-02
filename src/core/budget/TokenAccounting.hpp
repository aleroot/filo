#pragma once

#include <cctype>
#include <cstdint>
#include <cstddef>
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
    if (model.find("grok-4")           != std::string_view::npos) return   256'000;
    if (model.find("grok-3-mini")      != std::string_view::npos) return   131'072;
    if (model.find("grok-3")           != std::string_view::npos) return   131'072;
    if (model.find("grok-2")           != std::string_view::npos) return   131'072;
    if (model.find("moonshot-v1-8k")   != std::string_view::npos) return     8'192;
    if (model.find("moonshot-v1-32k")  != std::string_view::npos) return    32'768;
    if (model.find("moonshot-v1-128k") != std::string_view::npos) return   128'000;
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
    if (model.find("claude-sonnet-5")  != std::string_view::npos) return 1'000'000;
    if (model.find("fable")            != std::string_view::npos) return 1'000'000;
    if (model == "sonnet") return 1'000'000;
    if (model.find("claude-haiku-4-5") != std::string_view::npos) return   200'000;
    if (model == "haiku") return 200'000;
    if (model.find("claude-opus-4-8")  != std::string_view::npos) return   200'000;
    if (model.find("claude-sonnet-4-6") != std::string_view::npos) return 1'000'000;
    if (model.find("claude")           != std::string_view::npos) return   200'000;
    if (model.find("gpt-5.4")          != std::string_view::npos) return   200'000;
    if (model.find("gpt-5")            != std::string_view::npos) return   200'000;
    if (model.find("gpt-4o")           != std::string_view::npos) return   128'000;
    if (model.find("gpt-4-turbo")      != std::string_view::npos) return   128'000;
    if (model.find("gpt-4")            != std::string_view::npos) return     8'192;
    if (model.find("gpt-3.5")          != std::string_view::npos) return    16'385;
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
    return 128'000;
}

struct ModelRates {
    double input_per_m  = 2.00;
    double output_per_m = 8.00;
};

[[nodiscard]] inline ModelRates rates_for_model(std::string_view model) noexcept {
    if (model.find("grok-code-fast-1") != std::string_view::npos) return { 0.20,  1.50 };
    if (model.find("grok-4-fast")      != std::string_view::npos) return { 0.20,  0.50 };
    if (model.find("grok-4.1-fast")    != std::string_view::npos) return { 0.20,  0.50 };
    if (model.find("grok-3-mini") != std::string_view::npos) return { 0.30,  0.50 };
    if (model.find("grok-3")      != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("grok-2")      != std::string_view::npos) return { 2.00, 10.00 };
    if (model.find("grok-4")      != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("claude-fable-5")   != std::string_view::npos) return {10.00, 50.00 };
    if (model.find("claude-sonnet-5")  != std::string_view::npos) return { 2.00, 10.00 };
    if (model.find("fable")            != std::string_view::npos) return {10.00, 50.00 };
    if (model == "sonnet") return { 2.00, 10.00 };
    if (model.find("claude-haiku-4-5") != std::string_view::npos) return { 1.00,  5.00 };
    if (model.find("claude-opus-4-8")  != std::string_view::npos) return { 5.00, 25.00 };
    if (model.find("claude-sonnet-4-6") != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("opus")        != std::string_view::npos) return { 5.00, 25.00 };
    if (model.find("sonnet")      != std::string_view::npos) return { 3.00, 15.00 };
    if (model.find("haiku")       != std::string_view::npos) return { 1.00,  5.00 };
    if (model.find("gpt-5.4")      != std::string_view::npos) return { 2.50, 10.00 };
    if (model.find("gpt-5")        != std::string_view::npos) return { 2.50, 10.00 };
    if (model.find("gpt-4o")      != std::string_view::npos) return { 2.50, 10.00 };
    if (model.find("gpt-4")       != std::string_view::npos) return {30.00, 60.00 };
    if (model.find("gpt-3.5")     != std::string_view::npos) return { 0.50,  1.50 };
    if (model.find("gemini-3.1")  != std::string_view::npos) return { 1.25, 10.00 };
    if (model.find("gemini-3")    != std::string_view::npos) return { 1.25, 10.00 };
    if (model.find("gemini-2.5")  != std::string_view::npos) return { 1.25,  5.00 };
    if (model.find("gemini-2.0")  != std::string_view::npos) return { 0.10,  0.40 };
    if (model.find("gemini-1.5")  != std::string_view::npos) return { 0.075, 0.30 };
    if (model.find("gemini-pro-latest") != std::string_view::npos) return { 1.25,  5.00 };
    if (model.find("gemini-flash-latest") != std::string_view::npos) return { 0.30,  2.50 };
    if (model.find("gemini-flash-lite-latest") != std::string_view::npos) return { 0.10,  0.40 };
    if (model.find("auto-gemini-3") != std::string_view::npos) return { 1.25, 10.00 };
    if (model.find("auto-gemini-2.5") != std::string_view::npos) return { 1.25,  5.00 };
    return { 2.00, 8.00 };
}

} // namespace core::budget
