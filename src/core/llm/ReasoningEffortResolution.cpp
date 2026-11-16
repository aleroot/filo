#include "ReasoningEffortResolution.hpp"

#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <utility>
#include <vector>

namespace core::llm {

namespace {

constexpr std::string_view kUltrathinkKeyword = "ultrathink";

// Weakest first; auto and off rank below all of them.
constexpr std::array<std::string_view, 7> kEffortLadder{
    "minimal", "low", "medium", "high", "xhigh", "max", "ultra"};

[[nodiscard]] std::optional<std::size_t> effort_rank(std::string_view effort) {
    const auto normalized = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(effort));
    if (normalized.empty() || is_reasoning_off_effort(normalized)) return 0;
    for (std::size_t i = 0; i < kEffortLadder.size(); ++i) {
        if (kEffortLadder[i] == normalized) return i + 1;
    }
    return std::nullopt;
}

[[nodiscard]] bool is_ascii_word_char(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

// Byte ranges inside backtick code: a run of N backticks opens a span that the
// next run of exactly N closes, which covers inline code and fences alike. An
// unclosed run is literal text, as in Markdown.
[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> code_spans(std::string_view text) {
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    const auto run_end = [&](std::size_t from) {
        const auto end = text.find_first_not_of('`', from);
        return end == std::string_view::npos ? text.size() : end;
    };
    std::size_t i = text.find('`');
    while (i != std::string_view::npos) {
        const std::size_t open_end = run_end(i);
        const std::size_t width = open_end - i;
        std::size_t next = open_end;
        bool closed = false;
        for (auto close = text.find('`', open_end); close != std::string_view::npos;
             close = text.find('`', next)) {
            next = run_end(close);
            if (next - close == width) {
                spans.emplace_back(i, next);
                closed = true;
                break;
            }
        }
        i = text.find('`', closed ? next : open_end);
    }
    return spans;
}

// True when [begin, end) sits inside an open/close quote or bracket pair on its
// own line.
[[nodiscard]] bool inside_pair(std::string_view text, std::size_t begin, std::size_t end,
                               std::string_view open, std::string_view close) {
    const auto line_start = text.rfind('\n', begin);
    const std::size_t from = line_start == std::string_view::npos ? 0 : line_start + 1;
    const auto line_end = text.find('\n', end);
    const std::string_view before = text.substr(from, begin - from);
    const std::string_view after = text.substr(
        end, (line_end == std::string_view::npos ? text.size() : line_end) - end);
    if (after.find(close) == std::string_view::npos) return false;
    if (open == close) {
        std::size_t count = 0;
        for (auto at = before.find(open); at != std::string_view::npos;
             at = before.find(open, at + open.size())) {
            ++count;
        }
        return count % 2 == 1;
    }
    const auto last_open = before.rfind(open);
    if (last_open == std::string_view::npos) return false;
    const auto last_close = before.rfind(close);
    return last_close == std::string_view::npos || last_close < last_open;
}

// A keyword occurrence counts only as a standalone word: not part of a longer
// word or identifier, not a path or file-name component, not quoted or
// bracketed.
[[nodiscard]] bool is_instruction_occurrence(std::string_view text, std::size_t begin,
                                             std::size_t end) {
    if (begin > 0) {
        const char before = text[begin - 1];
        if (is_ascii_word_char(before)) return false;
        static constexpr std::string_view kPathLeaders = "-/\\.@~$";
        if (kPathLeaders.find(before) != std::string_view::npos) return false;
    }
    if (end < text.size()) {
        const char after = text[end];
        if (is_ascii_word_char(after)) return false;
        static constexpr std::string_view kPathTrailers = "-/\\";
        if (kPathTrailers.find(after) != std::string_view::npos) return false;
        // `ultrathink.md` is a file; `ultrathink.` ends a sentence.
        if (after == '.' && end + 1 < text.size() && is_ascii_word_char(text[end + 1])) {
            return false;
        }
    }
    // 'ultrathink' and ‘ultrathink’ (a single quote alone is often an apostrophe).
    if (begin > 0 && end < text.size() && text[begin - 1] == '\'' && text[end] == '\'') {
        return false;
    }
    if (begin >= 3 && text.substr(begin - 3, 3) == "\xe2\x80\x98"
        && text.substr(end, 3) == "\xe2\x80\x99") {
        return false;
    }
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 6> kEnclosures{{
        {"\"", "\""},
        {"\xe2\x80\x9c", "\xe2\x80\x9d"},
        {"(", ")"},
        {"[", "]"},
        {"{", "}"},
        {"<", ">"},
    }};
    return std::ranges::none_of(kEnclosures, [&](const auto& pair) {
        return inside_pair(text, begin, end, pair.first, pair.second);
    });
}

[[nodiscard]] bool ultrathink_disabled() {
    const char* raw = std::getenv(std::string(kDisableUltrathinkEnv).c_str());
    if (raw == nullptr) return false;
    const auto value = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(raw));
    return !value.empty() && value != "0" && value != "false" && value != "no" && value != "off";
}

} // namespace

bool is_reasoning_off_effort(std::string_view effort) {
    const auto normalized = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(effort));
    return normalized == "off" || normalized == "none" || normalized == "disabled";
}

std::optional<std::string> parse_effort_selection(std::string_view requested) {
    std::string normalized;
    normalized.reserve(requested.size());
    for (const char ch : requested) {
        if (ch == '-' || ch == '_' || std::isspace(static_cast<unsigned char>(ch))) continue;
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (normalized.empty()) return std::nullopt;
    if (normalized == "auto" || normalized == "unset" || normalized == "default") {
        return std::string{};
    }
    if (is_reasoning_off_effort(normalized)) return std::string("none");
    if (normalized == "low" || normalized == "medium" || normalized == "high"
        || normalized == "xhigh" || normalized == "max" || normalized == "ultra") {
        return normalized;
    }
    return std::nullopt;
}

bool mentions_ultrathink(std::string_view prompt) {
    // A prompt that starts with `/` is a command, and one that starts with `!`
    // a shell command; neither is a prompt the keyword can raise.
    const auto first = prompt.find_first_not_of(" \t\r\n");
    if (first != std::string_view::npos && (prompt[first] == '/' || prompt[first] == '!')) {
        return false;
    }
    // ASCII lowering keeps byte offsets, so positions map back onto `prompt`.
    const std::string lowered = core::utils::str::to_lower_ascii_copy(prompt);
    auto at = lowered.find(kUltrathinkKeyword);
    if (at == std::string::npos) return false;
    const auto spans = code_spans(prompt);
    for (; at != std::string::npos; at = lowered.find(kUltrathinkKeyword, at + 1)) {
        const std::size_t end = at + kUltrathinkKeyword.size();
        const bool in_code = std::ranges::any_of(spans, [&](const auto& span) {
            return at >= span.first && end <= span.second;
        });
        if (!in_code && is_instruction_occurrence(prompt, at, end)) return true;
    }
    return false;
}

bool ultrathink_requested(std::string_view prompt) {
    return !ultrathink_disabled() && mentions_ultrathink(prompt);
}

std::optional<std::string> raised_effort(std::string_view current, std::string_view floor) {
    const auto current_rank = effort_rank(current);
    const auto floor_rank = effort_rank(floor);
    if (!current_rank || !floor_rank || *current_rank >= *floor_rank) return std::nullopt;
    return core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_view(floor));
}

ReasoningEffortResolution resolve_reasoning_effort(
    std::string_view configured,
    const ReasoningCapabilities& capabilities) {
    using Kind = ReasoningEffortResolutionKind;

    if (configured.empty()) {
        return {"provider default", Kind::ProviderDefault};
    }

    if (is_reasoning_off_effort(configured)) {
        if (capabilities.supports(ReasoningCapability::Disable)) {
            return {"off", Kind::Exact};
        }
        if (capabilities.supports(ReasoningCapability::Required)
            && capabilities.supports_effort()) {
            return {"low", Kind::RequiredMinimum};
        }
        return {"provider default", Kind::UnsupportedProviderDefault};
    }

    if (capabilities.supports(ReasoningCapability::ToggleOnly)) {
        return {"on", Kind::Mapped};
    }
    if ((configured == "high" || configured == "max" || configured == "ultra")
        && capabilities.supports(ReasoningCapability::MapsHighToXHigh)) {
        return {"xhigh", Kind::Mapped};
    }

    if (configured == "low"
        && capabilities.supports(ReasoningCapability::MapsLowToHigh)) {
        return {"high", Kind::Mapped};
    }
    if (configured == "medium"
        && capabilities.supports(ReasoningCapability::MapsMediumToHigh)) {
        return {"high", Kind::Mapped};
    }
    if (configured == "minimal"
        && capabilities.supports(ReasoningCapability::MapsMinimalToLow)) {
        return {"low", Kind::Mapped};
    }
    if (configured == "minimal"
        && capabilities.supports(ReasoningCapability::MapsMinimalToOff)) {
        return {"off", Kind::Mapped};
    }
    if (configured == "xhigh"
        && capabilities.supports(ReasoningCapability::MapsXHighToMax)) {
        return {"max", Kind::Mapped};
    }
    if (configured == "xhigh"
        && capabilities.supports(ReasoningCapability::MapsXHighToHigh)) {
        return {"high", Kind::Mapped};
    }
    // Qwen4 accepts xhigh but not the client name `max`. The 3.8 ladder uses
    // MapsHighToXHigh instead, which also rewrites `high`.
    if ((configured == "max" || configured == "ultra")
        && capabilities.supports(ReasoningCapability::XHighEffort)
        && capabilities.supports(ReasoningCapability::MapsMinimalToLow)
        && !capabilities.supports(ReasoningCapability::MapsHighToXHigh)
        && !capabilities.supports(ReasoningCapability::MaxEffort)) {
        return {"xhigh", Kind::Mapped};
    }
    if (configured == "max"
        && !capabilities.supports(ReasoningCapability::MaxEffort)) {
        return {"high", Kind::UnsupportedFallback};
    }
    if (configured == "xhigh"
        && !capabilities.supports(ReasoningCapability::XHighEffort)) {
        return {"high", Kind::UnsupportedFallback};
    }
    if (configured == "ultra"
        && !capabilities.supports(ReasoningCapability::UltraEffort)) {
        return {
            capabilities.supports(ReasoningCapability::MaxEffort) ? "max" : "high",
            Kind::UnsupportedFallback};
    }

    return {std::string(configured), Kind::Exact};
}

} // namespace core::llm
