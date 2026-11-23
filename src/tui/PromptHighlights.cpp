#include "PromptHighlights.hpp"

#include "TuiTheme.hpp"

#include "core/llm/ReasoningEffortResolution.hpp"

#include <algorithm>
#include <ranges>
#include <utility>

using namespace ftxui;

namespace tui {

namespace {

using core::commands::InlineCommandState;
using core::context::MentionTarget;

[[nodiscard]] std::optional<PromptToken> mention_token(const core::context::MentionToken& mention,
                                                       MentionTarget target) {
    const auto token = [&](PromptTokenKind kind, std::string_view detail) {
        return PromptToken{
            .begin = mention.begin,
            .end = mention.end,
            .kind = kind,
            .subject = mention.raw_path,
            .detail = std::string(detail),
        };
    };
    switch (target) {
    case MentionTarget::Prose:      return std::nullopt;
    case MentionTarget::Subagent:   return token(PromptTokenKind::Subagent, "subagent");
    case MentionTarget::File:       return token(PromptTokenKind::Attachment, "file");
    case MentionTarget::Directory:  return token(PromptTokenKind::Attachment, "directory");
    case MentionTarget::Image:      return token(PromptTokenKind::Attachment, "image");
    case MentionTarget::Video:      return token(PromptTokenKind::Attachment, "video");
    case MentionTarget::Ignored:    return token(PromptTokenKind::Unavailable, "ignored");
    case MentionTarget::OutOfScope: return token(PromptTokenKind::Unavailable, "outside the workspace");
    case MentionTarget::Missing:    return token(PromptTokenKind::Unavailable, "not found");
    }
    return std::nullopt;
}

[[nodiscard]] Decorator token_style(PromptTokenKind kind) {
    switch (kind) {
    case PromptTokenKind::Command:
        return color(ColorPromptCommand) | bgcolor(TintPromptCommand) | ftxui::bold;
    case PromptTokenKind::PendingCommand:
        return color(ColorPromptCommand) | underlined;
    case PromptTokenKind::SupersededCommand:
        return color(ColorTextMuted) | strikethrough;
    case PromptTokenKind::Attachment:
        return color(ColorPromptFile) | bgcolor(TintPromptFile);
    case PromptTokenKind::Subagent:
        return color(ColorPromptAgent) | bgcolor(TintPromptAgent);
    case PromptTokenKind::Conversation:
        return color(ColorPromptConversation) | bgcolor(TintPromptConversation);
    case PromptTokenKind::Unavailable:
        return color(ColorWarn) | underlinedDouble;
    case PromptTokenKind::Ultrathink:
        break;  // coloured glyph by glyph, see gradient_color
    }
    return ftxui::bold;
}

/// The colour of glyph @p index of @p count along UltrathinkGradient.
[[nodiscard]] Color gradient_color(std::size_t index, std::size_t count) {
    constexpr std::size_t kLastStop = std::size(UltrathinkGradient) - 1;
    const float position = count > 1
        ? static_cast<float>(index) * kLastStop / static_cast<float>(count - 1)
        : 0.0F;
    const auto stop = std::min(static_cast<std::size_t>(position), kLastStop - 1);
    return Color::Interpolate(position - static_cast<float>(stop),
                              UltrathinkGradient[stop],
                              UltrathinkGradient[stop + 1]);
}

} // namespace

std::vector<PromptToken> scan_prompt_tokens(std::string_view prompt,
                                            std::size_t cursor,
                                            const PromptTokenSources& sources) {
    std::vector<PromptToken> tokens;
    // The prompt as the model will read it for `ultrathink`: an applied
    // inline command is blanked, the same as it is removed on send.
    std::string sent(prompt);

    if (sources.commands != nullptr) {
        for (auto& command : core::commands::scan_inline_commands(prompt, *sources.commands)) {
            PromptTokenKind kind = PromptTokenKind::PendingCommand;
            if (command.state == InlineCommandState::Complete) {
                kind = command.superseded ? PromptTokenKind::SupersededCommand
                                          : PromptTokenKind::Command;
                std::fill(sent.begin() + static_cast<std::ptrdiff_t>(command.begin),
                          sent.begin() + static_cast<std::ptrdiff_t>(command.end), ' ');
            } else if (cursor < command.begin || cursor > command.end) {
                continue;  // prose once the cursor has left it
            }
            tokens.push_back({
                .begin = command.begin,
                .end = command.end,
                .kind = kind,
                .subject = std::move(command.name),
                .detail = std::move(command.value),
            });
        }
    }

    if (sources.classify_mention) {
        for (const auto& mention : core::context::scan_mentions(prompt)) {
            if (auto token = mention_token(mention, sources.classify_mention(mention))) {
                tokens.push_back(std::move(*token));
            }
        }
    }

    if (sources.conversation_title) {
        for (const auto& reference : core::session::scan_session_references(prompt)) {
            if (auto title = sources.conversation_title(reference.token)) {
                tokens.push_back({
                    .begin = reference.begin,
                    .end = reference.end,
                    .kind = PromptTokenKind::Conversation,
                    .subject = std::move(*title),
                    .detail = reference.token.target,
                });
            } else if (reference.token.quoted) {
                tokens.push_back({
                    .begin = reference.begin,
                    .end = reference.end,
                    .kind = PromptTokenKind::Unavailable,
                    .subject = reference.token.target,
                    .detail = "no such conversation",
                });
            }
        }
    }

    if (sources.ultrathink && core::llm::ultrathink_requested(sent)) {
        for (const auto [begin, end] : core::llm::ultrathink_spans(sent)) {
            tokens.push_back({.begin = begin, .end = end, .kind = PromptTokenKind::Ultrathink});
        }
    }

    std::ranges::stable_sort(tokens, {}, &PromptToken::begin);
    std::size_t covered = 0;
    std::erase_if(tokens, [&](const PromptToken& token) {
        if (token.begin < covered) return true;
        covered = token.end;
        return false;
    });
    return tokens;
}

std::vector<PromptInputSpan> prompt_input_spans(std::span<const PromptToken> tokens) {
    std::vector<PromptInputSpan> spans;
    spans.reserve(tokens.size());
    for (const auto& token : tokens) {
        if (token.kind != PromptTokenKind::Ultrathink) {
            spans.push_back({.begin = token.begin, .end = token.end, .style = token_style(token.kind)});
            continue;
        }
        // The keyword is ASCII, so each byte is one glyph.
        const std::size_t count = token.end - token.begin;
        for (std::size_t i = 0; i < count; ++i) {
            spans.push_back({
                .begin = token.begin + i,
                .end = token.begin + i + 1,
                .style = color(gradient_color(i, count)) | ftxui::bold,
            });
        }
    }
    return spans;
}

const std::vector<PromptToken>& PromptTokenCache::tokens(std::string_view prompt,
                                                         std::size_t cursor,
                                                         const PromptTokenSources& sources) {
    const auto now = Clock::now();
    if (built_at_.has_value() && now - *built_at_ < ttl_ && cursor == cursor_ && prompt == prompt_) {
        return tokens_;
    }
    tokens_ = scan_prompt_tokens(prompt, cursor, sources);
    prompt_.assign(prompt);
    cursor_ = cursor;
    built_at_ = now;
    return tokens_;
}

} // namespace tui
