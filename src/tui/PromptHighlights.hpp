#pragma once

#include "PromptInput.hpp"

#include "core/commands/InlineCommands.hpp"
#include "core/context/ContextMentions.hpp"
#include "core/session/SessionReference.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Prompt tokens: what a draft will do when it is sent, shown while it is typed.
//
// Each kind of token keeps its own grammar in core (inline commands, `@`
// mentions, `#` conversation references, `ultrathink`); this layer runs those
// same scanners, asks the session what each token names, and colours the
// token in place. The styles never change its width, so the cursor stays
// where the plain text puts it.
//
// A token is only highlighted when sending it does something, so prose that
// looks like a token (`@here`, `#123`, `/settings page`) stays plain.
// ---------------------------------------------------------------------------

namespace tui {

enum class PromptTokenKind : std::uint8_t {
    Command,            ///< inline command, applied on send
    PendingCommand,     ///< inline command still being typed
    SupersededCommand,  ///< a later token sets the same command
    Attachment,         ///< `@` file, directory, image or video
    Subagent,           ///< `@explore`, `@general`, a configured subagent
    Conversation,       ///< `#<id>` naming a saved conversation
    Unavailable,        ///< asked for explicitly, but nothing will be attached
    Ultrathink,         ///< raises this turn's effort
};

struct PromptToken {
    std::size_t begin = 0;
    std::size_t end = 0;
    PromptTokenKind kind = PromptTokenKind::Command;
    std::string subject = {};  ///< command name, path, conversation title, subagent
    std::string detail = {};   ///< command value, attachment kind, or why it is unavailable
};

/// What the session knows about the names in a prompt. A missing service
/// leaves its kind of token unrecognised.
struct PromptTokenSources {
    const core::commands::InlineCommandCatalog* commands = nullptr;
    std::function<core::context::MentionTarget(const core::context::MentionToken&)>
        classify_mention = {};
    /// The title of the conversation a reference names, or nullopt.
    std::function<std::optional<std::string>(const core::session::SessionReferenceToken&)>
        conversation_title = {};
    bool ultrathink = false;
};

/// Every token of @p prompt, sorted and non-overlapping. An incomplete inline
/// command is reported only while @p cursor is in it.
[[nodiscard]] std::vector<PromptToken> scan_prompt_tokens(std::string_view prompt,
                                                          std::size_t cursor,
                                                          const PromptTokenSources& sources);

/// Input styles for @p tokens, resolved against the active theme.
[[nodiscard]] std::vector<PromptInputSpan> prompt_input_spans(std::span<const PromptToken> tokens);

/// Memoizes scan_prompt_tokens for the per-frame callers. A result is reused
/// while the prompt and cursor are unchanged and it is younger than the TTL,
/// so a file created meanwhile still lights up.
class PromptTokenCache {
public:
    using Clock = std::chrono::steady_clock;

    explicit PromptTokenCache(Clock::duration ttl = std::chrono::seconds(1)) : ttl_(ttl) {}

    [[nodiscard]] const std::vector<PromptToken>& tokens(std::string_view prompt,
                                                         std::size_t cursor,
                                                         const PromptTokenSources& sources);

    /// Forget the result, e.g. after a command changed a current value.
    void invalidate() noexcept { built_at_.reset(); }

private:
    Clock::duration ttl_;
    std::optional<Clock::time_point> built_at_;
    std::string prompt_;
    std::size_t cursor_ = 0;
    std::vector<PromptToken> tokens_;
};

} // namespace tui
