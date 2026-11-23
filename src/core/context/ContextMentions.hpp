#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include "../llm/Models.hpp"

namespace core::session {
class SessionStore;
}

namespace core::workspace {
class PathVisibility;
}

namespace core::context {

struct MentionExpansionOptions {
    std::size_t max_file_bytes = 24 * 1024;
    std::size_t max_directory_entries = 64;
    /// Resolves `#<session-id>` / `#"name"` conversation references; when
    /// null, `#` is plain text, so callers without a session store are
    /// unaffected.
    const core::session::SessionStore* session_store = nullptr;
    /// The session the prompt belongs to; referencing it is rejected.
    std::string current_session_id = {};
    std::size_t max_session_bytes = 32 * 1024;
    /// Aggregate digest budget per prompt, across distinct conversations.
    std::size_t max_total_session_bytes = 64 * 1024;
    std::size_t max_session_references = 16;
};

struct ActiveMention {
    std::size_t replace_begin = 0;
    std::size_t replace_end = 0;
    std::string raw_path = {};
    bool quoted = false;
};

struct MentionCompletion {
    std::string text;
    std::size_t cursor = 0;
};

struct ExpandedPrompt {
    std::string display_text;
    std::vector<core::llm::ContentPart> content_parts;
    // Absolute paths explicitly referenced with @. This reports user input;
    // the caller decides whether those references should change access scope.
    std::vector<std::filesystem::path> explicit_path_mentions;
};

std::optional<ActiveMention> find_active_mention(std::string_view input,
                                                 std::size_t cursor);

/// An `@` reference exactly as a sent prompt reads it.
struct MentionToken {
    std::size_t begin = 0;      ///< the '@'
    std::size_t end = 0;        ///< one past the path, closing quote included
    std::size_t token_end = 0;  ///< `end` plus sentence punctuation after the path
    std::string raw_path = {};  ///< unescaped, without quotes
    bool quoted = false;
};

/// Every `@` reference outside code, in order: the tokens expand_prompt reads.
[[nodiscard]] std::vector<MentionToken> scan_mentions(std::string_view input);

/// What sending a mention puts in front of the model. expand_prompt branches
/// on this same value, so what an editor shows is what is sent.
enum class MentionTarget : std::uint8_t {
    Prose,       ///< unquoted and names nothing (`@here`, a handle): sent verbatim
    Subagent,    ///< `@explore`, `@general` or a configured subagent: sent verbatim
    File,
    Directory,
    Image,
    Video,
    Ignored,     ///< excluded by .agentignore: the model is told so
    OutOfScope,  ///< a relative path that climbs out of the workspace
    Missing,     ///< asked for explicitly (quoted) but not on disk
};

/// True when @p target attaches something the user asked for.
[[nodiscard]] constexpr bool attaches_content(MentionTarget target) noexcept {
    switch (target) {
    case MentionTarget::File:
    case MentionTarget::Directory:
    case MentionTarget::Image:
    case MentionTarget::Video:
        return true;
    default:
        return false;
    }
}

/// Classifies @p mention against @p base_dir. Without @p visibility,
/// .agentignore is not consulted and nothing is reported Ignored.
[[nodiscard]] MentionTarget classify_mention(
    const MentionToken& mention,
    const std::filesystem::path& base_dir,
    const core::workspace::PathVisibility* visibility = nullptr);

MentionCompletion apply_mention_completion(std::string_view input,
                                           const ActiveMention& mention,
                                           std::string_view replacement);

ExpandedPrompt expand_prompt(std::string_view input,
                             const std::filesystem::path& base_dir,
                             const MentionExpansionOptions& options = {});

std::string expand_mentions(std::string_view input,
                            const std::filesystem::path& base_dir,
                            const MentionExpansionOptions& options = {});

} // namespace core::context
