#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace core::context {

struct MentionExpansionOptions {
    std::size_t max_file_bytes = 24 * 1024;
    std::size_t max_directory_entries = 64;
};

struct ActiveMention {
    std::size_t replace_begin = 0;
    std::size_t replace_end = 0;
    std::string raw_path;
    bool quoted = false;
};

struct MentionCompletion {
    std::string text;
    std::size_t cursor = 0;
};

std::optional<ActiveMention> find_active_mention(std::string_view input,
                                                 std::size_t cursor);

MentionCompletion apply_mention_completion(std::string_view input,
                                           const ActiveMention& mention,
                                           std::string_view replacement);

std::string expand_mentions(std::string_view input,
                            const std::filesystem::path& base_dir,
                            const MentionExpansionOptions& options = {});

} // namespace core::context
