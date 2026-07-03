#include "MemoryBackgroundService.hpp"

#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <ranges>
#include <string_view>
#include <thread>
#include <unordered_set>

namespace core::memory {
namespace {

[[nodiscard]] std::string strip_wrapping_punctuation(std::string text) {
    text = core::utils::str::trim_ascii_copy(text);
    while (!text.empty() && (text.front() == ':' || text.front() == '-' || text.front() == '"' || text.front() == '\'')) {
        text.erase(text.begin());
        text = core::utils::str::trim_ascii_copy(text);
    }
    while (!text.empty() && (text.back() == '"' || text.back() == '\'')) {
        text.pop_back();
        text = core::utils::str::trim_ascii_copy(text);
    }
    return text;
}

[[nodiscard]] bool looks_sensitive(std::string_view value) {
    const std::string text = core::utils::str::to_lower_ascii_copy(value);
    constexpr std::array needles{
        "api key",
        "apikey",
        "secret",
        "password",
        "token",
        "credential",
        "private key",
        "bearer ",
        "ssh-rsa",
        "-----begin",
    };
    return std::ranges::any_of(needles, [&](std::string_view needle) {
        return text.find(needle) != std::string::npos;
    });
}

[[nodiscard]] bool durable_length(std::string_view value) {
    const auto clean = core::utils::str::trim_ascii_copy(value);
    return clean.size() >= 8 && clean.size() <= 300;
}

[[nodiscard]] std::optional<std::string> after_phrase(std::string_view original,
                                                      std::string_view lowered,
                                                      std::string_view phrase) {
    const auto pos = lowered.find(phrase);
    if (pos == std::string_view::npos) return std::nullopt;
    auto result = strip_wrapping_punctuation(
        std::string(original.substr(pos + phrase.size())));
    if (!durable_length(result) || looks_sensitive(result)) return std::nullopt;
    return core::utils::str::collapse_ascii_whitespace_copy(result);
}

[[nodiscard]] std::vector<std::string> explicit_memory_candidates(std::string_view text) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(text);
    std::vector<std::string> out;
    constexpr std::array phrases{
        "remember that ",
        "please remember that ",
        "remember this: ",
        "save this in memory: ",
        "save in memory: ",
        "store this in memory: ",
        "add this to memory: ",
        "keep this in memory: ",
        "from now on ",
    };
    for (std::string_view phrase : phrases) {
        if (auto candidate = after_phrase(text, lowered, phrase); candidate.has_value()) {
            out.push_back(std::move(*candidate));
        }
    }

    constexpr std::array durable_prefixes{
        "i prefer ",
        "prefer ",
        "always ",
        "do not ",
        "don't ",
    };
    const std::string clean = core::utils::str::collapse_ascii_whitespace_copy(text);
    const std::string clean_lower = core::utils::str::to_lower_ascii_copy(clean);
    for (std::string_view prefix : durable_prefixes) {
        if (clean_lower.starts_with(prefix) && durable_length(clean) && !looks_sensitive(clean)) {
            out.push_back(clean);
            break;
        }
    }
    return out;
}

[[nodiscard]] std::string slug(std::string_view value) {
    std::string out;
    out.reserve(std::min<std::size_t>(value.size(), 48));
    bool dash = false;
    for (const unsigned char ch : value) {
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(std::tolower(ch)));
            dash = false;
        } else if (!dash && !out.empty()) {
            out.push_back('-');
            dash = true;
        }
        if (out.size() >= 48) break;
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? std::string("memory-skill") : out;
}

[[nodiscard]] bool is_skill_shaped_memory(const MemoryEntry& entry) {
    const auto text = core::utils::str::to_lower_ascii_copy(entry.content);
    return text.find("workflow") != std::string::npos
        || text.find("process") != std::string::npos
        || text.find("always ") != std::string::npos
        || text.find("run ") != std::string::npos
        || text.find("before ") != std::string::npos;
}

[[nodiscard]] float remaining_ratio(const core::llm::protocols::RateLimitInfo& info) {
    float ratio = 1.0f;
    bool known = false;
    if (info.requests_limit > 0) {
        ratio = std::min(ratio, static_cast<float>(info.requests_remaining) / info.requests_limit);
        known = true;
    }
    if (info.tokens_limit > 0) {
        ratio = std::min(ratio, static_cast<float>(info.tokens_remaining) / info.tokens_limit);
        known = true;
    }
    if (!info.usage_windows.empty()) {
        ratio = std::min(ratio, std::max(0.0f, 1.0f - info.max_window_utilization()));
        known = true;
    }
    return known ? std::clamp(ratio, 0.0f, 1.0f) : 1.0f;
}

} // namespace

MemoryBackgroundService::MemoryBackgroundService(MemoryStore store)
    : store_(std::move(store)) {}

bool MemoryBackgroundService::rate_limit_allows(
    const MemorySettings& settings,
    const core::llm::protocols::RateLimitInfo& info) {
    if (settings.min_rate_limit_remaining_percent <= 0) return true;
    if (info.is_rate_limited) return false;
    const float minimum = static_cast<float>(settings.min_rate_limit_remaining_percent) / 100.0f;
    return remaining_ratio(info) >= minimum;
}

std::vector<std::string> MemoryBackgroundService::extract_candidate_memories(
    const std::vector<core::llm::Message>& history,
    std::size_t max_recent_messages) {
    std::vector<std::string> candidates;
    std::unordered_set<std::string> seen;
    std::size_t inspected = 0;
    for (auto it = history.rbegin(); it != history.rend() && inspected < max_recent_messages; ++it) {
        if (it->role != "user") continue;
        ++inspected;
        for (auto candidate : explicit_memory_candidates(it->content)) {
            const auto fingerprint = core::utils::str::to_lower_ascii_copy(
                core::utils::str::collapse_ascii_whitespace_copy(candidate));
            if (!fingerprint.empty() && seen.insert(fingerprint).second) {
                candidates.push_back(std::move(candidate));
            }
        }
    }
    std::ranges::reverse(candidates);
    return candidates;
}

MemoryReviewResult MemoryBackgroundService::review(const MemoryReviewInput& input) const {
    auto state = store_.load();
    if (!state.settings.enabled
        || !input.thread_policy.generate_memories
        || (!state.settings.background_review && !state.settings.consolidation && !state.settings.skill_curation)) {
        return {.skipped_for_policy = true};
    }
    if (!rate_limit_allows(state.settings, input.rate_limit)) {
        return {.skipped_for_rate_limit = true};
    }

    MemoryReviewResult result;
    result.ran = true;

    if (state.settings.background_review) {
        for (const auto& candidate : extract_candidate_memories(input.history)) {
            auto stored = store_.remember(candidate, "global", {"background"}, "background_review");
            if (stored.ok) ++result.memories_stored;
        }
    }

    if (state.settings.consolidation) {
        const auto cleaned = store_.clean();
        if (cleaned.ok && cleaned.message.find("Archived ") != std::string::npos) {
            result.memories_cleaned = 1;
        }
    }

    if (state.settings.skill_curation && input.thread_policy.curate_skills) {
        result.skill_drafts_written =
            curate_skill_drafts(store_.load(), input.session_context.workspace_view().primary())
                .skill_drafts_written;
    }

    if (result.memories_stored > 0 || result.memories_cleaned > 0 || result.skill_drafts_written > 0) {
        result.message = std::format(
            "Memory background review stored {}, cleaned {}, drafted {} skill(s).",
            result.memories_stored,
            result.memories_cleaned,
            result.skill_drafts_written);
    }
    return result;
}

void MemoryBackgroundService::review_async(
    MemoryReviewInput input,
    std::function<void(MemoryReviewResult)> on_done) const {
    const auto store = store_;
    std::thread([store, input = std::move(input), on_done = std::move(on_done)]() mutable {
        auto result = MemoryBackgroundService{store}.review(input);
        if (on_done) on_done(std::move(result));
    }).detach();
}

MemoryReviewResult MemoryBackgroundService::curate_skill_drafts(
    const MemoryState& state,
    const std::filesystem::path& project_root) const {
    MemoryReviewResult result;
    if (project_root.empty()) return result;

    const auto draft_root = store_.path().parent_path() / "skill-drafts";
    std::error_code ec;
    std::filesystem::create_directories(draft_root, ec);
    if (ec) return result;

    std::size_t written = 0;
    for (const auto& entry : state.entries) {
        if (entry.archived || !is_skill_shaped_memory(entry)) continue;
        const auto dir = draft_root / slug(entry.content);
        const auto path = dir / "SKILL.md";
        if (std::filesystem::exists(path, ec)) continue;
        std::filesystem::create_directories(dir, ec);
        if (ec) continue;
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) continue;
        file << "---\n";
        file << "name: " << slug(entry.content) << "\n";
        file << "description: Memory-curated draft. Review before moving into .filo/skills or ~/.config/filo/skills.\n";
        file << "enabled: false\n";
        file << "user-invocable: false\n";
        file << "---\n\n";
        file << "# Memory-curated skill draft\n\n";
        file << "Source memory: " << entry.content << "\n\n";
        file << "Refine this draft into concrete reusable instructions before enabling it.\n";
        if (file) ++written;
    }
    result.skill_drafts_written = written;
    return result;
}

} // namespace core::memory
