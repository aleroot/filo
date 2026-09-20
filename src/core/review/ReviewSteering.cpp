#include "ReviewSteering.hpp"

#include "Plan.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core::review {
namespace {

constexpr std::size_t kMaxReviewGuidanceBytes = 48 * 1024;

struct ScopedDirectory {
    std::filesystem::path path;
    std::vector<std::string> changed_paths;
};

[[nodiscard]] std::string normalize_whitespace(std::string_view value) {
    std::string normalized;
    normalized.reserve(value.size());
    bool pending_space = false;
    for (const char ch : value) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            pending_space = !normalized.empty();
            continue;
        }
        if (pending_space) normalized.push_back(' ');
        normalized.push_back(ch);
        pending_space = false;
    }
    return normalized;
}

[[nodiscard]] std::string normalize_path(std::string_view value) {
    std::string normalized = std::filesystem::path(value).lexically_normal().generic_string();
    for (char& ch : normalized) {
        if (ch == '\\') ch = '/';
#if defined(_WIN32)
        else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
#endif
    }
    while (normalized.size() > 1 && normalized.back() == '/') {
        normalized.pop_back();
    }
    return normalized;
}

[[nodiscard]] bool source_applies_to_path(const ReviewSteeringSource& source,
                                          std::string_view path,
                                          std::string_view worktree_root) {
    const auto absolute_path = normalize_path(
        absolute_review_path(worktree_root, path));
    return std::ranges::any_of(source.applies_to, [&](const std::string& applies_to) {
        return absolute_path == normalize_path(
            absolute_review_path(worktree_root, applies_to));
    });
}

} // namespace

ReviewSteeringContext load_review_steering_guidance(
    const GitSnapshot& snapshot,
    const std::vector<std::filesystem::path>& workspace_roots,
    const core::context::SteeringPolicy& policy) {
    ReviewSteeringContext context;
    const auto worktree_root = std::filesystem::path(snapshot.worktree_root);
    const auto base = core::context::load_workspace_steering_context(
        workspace_roots, policy);

    std::size_t guidance_bytes = 0;
    const auto append_sources = [&](const core::context::SteeringLoadResult& loaded,
                                    std::string_view changed_path) {
        for (const auto& file : loaded.files) {
            if (!file.enabled || file.content.empty()) continue;

            const std::string identity =
                core::context::steering_file_identity(file.path).string();
            const auto found = std::ranges::find_if(
                context.sources,
                [&](const ReviewSteeringSource& source) {
                    return source.identity == identity;
                });
            if (found != context.sources.end()) {
                if (std::ranges::find(found->applies_to, changed_path)
                    == found->applies_to.end()) {
                    found->applies_to.emplace_back(changed_path);
                }
                continue;
            }

            if (guidance_bytes + file.content.size() > kMaxReviewGuidanceBytes) {
                context.truncated = true;
                continue;
            }
            guidance_bytes += file.content.size();
            context.sources.push_back({
                .id = std::format("S{}", context.sources.size() + 1),
                .label = file.label,
                .identity = identity,
                .content = file.content,
                .applies_to = {std::string(changed_path)},
            });
        }
    };

    std::vector<ScopedDirectory> scoped_directories;
    std::unordered_map<std::string, std::size_t> scoped_directory_indices;
    for (const auto& change : parse_unified_diff(snapshot.patch)) {
        std::filesystem::path changed_path(change.path);
        if (changed_path.is_relative()) {
            changed_path = worktree_root / changed_path;
        }
        changed_path = changed_path.lexically_normal();
        append_sources(base, change.path);

        auto directory = changed_path.parent_path();
        const auto key = directory.generic_string();
        const auto [slot, inserted] = scoped_directory_indices.try_emplace(
            key, scoped_directories.size());
        if (inserted) {
            scoped_directories.push_back({
                .path = std::move(directory),
            });
        }
        auto& paths = scoped_directories[slot->second].changed_paths;
        if (std::ranges::find(paths, change.path) == paths.end()) {
            paths.push_back(change.path);
        }
    }

    for (const auto& directory : scoped_directories) {
        // The shared loader selects ancestor and nested AGENTS.md instructions
        // while preserving override and workspace policy behavior.
        std::vector<std::filesystem::path> scoped_roots{directory.path};
        for (const auto& root : workspace_roots) {
            const auto normalized = root.lexically_normal();
            if (std::ranges::find(scoped_roots, normalized) == scoped_roots.end()) {
                scoped_roots.push_back(normalized);
            }
        }
        const auto loaded = core::context::load_workspace_steering_context(
            scoped_roots, policy);
        for (const auto& changed_path : directory.changed_paths) {
            append_sources(loaded, changed_path);
        }
    }

    return context;
}

std::size_t validate_steering_references(
    Report& report,
    const ReviewGroup& group,
    const ReviewSteeringContext& steering,
    std::string_view worktree_root) {
    std::size_t rejected = 0;
    for (auto& finding : report.findings) {
        std::vector<SteeringReference> verified;
        verified.reserve(finding.steering_references.size());
        for (auto& reference : finding.steering_references) {
            const auto source = std::ranges::find_if(
                steering.sources, [&](const ReviewSteeringSource& candidate) {
                    return candidate.id == reference.source_id;
                });
            const auto excerpt = normalize_whitespace(reference.rule_excerpt);
            const bool applies_to_group = source != steering.sources.end()
                && std::ranges::any_of(source->applies_to, [&](const std::string& path) {
                    return std::ranges::any_of(group.files, [&](const FileChange& file) {
                        return normalize_path(absolute_review_path(worktree_root, path))
                            == normalize_path(absolute_review_path(
                                worktree_root, file_display_path(file)));
                    });
                });
            const bool applies_to_finding = source != steering.sources.end()
                && source_applies_to_path(*source,
                                          finding.absolute_file_path,
                                          worktree_root);
            const bool excerpt_exists = source != steering.sources.end()
                && excerpt.size() >= 12
                && excerpt.size() <= 240
                && normalize_whitespace(source->content).find(excerpt)
                    != std::string::npos;
            if (!applies_to_group || !applies_to_finding || !excerpt_exists) {
                ++rejected;
                continue;
            }
            reference.source_label = source->label;
            reference.rule_excerpt = excerpt;
            verified.push_back(std::move(reference));
        }
        finding.steering_references = std::move(verified);
    }
    return rejected;
}

} // namespace core::review
