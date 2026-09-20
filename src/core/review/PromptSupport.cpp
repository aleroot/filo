#include "PromptSupport.hpp"
#include "Plan.hpp"

#include <algorithm>
#include <format>

namespace core::review::prompt_detail {

void append_steering_context(std::string& prompt,
                             const CampaignInput& input,
                             const ReviewGroup* group) {
    bool appended = false;
    for (const auto& source : input.steering.sources) {
        const bool applies = group == nullptr || std::ranges::any_of(
            group->files, [&](const FileChange& file) {
                return std::ranges::find(source.applies_to, file_display_path(file))
                    != source.applies_to.end();
            });
        if (!applies) continue;

        if (!appended) {
            prompt += "\n## Active project steering\n";
            prompt += "These selected project instructions are review criteria. Use only rules that apply to the changed paths and assess them against the supplied code evidence.\n";
            appended = true;
        }
        prompt += "\nSource [";
        prompt += source.id;
        prompt += "]: ";
        prompt += source.label;
        prompt += "\nApplies to changed path(s): ";
        const auto path_is_in_group = [&](const std::string& path) {
            return group == nullptr || std::ranges::any_of(
                group->files, [&](const FileChange& file) {
                    return file_display_path(file) == path;
                });
        };
        const auto applicable_path_count = static_cast<std::size_t>(
            std::ranges::count_if(source.applies_to, path_is_in_group));
        const auto path_count = std::min<std::size_t>(applicable_path_count, 8);
        std::size_t emitted_paths = 0;
        for (const auto& path : source.applies_to) {
            if (!path_is_in_group(path) || emitted_paths >= path_count) continue;
            if (emitted_paths > 0) prompt += ", ";
            prompt += path;
            ++emitted_paths;
        }
        if (applicable_path_count > path_count) {
            prompt += std::format(", and {} more", applicable_path_count - path_count);
        }
        prompt += "\n";
        prompt += source.content;
        if (prompt.back() != '\n') prompt.push_back('\n');
    }
    if (!appended) {
        prompt += "\nNo active path-scoped project steering was loaded for this review. Apply the general review fallback.\n";
    }
    if (input.steering.truncated) {
        prompt += "\nSome selected steering files exceeded the review guidance budget and were omitted. Do not claim compliance with rules you could not inspect.\n";
    }
    prompt += "If a steering source contains a truncation marker, do not claim compliance with its unseen content.\n";
}

} // namespace core::review::prompt_detail
