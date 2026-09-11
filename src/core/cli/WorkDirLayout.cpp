#include "WorkDirLayout.hpp"

#include "../workspace/SessionWorkspace.hpp"

#include <algorithm>
#include <system_error>

namespace core::cli {

namespace {

namespace fs = std::filesystem;

using core::workspace::SessionWorkspace;

/// Absolutize @p raw against @p base and normalize it the same way every other
/// workspace root is normalized, so roots coming from the command line compare
/// equal to roots coming from `/workspace add` or an MCP `roots` notification.
[[nodiscard]] fs::path absolutize(const fs::path& raw, const fs::path& base) {
    const fs::path candidate = raw.is_absolute() ? raw : base / raw;
    return SessionWorkspace::normalize_path(candidate);
}

} // namespace

WorkDirLayout resolve_work_dirs(const std::vector<std::string>& work_dirs,
                                const fs::path& launch_cwd) {
    WorkDirLayout layout;
    if (work_dirs.empty()) {
        return layout;
    }

    const fs::path base = launch_cwd.empty() ? fs::current_path() : launch_cwd;

    auto reject = [&](const std::string& requested, fs::path resolved, std::string reason) {
        layout.rejected.push_back(RejectedWorkDir{
            .requested = requested,
            .resolved = std::move(resolved),
            .reason = std::move(reason),
        });
    };

    // The first entry is the primary. Its existence is deliberately *not*
    // validated here: main() reports the chdir failure with the underlying
    // errno text, which is more useful than anything this layer could say.
    // An empty first entry stays empty so that chdir fails exactly as it did
    // before this resolver existed.
    if (work_dirs.front().empty()) {
        reject(work_dirs.front(), {}, "the path is empty");
    } else {
        layout.primary = absolutize(fs::path(work_dirs.front()), base);
    }

    for (std::size_t i = 1; i < work_dirs.size(); ++i) {
        const std::string& requested = work_dirs[i];
        const fs::path raw(requested);

        if (requested.empty()) {
            reject(requested, {}, "the path is empty");
            continue;
        }

        const auto resolved = absolutize(raw, base);

        // Redundant roots are dropped rather than carried: they would be
        // rendered to the model as extra workspace lines that grant nothing.
        if (!layout.primary.empty()
            && SessionWorkspace::is_subpath(layout.primary, resolved)) {
            reject(requested, resolved, resolved == layout.primary
                ? "it is the primary workspace"
                : "it is already inside the primary workspace");
            continue;
        }

        if (std::ranges::any_of(layout.additional,
                                [&](const auto& existing) { return existing == resolved; })) {
            reject(requested, resolved, "it was already added");
            continue;
        }

        std::error_code ec;
        if (!fs::exists(resolved, ec) || ec) {
            reject(requested, resolved, "it does not exist");
            continue;
        }

        layout.additional.push_back(resolved);
    }

    return layout;
}

} // namespace core::cli
