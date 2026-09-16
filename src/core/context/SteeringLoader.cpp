#include "SteeringLoader.hpp"

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/workspace/SessionWorkspace.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace core::context {

namespace {

constexpr std::size_t kMaxSteeringBytesPerFile = 12 * 1024;
constexpr std::size_t kMaxSteeringBytesTotal = 48 * 1024;
constexpr std::string_view kMarkdownExtension = ".md";

[[nodiscard]] std::string read_clamped_text(const std::filesystem::path& path, std::size_t max_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }

    std::string content(max_bytes + 1, '\0');
    in.read(content.data(), static_cast<std::streamsize>(content.size()));
    const auto read_bytes = static_cast<std::size_t>(in.gcount());
    content.resize(std::min(read_bytes, max_bytes));
    if (read_bytes > max_bytes) {
        content += "\n\n[... truncated ...]";
    }
    return content;
}

[[nodiscard]] std::string relative_label(const std::filesystem::path& root, const std::filesystem::path& path) {
    // Prefer a purely lexical relative path so that symlinked steering files keep
    // their own name instead of being relabelled with their resolved target
    // (std::filesystem::relative canonicalizes both operands, which follows links).
    const auto lexical = path.lexically_normal().lexically_relative(root.lexically_normal());
    if (!lexical.empty() && *lexical.begin() != "..") {
        return lexical.string();
    }

    std::error_code ec;
    const auto relative = std::filesystem::relative(path, root, ec);
    if (!ec && !relative.empty()) {
        return relative.string();
    }
    return path.filename().string();
}

[[nodiscard]] std::filesystem::path normalize_path(const std::filesystem::path& path) {
    std::error_code ec;
    auto normalized = std::filesystem::weakly_canonical(path, ec);
    if (!ec) {
        return normalized.lexically_normal();
    }

    ec.clear();
    normalized = std::filesystem::absolute(path, ec);
    if (!ec) {
        return normalized.lexically_normal();
    }

    return path.lexically_normal();
}

[[nodiscard]] std::filesystem::path discover_project_root(
    const std::filesystem::path& start_dir) {
    auto current = normalize_path(start_dir);
    const auto filesystem_root = current.root_path();

    while (true) {
        if (std::filesystem::exists(current / ".git")) {
            return current;
        }

        if (current == filesystem_root || current.filename().empty()) {
            break;
        }

        const auto parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }

    return normalize_path(start_dir);
}

[[nodiscard]] std::vector<std::filesystem::path> discover_search_dirs(
    const std::filesystem::path& start_dir) {
    if (start_dir.empty() || !std::filesystem::exists(start_dir)) {
        return {};
    }

    const auto normalized_start = normalize_path(start_dir);
    const auto project_root = discover_project_root(normalized_start);

    if (project_root == normalized_start) {
        return {normalized_start};
    }

    std::vector<std::filesystem::path> dirs;
    auto cursor = normalized_start;
    while (true) {
        dirs.push_back(cursor);
        if (cursor == project_root) {
            break;
        }

        const auto parent = cursor.parent_path();
        if (parent == cursor || parent.empty()) {
            break;
        }
        cursor = parent;
    }

    std::ranges::reverse(dirs);
    return dirs;
}

/// Every hierarchical steering file present in each directory from the project
/// root down to @p start_dir.
///
/// Deliberately does NOT apply the override rule. "What is on disk" and "what the
/// loader reads" are different questions with different consumers, and collapsing
/// them is what let a shadowed AGENTS.md be neither loaded nor blocked.
void append_hierarchical_steering_files(const std::filesystem::path& start_dir,
                                        std::vector<std::filesystem::path>& files) {
    for (const auto& dir : discover_search_dirs(start_dir)) {
        for (const auto& name : hierarchical_steering_names()) {
            const auto candidate = dir / name;
            if (std::filesystem::is_regular_file(candidate)) {
                files.push_back(candidate);
            }
        }
    }
}

/// Rank of @p filename within hierarchical_steering_names(), or nullopt when it
/// is not a hierarchical name. Lower is stronger.
[[nodiscard]] std::optional<std::size_t> hierarchical_priority(
    std::string_view filename) noexcept {
    const auto names = hierarchical_steering_names();
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (core::utils::ascii::iequals(filename, names[index])) {
            return index;
        }
    }
    return std::nullopt;
}

/// The override rule, as a pure filter over an already-collected list: within one
/// directory the strongest present hierarchical name wins and its siblings are
/// shadowed, i.e. not read.
///
/// Only the loader applies this. A shadowed file still exists on disk, so the
/// guard must still see it — otherwise `AGENTS.override.md` sitting next to
/// `AGENTS.md` would silently put AGENTS.md outside every policy, including
/// `none`.
[[nodiscard]] std::vector<std::filesystem::path> apply_hierarchical_override(
    const std::vector<std::filesystem::path>& files) {
    std::vector<std::filesystem::path> kept;
    kept.reserve(files.size());
    for (const auto& file : files) {
        const auto priority = hierarchical_priority(file.filename().string());
        const bool shadowed = priority.has_value() && std::ranges::any_of(
            files, [&](const std::filesystem::path& other) {
                if (other.parent_path() != file.parent_path()) {
                    return false;
                }
                const auto stronger = hierarchical_priority(other.filename().string());
                return stronger.has_value() && *stronger < *priority;
            });
        if (!shadowed) {
            kept.push_back(file);
        }
    }
    return kept;
}

/// Single pass over @p dir collecting every regular file's lowercased name, so
/// the root-level lookup costs one directory scan instead of one per candidate.
[[nodiscard]] std::vector<std::pair<std::string, std::filesystem::path>>
list_regular_files_by_lower_name(const std::filesystem::path& dir) {
    std::vector<std::pair<std::string, std::filesystem::path>> entries;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        std::error_code status_ec;
        if (!entry.is_regular_file(status_ec) || status_ec) {
            continue;
        }
        entries.emplace_back(
            core::utils::str::to_lower_ascii_copy(entry.path().filename().string()),
            entry.path());
    }
    return entries;
}

[[nodiscard]] std::vector<std::filesystem::path> find_root_steering_files(
    const std::filesystem::path& dir) {
    const auto present = list_regular_files_by_lower_name(dir);

    std::vector<std::filesystem::path> files;
    for (const auto name : root_steering_names()) {
        const auto target = core::utils::str::to_lower_ascii_copy(name);
        const auto found = std::ranges::find_if(
            present, [&](const auto& entry) { return entry.first == target; });
        if (found != present.end()) {
            files.push_back(found->second);
        }
    }
    return files;
}

[[nodiscard]] std::vector<std::filesystem::path> find_steering_directory_files(
    const std::filesystem::path& dir) {
    // Derived from the one constant that names the directory, so this lookup and
    // is_steering_directory() cannot disagree about where steering lives.
    const auto steering_dir = dir / std::filesystem::path(std::string(kSteeringDirectoryName));
    std::error_code ec;
    if (!std::filesystem::is_directory(steering_dir, ec) || ec) {
        return {};
    }

    std::vector<std::filesystem::path> steering_files;
    for (const auto& [_, path] : list_regular_files_by_lower_name(steering_dir)) {
        const auto extension = core::utils::str::to_lower_ascii_copy(path.extension().string());
        if (extension == kMarkdownExtension) {
            steering_files.push_back(path);
        }
    }
    std::ranges::sort(steering_files);
    return steering_files;
}

// Drops entries that resolve to the same file, keeping the first occurrence so
// that discovery order (and therefore the canonical label) is preserved.
void dedupe_steering_files(std::vector<std::filesystem::path>& files) {
    std::vector<std::filesystem::path> unique_identities;
    unique_identities.reserve(files.size());

    std::vector<std::filesystem::path> deduped;
    deduped.reserve(files.size());

    for (const auto& file : files) {
        const auto identity = steering_file_identity(file);
        if (std::ranges::find(unique_identities, identity) != unique_identities.end()) {
            continue;
        }
        unique_identities.push_back(identity);
        deduped.push_back(file);
    }

    files = std::move(deduped);
}

/// Every steering file physically present under @p dir, in load order and
/// de-duplicated by on-disk identity. No selection rule applied: this is "what
/// exists", which is the set a policy has to be measured against.
[[nodiscard]] std::vector<std::filesystem::path> enumerate_directory_steering_files(
    const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> files;
    if (dir.empty() || !std::filesystem::exists(dir)) {
        return files;
    }

    append_hierarchical_steering_files(dir, files);

    for (const auto& found : find_root_steering_files(dir)) {
        files.push_back(found);
    }

    for (const auto& found : find_steering_directory_files(dir)) {
        files.push_back(found);
    }

    dedupe_steering_files(files);
    return files;
}

/// The loader's view of @p dir: presence minus the override rule, i.e. what
/// would actually be read into the prompt.
[[nodiscard]] std::vector<std::filesystem::path> discover_directory_steering_files(
    const std::filesystem::path& dir) {
    return apply_hierarchical_override(enumerate_directory_steering_files(dir));
}

} // namespace

bool is_steering_filename(std::string_view filename) noexcept {
    if (filename.empty()) {
        return false;
    }
    const auto matches = [&](std::string_view name) {
        return core::utils::ascii::iequals(filename, name);
    };
    return std::ranges::any_of(hierarchical_steering_names(), matches)
        || std::ranges::any_of(root_steering_names(), matches);
}

bool is_steering_directory(const std::filesystem::path& dir) noexcept {
    // Derived from kSteeringDirectoryName rather than spelled out again: the
    // constant and this predicate describe one directory, and a second spelling
    // is a second thing to keep in sync. Still matched case-insensitively,
    // because discovery is.
    const std::filesystem::path expected{std::string(kSteeringDirectoryName)};
    return core::utils::ascii::iequals(dir.filename().string(),
                                       expected.filename().string())
        && core::utils::ascii::iequals(dir.parent_path().filename().string(),
                                       expected.parent_path().filename().string());
}

bool is_steering_file_candidate(const std::filesystem::path& path) noexcept {
    if (is_steering_filename(path.filename().string())) {
        return true;
    }
    return is_steering_directory(path.parent_path())
        && core::utils::ascii::iequals(path.extension().string(), kMarkdownExtension);
}

bool is_steering_candidate(const std::filesystem::path& path) noexcept {
    return is_steering_file_candidate(path) || is_steering_directory(path);
}

std::filesystem::path steering_file_identity(const std::filesystem::path& path) {
    std::error_code ec;
    const auto resolved = std::filesystem::canonical(path, ec);
    if (!ec && !resolved.empty()) {
        return resolved;
    }
    return normalize_path(path);
}

bool SteeringPolicy::is_disabled(const std::filesystem::path& file_path, std::string_view label) const {
    if (disabled_sources.empty()) {
        return false;
    }
    const std::string lower_label = core::utils::str::to_lower_ascii_copy(label);
    const std::string filename_lower = core::utils::str::to_lower_ascii_copy(file_path.filename().string());
    const std::string path_str = file_path.lexically_normal().string();
    const std::string path_lower = core::utils::str::to_lower_ascii_copy(path_str);

    for (const auto& disabled : disabled_sources) {
        const std::string lower_disabled = core::utils::str::to_lower_ascii_copy(
            core::utils::str::trim_ascii_copy(disabled));
        if (lower_disabled.empty()) {
            continue;
        }
        if (lower_disabled == lower_label || lower_disabled == filename_lower) {
            return true;
        }
        if (path_lower.ends_with(lower_disabled) || path_lower == lower_disabled) {
            return true;
        }
        if (lower_label.ends_with(lower_disabled)) {
            return true;
        }
    }
    return false;
}

void SteeringPolicy::disable_source(std::string_view label_or_path) {
    const std::string target = core::utils::str::trim_ascii_copy(label_or_path);
    if (target.empty()) {
        return;
    }
    const bool already = std::ranges::any_of(disabled_sources, [&](const std::string& item) {
        return core::utils::str::to_lower_ascii_copy(item) == core::utils::str::to_lower_ascii_copy(target);
    });
    if (!already) {
        disabled_sources.push_back(target);
    }
}

void SteeringPolicy::enable_source(std::string_view label_or_path) {
    const std::string target = core::utils::str::trim_ascii_copy(label_or_path);
    std::erase_if(disabled_sources, [&](const std::string& item) {
        return core::utils::str::to_lower_ascii_copy(item) == core::utils::str::to_lower_ascii_copy(target);
    });
}

std::string SteeringPolicy::format() const {
    switch (mode) {
        case SteeringMode::Default:
            return "default (project root)";
        case SteeringMode::None:
            return "none (disabled)";
        case SteeringMode::CustomFile:
            return "file (" + custom_path.string() + ")";
        case SteeringMode::CustomDir:
            return "directory (" + custom_path.string() + ")";
        case SteeringMode::Fallback:
            return "fallback (first workspace root with steering)";
    }
    return "default (project root)";
}

SteeringPolicy parse_steering_policy(std::string_view spec) {
    const auto trimmed = core::utils::str::trim_ascii_copy(spec);
    if (trimmed.empty()) {
        return SteeringPolicy{.mode = SteeringMode::Default};
    }

    const auto lower = core::utils::str::to_lower_ascii_copy(trimmed);

    // Canonical mode tokens come from the same table that feeds the settings
    // menu and the persisted `steering_mode` value, so a token can never mean
    // one thing here and another there. Matching them before the path branch is
    // what keeps "fallback" from being read as a missing relative file.
    for (const auto& option : steering_mode_options()) {
        if (lower == option.token) {
            return SteeringPolicy{.mode = option.mode};
        }
    }

    // Tolerated aliases for the two modes people spell several ways.
    if (lower == "off" || lower == "disabled" || lower == "no" || lower == "false") {
        return SteeringPolicy{.mode = SteeringMode::None};
    }
    if (lower == "chain" || lower == "fallback-chain") {
        return SteeringPolicy{.mode = SteeringMode::Fallback};
    }

    std::filesystem::path raw_path(trimmed);
    if (!trimmed.empty() && trimmed.front() == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            std::string_view rest(trimmed);
            rest.remove_prefix(1);
            if (!rest.empty() && (rest.front() == '/' || rest.front() == '\\')) {
                rest.remove_prefix(1);
            }
            raw_path = std::filesystem::path(home) / rest;
        }
    }

    const auto norm = normalize_path(raw_path);
    std::error_code ec;
    if (std::filesystem::is_directory(norm, ec)) {
        return SteeringPolicy{
            .mode = SteeringMode::CustomDir,
            .custom_path = norm,
        };
    }
    ec.clear();
    if (std::filesystem::is_regular_file(norm, ec)) {
        return SteeringPolicy{
            .mode = SteeringMode::CustomFile,
            .custom_path = norm,
        };
    }

    if (trimmed.ends_with('/') || trimmed.ends_with('\\')) {
        return SteeringPolicy{
            .mode = SteeringMode::CustomDir,
            .custom_path = norm,
        };
    }
    return SteeringPolicy{
        .mode = SteeringMode::CustomFile,
        .custom_path = norm,
    };
}

namespace {

/// Renders a selection into a result. Shared by every mode so the byte budget,
/// labeling, and disabled-source handling cannot drift apart.
[[nodiscard]] SteeringLoadResult render_steering_result(const SteeringSelection& selection) {
    SteeringLoadResult result;
    result.mode = selection.mode;
    result.root = selection.root;
    result.searched_roots = selection.searched_roots;

    if (selection.files.empty()) {
        return result;
    }

    std::string block = "\n\n[Project Steering]\n";
    std::size_t bytes_remaining = kMaxSteeringBytesTotal;

    for (const auto& file : selection.files) {
        std::string content = read_clamped_text(file.path, kMaxSteeringBytesPerFile);
        if (content.empty()) {
            continue;
        }

        SteeringFile loaded = file;
        loaded.content = content;
        result.files.push_back(std::move(loaded));

        if (!file.enabled || bytes_remaining == 0) {
            continue;
        }

        const std::size_t max_for_file = std::min(bytes_remaining, kMaxSteeringBytesPerFile);
        if (content.size() > max_for_file) {
            content = read_clamped_text(file.path, max_for_file);
        }

        result.source_labels.push_back(file.label);
        block += "Source: " + file.label + "\n";
        block += content;
        if (!block.empty() && block.back() != '\n') {
            block.push_back('\n');
        }
        block.push_back('\n');

        const std::size_t consumed = std::min(max_for_file, content.size());
        bytes_remaining = consumed >= bytes_remaining ? 0 : bytes_remaining - consumed;
    }

    if (!result.source_labels.empty()) {
        result.block = std::move(block);
    }

    return result;
}

/// Normalized, de-duplicated roots to consult, in chain order. Default mode
/// stops after the first entry because it has only ever read the primary.
[[nodiscard]] std::vector<std::filesystem::path> candidate_roots(
    const std::vector<std::filesystem::path>& roots,
    bool chain) {
    std::vector<std::filesystem::path> candidates;
    for (const auto& root : roots) {
        if (root.empty()) {
            continue;
        }
        const auto normalized = normalize_path(root);
        if (std::ranges::find(candidates, normalized) != candidates.end()) {
            continue;
        }
        candidates.push_back(normalized);
        if (!chain) {
            break;
        }
    }
    return candidates;
}

/// Which question a steering walk over the workspace roots answers.
enum class SteeringView {
    /// What the loader would read: presence minus the override rule.
    Loadable,
    /// What exists on disk, shadowed files included: the guard's candidate set.
    Present,
};

/// Shared body of discover_steering_files() and enumerate_steering_files(). The
/// walk, the identity de-duplication and the labelling are one rule; only
/// whether the override filter runs differs, and that follows from the caller's
/// question rather than being decided twice.
[[nodiscard]] std::vector<SteeringFile> gather_steering_files(
    const std::vector<std::filesystem::path>& roots, SteeringView view) {
    std::vector<SteeringFile> gathered;
    std::vector<std::filesystem::path> identities;

    for (const auto& root : candidate_roots(roots, /*chain=*/true)) {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec) || ec) {
            continue;
        }
        const auto label_root = discover_project_root(root);
        const auto present = enumerate_directory_steering_files(root);
        const auto files = view == SteeringView::Loadable
            ? apply_hierarchical_override(present)
            : present;
        for (const auto& file : files) {
            // The same on-disk file is reachable from more than one root (a
            // nested additional root, or the usual CLAUDE.md -> AGENTS.md
            // symlink). Keep the first occurrence, which is the one the loader
            // would have labelled canonically.
            const auto identity = steering_file_identity(file);
            if (std::ranges::find(identities, identity) != identities.end()) {
                continue;
            }
            identities.push_back(identity);
            gathered.push_back(SteeringFile{
                .path = file,
                .label = relative_label(label_root, file),
                .content = {},
                .enabled = true,
                .root = root,
            });
        }
    }

    return gathered;
}

} // namespace

std::vector<std::filesystem::path> collect_steering_roots(
    const std::filesystem::path& primary,
    const std::vector<std::filesystem::path>& additional) {
    // One ordering rule for the whole codebase: skill discovery walks the same
    // list, so "the workspace" can never mean two different orders in two
    // subsystems. Steering is the first-wins consumer of it.
    return core::workspace::ordered_roots(primary, additional);
}

std::vector<SteeringFile> discover_steering_files(
    const std::vector<std::filesystem::path>& roots) {
    return gather_steering_files(roots, SteeringView::Loadable);
}

std::vector<SteeringFile> enumerate_steering_files(
    const std::vector<std::filesystem::path>& roots) {
    return gather_steering_files(roots, SteeringView::Present);
}

SteeringSelection select_steering_files(
    const std::vector<std::filesystem::path>& roots,
    const SteeringPolicy& policy) {
    SteeringSelection selection;
    selection.mode = policy.mode;

    // Labeling and the disabled-source decision happen here, once, so that the
    // rendered prompt and the read-side guard classify a file identically.
    const auto adopt = [&](std::vector<std::filesystem::path> files,
                           const std::filesystem::path& target_root,
                           std::vector<std::filesystem::path> searched) {
        selection.root = target_root;
        selection.searched_roots = std::move(searched);
        const auto label_root = discover_project_root(target_root);
        for (const auto& file : files) {
            const std::string label = relative_label(label_root, file);
            selection.files.push_back(SteeringFile{
                .path = file,
                .label = label,
                .content = {},
                .enabled = !policy.is_disabled(file, label),
                .root = target_root,
            });
        }
    };

    if (policy.mode == SteeringMode::None) {
        return selection;
    }

    if (policy.mode == SteeringMode::CustomFile) {
        if (policy.custom_path.empty()
            || !std::filesystem::is_regular_file(policy.custom_path)) {
            return selection;
        }
        const auto file = normalize_path(policy.custom_path);
        adopt({file}, file.parent_path(), {});
        return selection;
    }

    if (policy.mode == SteeringMode::CustomDir) {
        if (policy.custom_path.empty()
            || !std::filesystem::is_directory(policy.custom_path)) {
            return selection;
        }
        const auto target_root = normalize_path(policy.custom_path);
        adopt(discover_directory_steering_files(target_root), target_root, {target_root});
        return selection;
    }

    const bool chain = policy.mode == SteeringMode::Fallback;
    std::vector<std::filesystem::path> searched;
    for (const auto& root : candidate_roots(roots, chain)) {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec) || ec) {
            continue;
        }
        searched.push_back(root);
        auto files = discover_directory_steering_files(root);
        // A root with nothing to offer hands off to the next one in Fallback
        // mode; in Default mode it simply *is* the (empty) answer.
        if (files.empty() && chain) {
            continue;
        }
        adopt(std::move(files), root, searched);
        return selection;
    }

    selection.searched_roots = std::move(searched);
    return selection;
}

SteeringLoadResult load_workspace_steering_context(
    const std::vector<std::filesystem::path>& roots,
    const SteeringPolicy& policy) {
    return render_steering_result(select_steering_files(roots, policy));
}

std::string load_workspace_steering_block(
    const std::vector<std::filesystem::path>& roots,
    const SteeringPolicy& policy) {
    return load_workspace_steering_context(roots, policy).block;
}

SteeringLoadResult load_project_steering_context(const std::filesystem::path& project_root,
                                                const SteeringPolicy& policy) {
    return load_workspace_steering_context({project_root}, policy);
}

std::string load_project_steering_block(const std::filesystem::path& project_root,
                                        const SteeringPolicy& policy) {
    return load_workspace_steering_context({project_root}, policy).block;
}

} // namespace core::context
