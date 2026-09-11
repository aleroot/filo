#include "SteeringLoader.hpp"

#include "core/utils/StringUtils.hpp"
#include "core/workspace/SessionWorkspace.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace core::context {

namespace {

constexpr std::size_t kMaxSteeringBytesPerFile = 12 * 1024;
constexpr std::size_t kMaxSteeringBytesTotal = 48 * 1024;
constexpr std::string_view kAgentsOverrideFilename = "AGENTS.override.md";
constexpr std::string_view kAgentsFilename = "AGENTS.md";

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

void append_hierarchical_agents_files(const std::filesystem::path& start_dir,
                                      std::vector<std::filesystem::path>& files) {
    for (const auto& dir : discover_search_dirs(start_dir)) {
        for (const auto& name : {kAgentsOverrideFilename, kAgentsFilename}) {
            const auto candidate = dir / name;
            if (std::filesystem::is_regular_file(candidate)) {
                files.push_back(candidate);
                break;
            }
        }
    }
}

[[nodiscard]] std::filesystem::path find_file_case_insensitive(
    const std::filesystem::path& dir,
    std::string_view name) {
    const std::string target_lower = core::utils::str::to_lower_ascii_copy(name);
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            const std::string filename_lower =
                core::utils::str::to_lower_ascii_copy(entry.path().filename().string());
            if (filename_lower == target_lower) {
                return entry.path();
            }
        }
    }
    return {};
}

// Fully resolves symlinks so that two discovered entries pointing at the same
// on-disk file (e.g. the common `CLAUDE.md -> AGENTS.md` symlink) share an identity.
[[nodiscard]] std::filesystem::path steering_identity(const std::filesystem::path& path) {
    std::error_code ec;
    const auto resolved = std::filesystem::canonical(path, ec);
    if (!ec && !resolved.empty()) {
        return resolved;
    }
    return normalize_path(path);
}

// Drops entries that resolve to the same file, keeping the first occurrence so
// that discovery order (and therefore the canonical label) is preserved.
void dedupe_steering_files(std::vector<std::filesystem::path>& files) {
    std::vector<std::filesystem::path> unique_identities;
    unique_identities.reserve(files.size());

    std::vector<std::filesystem::path> deduped;
    deduped.reserve(files.size());

    for (const auto& file : files) {
        const auto identity = steering_identity(file);
        if (std::ranges::find(unique_identities, identity) != unique_identities.end()) {
            continue;
        }
        unique_identities.push_back(identity);
        deduped.push_back(file);
    }

    files = std::move(deduped);
}

std::vector<std::filesystem::path> discover_directory_steering_files(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> files;
    if (dir.empty() || !std::filesystem::exists(dir)) {
        return files;
    }

    append_hierarchical_agents_files(dir, files);

    for (const auto& name : {"FILO.md", "GEMINI.md", "CLAUDE.md", "SYSTEM.md",
                              "CURSOR.md", "COPILOT.md"}) {
        const auto found = find_file_case_insensitive(dir, name);
        if (!found.empty()) {
            files.push_back(found);
        }
    }

    const auto steering_dir = dir / ".filo" / "steering";
    if (std::filesystem::is_directory(steering_dir)) {
        std::vector<std::filesystem::path> steering_files;
        for (const auto& entry : std::filesystem::directory_iterator(steering_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string ext_lower =
                core::utils::str::to_lower_ascii_copy(entry.path().extension().string());
            if (ext_lower == ".md") {
                steering_files.push_back(entry.path());
            }
        }
        std::ranges::sort(steering_files);
        files.insert(files.end(), steering_files.begin(), steering_files.end());
    }

    dedupe_steering_files(files);
    return files;
}

} // namespace

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

/// Renders discovered files into a result. Shared by every mode so the byte
/// budget, labeling, and disabled-source handling cannot drift apart.
[[nodiscard]] SteeringLoadResult assemble_steering_result(
    std::vector<std::filesystem::path> files,
    const std::filesystem::path& target_root,
    const SteeringPolicy& policy,
    std::vector<std::filesystem::path> searched_roots) {
    SteeringLoadResult result;
    result.mode = policy.mode;
    result.root = target_root;
    result.searched_roots = std::move(searched_roots);

    if (files.empty()) {
        return result;
    }

    const auto label_root = discover_project_root(target_root);
    std::string block = "\n\n[Project Steering]\n";
    std::size_t bytes_remaining = kMaxSteeringBytesTotal;

    for (const auto& file : files) {
        const std::string label = relative_label(label_root, file);
        const bool disabled = policy.is_disabled(file, label);

        std::string content = read_clamped_text(file, kMaxSteeringBytesPerFile);
        if (content.empty()) {
            continue;
        }

        result.files.push_back(SteeringFile{
            .path = file,
            .label = label,
            .content = content,
            .enabled = !disabled,
            .root = target_root,
        });

        if (disabled || bytes_remaining == 0) {
            continue;
        }

        const std::size_t max_for_file = std::min(bytes_remaining, kMaxSteeringBytesPerFile);
        if (content.size() > max_for_file) {
            content = read_clamped_text(file, max_for_file);
        }

        result.source_labels.push_back(label);
        block += "Source: " + label + "\n";
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

} // namespace

std::vector<std::filesystem::path> collect_steering_roots(
    const std::filesystem::path& primary,
    const std::vector<std::filesystem::path>& additional) {
    // One ordering rule for the whole codebase: skill discovery walks the same
    // list, so "the workspace" can never mean two different orders in two
    // subsystems. Steering is the first-wins consumer of it.
    return core::workspace::ordered_roots(primary, additional);
}

SteeringLoadResult load_workspace_steering_context(
    const std::vector<std::filesystem::path>& roots,
    const SteeringPolicy& policy) {
    SteeringLoadResult none;
    none.mode = policy.mode;

    if (policy.mode == SteeringMode::None) {
        return none;
    }

    if (policy.mode == SteeringMode::CustomFile) {
        if (policy.custom_path.empty()
            || !std::filesystem::is_regular_file(policy.custom_path)) {
            return none;
        }
        const auto file = normalize_path(policy.custom_path);
        return assemble_steering_result({file}, file.parent_path(), policy, {});
    }

    if (policy.mode == SteeringMode::CustomDir) {
        if (policy.custom_path.empty()
            || !std::filesystem::is_directory(policy.custom_path)) {
            return none;
        }
        const auto target_root = normalize_path(policy.custom_path);
        return assemble_steering_result(
            discover_directory_steering_files(target_root),
            target_root,
            policy,
            {target_root});
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
        return assemble_steering_result(std::move(files), root, policy, searched);
    }

    none.searched_roots = std::move(searched);
    return none;
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
