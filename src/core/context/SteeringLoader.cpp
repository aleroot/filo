#include "SteeringLoader.hpp"

#include "core/utils/StringUtils.hpp"

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
    }
    return "default (project root)";
}

SteeringPolicy parse_steering_policy(std::string_view spec) {
    const auto trimmed = core::utils::str::trim_ascii_copy(spec);
    if (trimmed.empty() || core::utils::str::to_lower_ascii_copy(trimmed) == "default") {
        return SteeringPolicy{.mode = SteeringMode::Default};
    }
    const auto lower = core::utils::str::to_lower_ascii_copy(trimmed);
    if (lower == "none" || lower == "off" || lower == "disabled" || lower == "no" || lower == "false") {
        return SteeringPolicy{.mode = SteeringMode::None};
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

SteeringLoadResult load_project_steering_context(const std::filesystem::path& project_root,
                                                const SteeringPolicy& policy) {
    SteeringLoadResult result;
    result.mode = policy.mode;

    if (policy.mode == SteeringMode::None) {
        return result;
    }

    std::filesystem::path target_root;
    std::vector<std::filesystem::path> files;

    if (policy.mode == SteeringMode::CustomFile) {
        if (policy.custom_path.empty() || !std::filesystem::is_regular_file(policy.custom_path)) {
            return result;
        }
        files.push_back(normalize_path(policy.custom_path));
        target_root = policy.custom_path.parent_path();
    } else if (policy.mode == SteeringMode::CustomDir) {
        if (policy.custom_path.empty() || !std::filesystem::is_directory(policy.custom_path)) {
            return result;
        }
        target_root = normalize_path(policy.custom_path);
        files = discover_directory_steering_files(target_root);
    } else {
        if (project_root.empty() || !std::filesystem::exists(project_root)) {
            return result;
        }
        target_root = normalize_path(project_root);
        files = discover_directory_steering_files(target_root);
    }

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

std::string load_project_steering_block(const std::filesystem::path& project_root,
                                        const SteeringPolicy& policy) {
    return load_project_steering_context(project_root, policy).block;
}

} // namespace core::context
