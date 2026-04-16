#include "SteeringLoader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
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

} // namespace

std::string load_project_steering_block(const std::filesystem::path& project_root) {
    if (project_root.empty() || !std::filesystem::exists(project_root)) {
        return {};
    }

    const auto normalized_root = normalize_path(project_root);
    const auto label_root = discover_project_root(normalized_root);

    std::vector<std::filesystem::path> files;
    append_hierarchical_agents_files(normalized_root, files);

    const auto filo_path = normalized_root / "FILO.md";
    if (std::filesystem::is_regular_file(filo_path)) {
        files.push_back(filo_path);
    }

    const auto steering_dir = normalized_root / ".filo" / "steering";
    if (std::filesystem::is_directory(steering_dir)) {
        std::vector<std::filesystem::path> steering_files;
        for (const auto& entry : std::filesystem::directory_iterator(steering_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().extension() == ".md") {
                steering_files.push_back(entry.path());
            }
        }
        std::ranges::sort(steering_files);
        files.insert(files.end(), steering_files.begin(), steering_files.end());
    }

    if (files.empty()) {
        return {};
    }

    std::string block = "\n\n[Project Steering]\n";
    std::size_t bytes_remaining = kMaxSteeringBytesTotal;
    for (const auto& file : files) {
        if (bytes_remaining == 0) {
            break;
        }

        const std::size_t max_for_file = std::min(bytes_remaining, kMaxSteeringBytesPerFile);
        std::string content = read_clamped_text(file, max_for_file);
        if (content.empty()) {
            continue;
        }

        block += "Source: " + relative_label(label_root, file) + "\n";
        block += content;
        if (!block.empty() && block.back() != '\n') {
            block.push_back('\n');
        }
        block.push_back('\n');

        const std::size_t consumed = std::min(max_for_file, content.size());
        bytes_remaining = consumed >= bytes_remaining ? 0 : bytes_remaining - consumed;
    }

    return block;
}

} // namespace core::context
