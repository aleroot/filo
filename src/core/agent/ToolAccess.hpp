#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

enum class ToolFileOperation {
    Read,
    Search,
    Write,
    ReadWrite,
};

struct ToolFileAccess {
    ToolFileOperation operation = ToolFileOperation::Read;
    std::string path;
    bool recursive = false;
};

struct ToolAccess {
    enum class Kind {
        None,
        ReadAll,
        All,
        File,
    };

    Kind kind = Kind::None;
    ToolFileAccess file;

    [[nodiscard]] static ToolAccess none() noexcept {
        return {};
    }

    [[nodiscard]] static ToolAccess all() noexcept {
        return {.kind = Kind::All};
    }

    [[nodiscard]] static ToolAccess read_all() noexcept {
        return {.kind = Kind::ReadAll};
    }

    [[nodiscard]] static ToolAccess file_access(ToolFileOperation operation,
                                                std::filesystem::path path,
                                                bool recursive = false) {
        return {
            .kind = Kind::File,
            .file = {
                .operation = operation,
                .path = path.lexically_normal().generic_string(),
                .recursive = recursive,
            },
        };
    }
};

using ToolAccessSet = std::vector<ToolAccess>;

[[nodiscard]] inline ToolAccessSet no_tool_access() {
    return {};
}

[[nodiscard]] inline ToolAccessSet all_tool_access() {
    return {ToolAccess::all()};
}

[[nodiscard]] inline ToolAccessSet read_all_tool_access() {
    return {ToolAccess::read_all()};
}

[[nodiscard]] inline bool tool_file_operation_writes(ToolFileOperation operation) noexcept {
    switch (operation) {
        case ToolFileOperation::Read:
        case ToolFileOperation::Search:
            return false;
        case ToolFileOperation::Write:
        case ToolFileOperation::ReadWrite:
            return true;
    }
    return true;
}

[[nodiscard]] inline bool tool_file_operations_conflict(ToolFileOperation left,
                                                        ToolFileOperation right) noexcept {
    return tool_file_operation_writes(left) || tool_file_operation_writes(right);
}

[[nodiscard]] inline std::string normalize_tool_access_path(std::string_view path) {
    std::string normalized(path);
    std::ranges::replace(normalized, '\\', '/');
    while (normalized.contains("//")) {
        normalized.replace(normalized.find("//"), 2, "/");
    }
    if (normalized.size() > 1 && normalized.ends_with('/')) {
        normalized.pop_back();
    }
    std::ranges::transform(normalized, normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return normalized;
}

[[nodiscard]] inline bool tool_file_accesses_overlap(const ToolFileAccess& left,
                                                     const ToolFileAccess& right) {
    const std::string left_path = normalize_tool_access_path(left.path);
    const std::string right_path = normalize_tool_access_path(right.path);
    if (left_path == right_path) {
        return true;
    }

    const std::string left_prefix = left_path.ends_with('/') ? left_path : left_path + "/";
    const std::string right_prefix = right_path.ends_with('/') ? right_path : right_path + "/";
    return (left.recursive && right_path.starts_with(left_prefix))
        || (right.recursive && left_path.starts_with(right_prefix));
}

[[nodiscard]] inline bool tool_accesses_conflict(const ToolAccess& left,
                                                 const ToolAccess& right) {
    if (left.kind == ToolAccess::Kind::None || right.kind == ToolAccess::Kind::None) {
        return false;
    }
    if (left.kind == ToolAccess::Kind::All || right.kind == ToolAccess::Kind::All) {
        return true;
    }
    if (left.kind == ToolAccess::Kind::ReadAll
        && right.kind == ToolAccess::Kind::ReadAll) {
        return false;
    }
    if (left.kind == ToolAccess::Kind::ReadAll) {
        return right.kind == ToolAccess::Kind::File
            && tool_file_operation_writes(right.file.operation);
    }
    if (right.kind == ToolAccess::Kind::ReadAll) {
        return left.kind == ToolAccess::Kind::File
            && tool_file_operation_writes(left.file.operation);
    }
    if (!tool_file_operations_conflict(left.file.operation, right.file.operation)) {
        return false;
    }
    return tool_file_accesses_overlap(left.file, right.file);
}

[[nodiscard]] inline bool tool_access_sets_conflict(const ToolAccessSet& left,
                                                    const ToolAccessSet& right) {
    return std::ranges::any_of(left, [&](const ToolAccess& left_access) {
        return std::ranges::any_of(right, [&](const ToolAccess& right_access) {
            return tool_accesses_conflict(left_access, right_access);
        });
    });
}

} // namespace core::agent
