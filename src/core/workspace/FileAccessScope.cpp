#include "FileAccessScope.hpp"

#include <algorithm>
#include <system_error>
#include <utility>

namespace core::workspace {

namespace {

[[nodiscard]] std::filesystem::path normalize_root(
    const std::filesystem::path& candidate)
{
    if (candidate.empty() || !candidate.is_absolute()) {
        return {};
    }
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(candidate, ec);
    if (!ec && std::filesystem::exists(candidate, ec)) {
        return canonical.lexically_normal();
    }
    ec.clear();
    return candidate.lexically_normal();
}

void append_unique(
    std::vector<std::filesystem::path>& roots,
    const std::filesystem::path& candidate)
{
    const auto normalized = normalize_root(candidate);
    if (normalized.empty() || !normalized.is_absolute()) {
        return;
    }
    if (std::ranges::find(roots, normalized) == roots.end()) {
        roots.push_back(normalized);
    }
}

[[nodiscard]] bool is_within(
    const std::filesystem::path& root,
    const std::filesystem::path& target)
{
    auto root_it = root.begin();
    auto target_it = target.begin();
    while (root_it != root.end() && target_it != target.end()) {
        if (*root_it != *target_it) {
            return false;
        }
        ++root_it;
        ++target_it;
    }
    return root_it == root.end();
}

[[nodiscard]] bool contains(
    const std::vector<std::filesystem::path>& roots,
    const std::filesystem::path& resolved_path)
{
    if (roots.empty() || resolved_path.empty() || !resolved_path.is_absolute()) {
        return false;
    }
    // Canonicalize only if the probe exists; otherwise stay lexical so both
    // sides of the comparison stay on the same footing as the roots.
    std::error_code ec;
    const auto normalized = std::filesystem::exists(resolved_path, ec)
        ? std::filesystem::weakly_canonical(resolved_path, ec).lexically_normal()
        : resolved_path.lexically_normal();
    return std::ranges::any_of(roots, [&](const std::filesystem::path& root) {
        return is_within(root, normalized);
    });
}

} // namespace

FileAccessScope::FileAccessScope(
    std::vector<std::filesystem::path> readable_roots,
    std::vector<std::filesystem::path> writable_roots)
{
    normalize_roots(readable_roots, writable_roots);
    readable_roots_ = std::move(readable_roots);
    writable_roots_ = std::move(writable_roots);
}

void FileAccessScope::normalize_roots(
    std::vector<std::filesystem::path>& readable,
    std::vector<std::filesystem::path>& writable)
{
    std::vector<std::filesystem::path> normalized_readable;
    for (const auto& root : readable) {
        append_unique(normalized_readable, root);
    }
    std::vector<std::filesystem::path> normalized_writable;
    for (const auto& root : writable) {
        append_unique(normalized_writable, root);
    }
    // Writable implies readable. Enforcing it here means callers cannot build a
    // scope that grants a write to a directory it will not admit a read from.
    for (const auto& root : normalized_writable) {
        append_unique(normalized_readable, root);
    }
    readable = std::move(normalized_readable);
    writable = std::move(normalized_writable);
}

void FileAccessScope::normalize() {
    normalize_roots(readable_roots_, writable_roots_);
}

FileAccessScope FileAccessScope::host_temp_directories() {
    std::vector<std::filesystem::path> roots;

    std::error_code ec;
    if (auto temp = std::filesystem::temp_directory_path(ec); !ec) {
        roots.push_back(std::move(temp));
    }

    // temp_directory_path() only reports $TMPDIR (or /tmp). Name the
    // conventional roots explicitly so a process started with an unusual
    // TMPDIR still treats the well-known scratch directories as scratch.
    roots.emplace_back("/tmp");
    roots.emplace_back("/private/tmp");
    roots.emplace_back("/var/tmp");
    roots.emplace_back("/private/var/tmp");

    return FileAccessScope(roots, roots);
}

bool FileAccessScope::allows_read(const std::filesystem::path& path) const {
    return contains(readable_roots_, path);
}

bool FileAccessScope::allows_write(const std::filesystem::path& path) const {
    return contains(writable_roots_, path);
}

} // namespace core::workspace
