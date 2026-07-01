#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::context {
struct SessionContext;
}

namespace core::workspace {

enum class PathVisibilityKind {
    File,
    Directory,
};

class IPathVisibilityPolicy {
public:
    virtual ~IPathVisibilityPolicy() = default;

    [[nodiscard]] virtual bool empty() const noexcept = 0;

    [[nodiscard]] virtual std::optional<std::string> hidden_reason(
        std::string_view display_path,
        const std::filesystem::path& path,
        PathVisibilityKind kind) const = 0;

    [[nodiscard]] virtual bool should_prune_directory(
        const std::filesystem::path& path) const;
};

class PathVisibilityDecorator : public IPathVisibilityPolicy {
public:
    explicit PathVisibilityDecorator(
        std::shared_ptr<const IPathVisibilityPolicy> wrapped);

    [[nodiscard]] bool empty() const noexcept override;

    [[nodiscard]] std::optional<std::string> hidden_reason(
        std::string_view display_path,
        const std::filesystem::path& path,
        PathVisibilityKind kind) const override;

    [[nodiscard]] bool should_prune_directory(
        const std::filesystem::path& path) const override;

protected:
    [[nodiscard]] const IPathVisibilityPolicy& wrapped() const noexcept;

private:
    std::shared_ptr<const IPathVisibilityPolicy> wrapped_;
};

class AllowAllPathVisibilityPolicy final : public IPathVisibilityPolicy {
public:
    [[nodiscard]] bool empty() const noexcept override;

    [[nodiscard]] std::optional<std::string> hidden_reason(
        std::string_view display_path,
        const std::filesystem::path& path,
        PathVisibilityKind kind) const override;

    [[nodiscard]] bool should_prune_directory(
        const std::filesystem::path& path) const override;
};

class PathVisibility {
public:
    explicit PathVisibility(std::shared_ptr<const IPathVisibilityPolicy> policy);

    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] bool is_visible(
        const std::filesystem::path& path) const;

    [[nodiscard]] bool is_visible(
        const std::filesystem::directory_entry& entry) const;

    [[nodiscard]] bool is_hidden(
        const std::filesystem::path& path) const;

    [[nodiscard]] bool is_hidden(
        const std::filesystem::directory_entry& entry) const;

    [[nodiscard]] std::optional<std::string> hidden_reason(
        std::string_view display_path,
        const std::filesystem::path& path) const;

    [[nodiscard]] bool should_prune_directory(
        const std::filesystem::path& path) const;

    [[nodiscard]] bool should_prune_directory(
        const std::filesystem::directory_entry& entry) const;

private:
    std::shared_ptr<const IPathVisibilityPolicy> policy_;
};

class IPathVisibilityFactory {
public:
    virtual ~IPathVisibilityFactory() = default;

    [[nodiscard]] virtual PathVisibility unrestricted() const = 0;

    [[nodiscard]] virtual PathVisibility for_root(
        const std::filesystem::path& root) const = 0;

    [[nodiscard]] virtual PathVisibility for_roots(
        const std::vector<std::filesystem::path>& roots) const = 0;

    [[nodiscard]] virtual PathVisibility for_context(
        const core::context::SessionContext& context) const = 0;
};

class AgentIgnorePathVisibilityFactory final : public IPathVisibilityFactory {
public:
    [[nodiscard]] PathVisibility unrestricted() const override;

    [[nodiscard]] PathVisibility for_root(
        const std::filesystem::path& root) const override;

    [[nodiscard]] PathVisibility for_roots(
        const std::vector<std::filesystem::path>& roots) const override;

    [[nodiscard]] PathVisibility for_context(
        const core::context::SessionContext& context) const override;
};

using DirectoryPrunePredicate =
    std::function<bool(const std::filesystem::path&)>;

using RegularFileVisitor =
    std::function<bool(const std::filesystem::path&)>;

[[nodiscard]] std::vector<std::filesystem::directory_entry>
collect_visible_directory_entries(
    const std::filesystem::path& directory,
    const PathVisibility* visibility);

[[nodiscard]] std::vector<std::filesystem::directory_entry>
collect_visible_directory_entries(
    const std::filesystem::path& directory,
    const core::context::SessionContext& context);

[[nodiscard]] std::vector<std::filesystem::directory_entry>
collect_visible_directory_entries(
    const std::filesystem::path& directory,
    const PathVisibility& visibility);

[[nodiscard]] std::vector<std::filesystem::path>
collect_visible_regular_files(
    const std::filesystem::path& root,
    const PathVisibility* visibility,
    DirectoryPrunePredicate should_prune_directory = {});

[[nodiscard]] std::vector<std::filesystem::path>
collect_visible_regular_files(
    const std::filesystem::path& root,
    const core::context::SessionContext& context,
    DirectoryPrunePredicate should_prune_directory = {});

[[nodiscard]] std::vector<std::filesystem::path>
collect_visible_regular_files(
    const std::filesystem::path& root,
    const PathVisibility& visibility,
    DirectoryPrunePredicate should_prune_directory = {});

void visit_visible_regular_files(
    const std::filesystem::path& root,
    const core::context::SessionContext& context,
    DirectoryPrunePredicate should_prune_directory,
    RegularFileVisitor visitor);

} // namespace core::workspace
