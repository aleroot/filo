#include "PathVisibility.hpp"

#include "AgentIgnore.hpp"
#include "SensitivePathPolicy.hpp"
#include "../context/SessionContext.hpp"
#include "../landrun/LandrunSettings.hpp"

#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace core::workspace {

namespace {

[[nodiscard]] PathVisibilityKind classify_path_kind(
    const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec)
        ? PathVisibilityKind::Directory
        : PathVisibilityKind::File;
}

[[nodiscard]] PathVisibilityKind classify_entry_kind(
    const std::filesystem::directory_entry& entry) {
    std::error_code ec;
    return entry.is_directory(ec)
        ? PathVisibilityKind::Directory
        : PathVisibilityKind::File;
}

[[nodiscard]] AgentIgnorePathKind to_agent_ignore_kind(PathVisibilityKind kind) {
    return kind == PathVisibilityKind::Directory
        ? AgentIgnorePathKind::Directory
        : AgentIgnorePathKind::File;
}

class AgentIgnoreVisibilityDecorator final : public PathVisibilityDecorator {
public:
    AgentIgnoreVisibilityDecorator(
        std::shared_ptr<const IPathVisibilityPolicy> wrapped,
        AgentIgnoreMatcher matcher)
        : PathVisibilityDecorator(std::move(wrapped))
        , matcher_(std::move(matcher)) {}

    [[nodiscard]] bool empty() const noexcept override {
        return PathVisibilityDecorator::empty() && matcher_.empty();
    }

    [[nodiscard]] std::optional<std::string> hidden_reason(
        std::string_view display_path,
        const std::filesystem::path& path,
        PathVisibilityKind kind) const override {
        if (const auto upstream =
                PathVisibilityDecorator::hidden_reason(display_path, path, kind)) {
            return upstream;
        }
        if (!matcher_.is_ignored(path, to_agent_ignore_kind(kind))) {
            return std::nullopt;
        }
        return format_agent_ignore_error(display_path);
    }

    [[nodiscard]] bool should_prune_directory(
        const std::filesystem::path& path) const override {
        return PathVisibilityDecorator::should_prune_directory(path)
            || matcher_.can_prune_directory(path);
    }

private:
    AgentIgnoreMatcher matcher_;
};

class SensitivePathVisibilityPolicy final : public IPathVisibilityPolicy {
public:
    [[nodiscard]] bool empty() const noexcept override { return false; }

    [[nodiscard]] std::optional<std::string> hidden_reason(
        std::string_view,
        const std::filesystem::path& path,
        PathVisibilityKind) const override {
        if (!is_sensitive_agent_path(path)) return std::nullopt;
        return std::string(kSensitivePathReason);
    }

    [[nodiscard]] bool should_prune_directory(
        const std::filesystem::path& path) const override {
        return is_sensitive_agent_path(path);
    }
};

[[nodiscard]] std::shared_ptr<const IPathVisibilityPolicy> base_policy() {
    if (core::landrun::LandrunSettings::instance().enabled()) {
        return std::make_shared<SensitivePathVisibilityPolicy>();
    }
    return std::make_shared<AllowAllPathVisibilityPolicy>();
}

[[nodiscard]] std::shared_ptr<const IPathVisibilityPolicy> with_agent_ignore(
    std::shared_ptr<const IPathVisibilityPolicy> wrapped,
    AgentIgnoreMatcher matcher) {
    return std::make_shared<AgentIgnoreVisibilityDecorator>(
        std::move(wrapped),
        std::move(matcher));
}

} // namespace

PathVisibilityDecorator::PathVisibilityDecorator(
    std::shared_ptr<const IPathVisibilityPolicy> wrapped)
    : wrapped_(std::move(wrapped)) {
    if (!wrapped_) {
        throw std::invalid_argument("PathVisibilityDecorator requires a wrapped policy");
    }
}

bool PathVisibilityDecorator::empty() const noexcept {
    return wrapped_->empty();
}

std::optional<std::string> PathVisibilityDecorator::hidden_reason(
    std::string_view display_path,
    const std::filesystem::path& path,
    PathVisibilityKind kind) const {
    return wrapped_->hidden_reason(display_path, path, kind);
}

bool IPathVisibilityPolicy::should_prune_directory(
    const std::filesystem::path&) const {
    return false;
}

bool PathVisibilityDecorator::should_prune_directory(
    const std::filesystem::path& path) const {
    return wrapped_->should_prune_directory(path);
}

const IPathVisibilityPolicy& PathVisibilityDecorator::wrapped() const noexcept {
    return *wrapped_;
}

bool AllowAllPathVisibilityPolicy::empty() const noexcept {
    return true;
}

std::optional<std::string> AllowAllPathVisibilityPolicy::hidden_reason(
    std::string_view,
    const std::filesystem::path&,
    PathVisibilityKind) const {
    return std::nullopt;
}

bool AllowAllPathVisibilityPolicy::should_prune_directory(
    const std::filesystem::path&) const {
    return false;
}

PathVisibility::PathVisibility(std::shared_ptr<const IPathVisibilityPolicy> policy)
    : policy_(std::move(policy)) {
    if (!policy_) {
        throw std::invalid_argument("PathVisibility requires a policy");
    }
}

PathVisibility AgentIgnorePathVisibilityFactory::unrestricted() const {
    return PathVisibility(base_policy());
}

PathVisibility AgentIgnorePathVisibilityFactory::for_root(
    const std::filesystem::path& root) const {
    return PathVisibility(with_agent_ignore(
        base_policy(),
        AgentIgnoreMatcher::load_for_root(root)));
}

PathVisibility AgentIgnorePathVisibilityFactory::for_roots(
    const std::vector<std::filesystem::path>& roots) const {
    return PathVisibility(with_agent_ignore(
        base_policy(),
        AgentIgnoreMatcher::load_for_roots(roots)));
}

PathVisibility AgentIgnorePathVisibilityFactory::for_context(
    const core::context::SessionContext& context) const {
    return PathVisibility(with_agent_ignore(
        base_policy(),
        AgentIgnoreMatcher::load_for_context(context)));
}

bool PathVisibility::empty() const noexcept {
    return policy_->empty();
}

bool PathVisibility::is_visible(const std::filesystem::path& path) const {
    return !is_hidden(path);
}

bool PathVisibility::is_visible(
    const std::filesystem::directory_entry& entry) const {
    return !is_hidden(entry);
}

bool PathVisibility::is_hidden(const std::filesystem::path& path) const {
    return hidden_reason(path.string(), path).has_value();
}

bool PathVisibility::is_hidden(
    const std::filesystem::directory_entry& entry) const {
    return policy_->hidden_reason(
        entry.path().string(),
        entry.path(),
        classify_entry_kind(entry)).has_value();
}

std::optional<std::string> PathVisibility::hidden_reason(
    std::string_view display_path,
    const std::filesystem::path& path) const {
    return policy_->hidden_reason(display_path, path, classify_path_kind(path));
}

bool PathVisibility::should_prune_directory(
    const std::filesystem::path& path) const {
    return policy_->should_prune_directory(path);
}

bool PathVisibility::should_prune_directory(
    const std::filesystem::directory_entry& entry) const {
    return should_prune_directory(entry.path());
}

std::vector<std::filesystem::directory_entry> collect_visible_directory_entries(
    const std::filesystem::path& directory,
    const PathVisibility* visibility) {
    std::vector<std::filesystem::directory_entry> entries;
    std::error_code ec;
    std::filesystem::directory_iterator it(
        directory,
        std::filesystem::directory_options::skip_permission_denied,
        ec);
    const auto end = std::filesystem::directory_iterator{};

    for (; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }

        if (visibility != nullptr && !visibility->is_visible(*it)) {
            continue;
        }
        entries.push_back(*it);
    }
    return entries;
}

std::vector<std::filesystem::directory_entry> collect_visible_directory_entries(
    const std::filesystem::path& directory,
    const PathVisibility& visibility) {
    return collect_visible_directory_entries(directory, &visibility);
}

std::vector<std::filesystem::directory_entry> collect_visible_directory_entries(
    const std::filesystem::path& directory,
    const core::context::SessionContext& context) {
    std::vector<std::filesystem::directory_entry> entries;
    for (const auto& entry : collect_visible_directory_entries(
             directory,
             context.path_visibility.get())) {
        if (!context.is_path_allowed(entry.path())) {
            continue;
        }
        entries.push_back(entry);
    }
    return entries;
}

void visit_visible_regular_files(
    const std::filesystem::path& root,
    const PathVisibility* visibility,
    DirectoryPrunePredicate should_prune_directory,
    RegularFileVisitor visitor);

std::vector<std::filesystem::path> collect_visible_regular_files(
    const std::filesystem::path& root,
    const PathVisibility* visibility,
    DirectoryPrunePredicate should_prune_directory) {
    std::vector<std::filesystem::path> files;
    visit_visible_regular_files(
        root,
        visibility,
        std::move(should_prune_directory),
        [&](const std::filesystem::path& file) {
            files.push_back(file);
            return true;
        });
    return files;
}

void visit_visible_regular_files(
    const std::filesystem::path& root,
    const PathVisibility* visibility,
    DirectoryPrunePredicate should_prune_directory,
    RegularFileVisitor visitor) {
    if (!visitor) {
        return;
    }

    std::error_code ec;
    std::filesystem::recursive_directory_iterator it(
        root,
        std::filesystem::directory_options::skip_permission_denied,
        ec);
    const auto end = std::filesystem::recursive_directory_iterator{};

    for (; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }

        const auto& entry = *it;
        if (entry.is_directory(ec)) {
            const bool prune_by_name =
                should_prune_directory && should_prune_directory(entry.path());
            const bool prune_by_visibility =
                visibility != nullptr && visibility->should_prune_directory(entry);
            if (prune_by_name || prune_by_visibility) {
                it.disable_recursion_pending();
            }
            ec.clear();
            continue;
        }
        ec.clear();

        if (!entry.is_regular_file(ec)) {
            ec.clear();
            continue;
        }
        if (visibility != nullptr && !visibility->is_visible(entry)) {
            continue;
        }
        if (!visitor(entry.path())) {
            break;
        }
    }
}

std::vector<std::filesystem::path> collect_visible_regular_files(
    const std::filesystem::path& root,
    const core::context::SessionContext& context,
    DirectoryPrunePredicate should_prune_directory) {
    std::vector<std::filesystem::path> files;
    visit_visible_regular_files(
        root,
        context,
        std::move(should_prune_directory),
        [&](const std::filesystem::path& file) {
            files.push_back(file);
            return true;
        });
    return files;
}

std::vector<std::filesystem::path> collect_visible_regular_files(
    const std::filesystem::path& root,
    const PathVisibility& visibility,
    DirectoryPrunePredicate should_prune_directory) {
    return collect_visible_regular_files(
        root,
        &visibility,
        std::move(should_prune_directory));
}

void visit_visible_regular_files(
    const std::filesystem::path& root,
    const core::context::SessionContext& context,
    DirectoryPrunePredicate should_prune_directory,
    RegularFileVisitor visitor) {
    if (!visitor) {
        return;
    }
    visit_visible_regular_files(
        root,
        context.path_visibility.get(),
        std::move(should_prune_directory),
        [&](const std::filesystem::path& file) {
            if (!context.is_path_allowed(file)) {
                return true;
            }
            return visitor(file);
        });
}

} // namespace core::workspace
