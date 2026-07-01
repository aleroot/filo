#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace core::context {
struct SessionContext;
}

namespace core::workspace {

enum class AgentIgnorePathKind {
    File,
    Directory,
};

namespace detail {

struct AgentIgnoreRule {
    std::string pattern;
    bool negated = false;
    bool directory_only = false;
    bool basename_only = false;
    std::vector<std::string> segments;
};

struct AgentIgnoreRootPolicy {
    std::filesystem::path root;
    std::vector<AgentIgnoreRule> rules;
};

} // namespace detail

class AgentIgnoreMatcher {
public:
    [[nodiscard]] static AgentIgnoreMatcher load_for_root(
        const std::filesystem::path& root);

    [[nodiscard]] static AgentIgnoreMatcher load_for_roots(
        const std::vector<std::filesystem::path>& roots);

    [[nodiscard]] static AgentIgnoreMatcher load_for_context(
        const core::context::SessionContext& context);

    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] bool is_ignored(
        const std::filesystem::path& path,
        AgentIgnorePathKind kind) const;

    [[nodiscard]] bool can_prune_directory(
        const std::filesystem::path& path) const;

private:
    explicit AgentIgnoreMatcher(std::vector<detail::AgentIgnoreRootPolicy> policies);

    std::vector<detail::AgentIgnoreRootPolicy> policies_;
};

[[nodiscard]] std::string format_agent_ignore_error(
    std::string_view path,
    std::string_view source = ".agentignore");

} // namespace core::workspace
