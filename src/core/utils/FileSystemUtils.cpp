#include "FileSystemUtils.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace core::utils {

namespace {

struct TreeNode {
  std::string name;
  bool directory = false;
  bool traversable = false;
  std::vector<std::unique_ptr<TreeNode>> children;
};

struct FrontierEntry {
  std::filesystem::path path;
  TreeNode *node = nullptr;
};

struct Candidate {
  std::filesystem::path path;
  std::string name;
  TreeNode *parent = nullptr;
  bool directory = false;
  bool traversable = false;
};

[[nodiscard]] bool is_scm_metadata(std::string_view name) noexcept {
  static constexpr std::array<std::string_view, 3> kNames = {
      ".git", ".hg", ".svn"};
  return std::ranges::find(kNames, name) != kNames.end();
}

[[nodiscard]] std::vector<Candidate> collect_level(
    const std::vector<FrontierEntry> &frontier,
    std::size_t scan_limit,
    bool &truncated) {
  std::vector<Candidate> candidates;
  candidates.reserve(std::min(scan_limit, std::size_t{256}));

  for (const auto &parent : frontier) {
    std::vector<Candidate> siblings;
    std::error_code ec;
    std::filesystem::directory_iterator it{
        parent.path,
        std::filesystem::directory_options::skip_permission_denied,
        ec};
    const std::filesystem::directory_iterator end;
    while (!ec && it != end) {
      const auto &entry = *it;
      const std::string name = entry.path().filename().string();
      if (!is_scm_metadata(name)) {
        std::error_code type_ec;
        const bool directory = entry.is_directory(type_ec);
        type_ec.clear();
        const bool symlink = entry.is_symlink(type_ec);
        siblings.push_back(Candidate{
            .path = entry.path(),
            .name = name,
            .parent = parent.node,
            .directory = directory,
            .traversable = directory && !symlink,
        });
      }
      it.increment(ec);
    }

    std::ranges::sort(siblings, [](const Candidate &lhs, const Candidate &rhs) {
      if (lhs.directory != rhs.directory) return lhs.directory > rhs.directory;
      return lhs.name < rhs.name;
    });

    for (auto &candidate : siblings) {
      if (candidates.size() >= scan_limit) {
        truncated = true;
        return candidates;
      }
      candidates.push_back(std::move(candidate));
    }
  }
  return candidates;
}

[[nodiscard]] std::vector<bool> ignored_flags(
    const core::scm::SourceControlProvider &scm,
    const std::vector<Candidate> &candidates) {
  std::vector<std::filesystem::path> paths;
  paths.reserve(candidates.size());
  for (const auto &candidate : candidates) paths.push_back(candidate.path);

  if (auto ignored = scm.ignored_paths(paths);
      ignored.has_value() && ignored->size() == candidates.size()) {
    return std::move(*ignored);
  }

  std::vector<bool> ignored;
  ignored.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    ignored.push_back(scm.is_ignored(candidate.path));
  }
  return ignored;
}

void render_nodes(const std::vector<std::unique_ptr<TreeNode>> &nodes,
                  std::string_view prefix,
                  std::string &out,
                  std::size_t max_chars,
                  bool &truncated) {
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = *nodes[i];
    const bool last = i + 1 == nodes.size();
    std::string line;
    line.reserve(prefix.size() + node.name.size() + 12);
    line += prefix;
    line += last ? "└── " : "├── ";
    line += node.name;
    if (node.directory) line += '/';
    line += '\n';

    if (line.size() > max_chars - std::min(max_chars, out.size())) {
      truncated = true;
      return;
    }
    out += line;

    if (!node.children.empty()) {
      std::string child_prefix(prefix);
      child_prefix += last ? "    " : "│   ";
      render_nodes(node.children, child_prefix, out, max_chars, truncated);
      if (truncated) return;
    }
  }
}

void append_truncation_marker(std::string &out, std::size_t max_chars) {
  constexpr std::string_view kMarker = "… (repository tree truncated)\n";
  if (max_chars < kMarker.size()) {
    out.clear();
    return;
  }
  while (!out.empty() && out.size() + kMarker.size() > max_chars) {
    const auto end = out.size() > 1 ? out.rfind('\n', out.size() - 2) : std::string::npos;
    if (end == std::string::npos) {
      out.clear();
      break;
    }
    out.resize(end + 1);
  }
  out += kMarker;
}

} // namespace

std::string get_file_tree(const std::filesystem::path &root,
                          const core::scm::SourceControlProvider &scm,
                          int max_depth,
                          std::size_t max_entries,
                          std::size_t max_chars) {
  std::error_code ec;
  if (max_depth <= 0 || max_entries == 0 || max_chars == 0
      || !std::filesystem::is_directory(root, ec)) {
    return {};
  }

  const std::size_t scan_limit = max_entries > std::numeric_limits<std::size_t>::max() / 8
      ? std::numeric_limits<std::size_t>::max()
      : max_entries * 8;
  std::vector<std::unique_ptr<TreeNode>> roots;
  std::vector<FrontierEntry> frontier{{root, nullptr}};
  std::size_t visible_entries = 0;
  bool truncated = false;

  for (int depth = 1; depth <= max_depth && !frontier.empty(); ++depth) {
    auto candidates = collect_level(frontier, scan_limit, truncated);
    const auto ignored = ignored_flags(scm, candidates);
    std::vector<FrontierEntry> next_frontier;

    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (ignored[i]) continue;
      if (visible_entries >= max_entries) {
        truncated = true;
        break;
      }

      auto node = std::make_unique<TreeNode>(TreeNode{
          .name = std::move(candidates[i].name),
          .directory = candidates[i].directory,
          .traversable = candidates[i].traversable,
      });
      TreeNode *node_ptr = node.get();
      if (candidates[i].parent) {
        candidates[i].parent->children.push_back(std::move(node));
      } else {
        roots.push_back(std::move(node));
      }
      ++visible_entries;

      if (depth < max_depth && node_ptr->traversable) {
        next_frontier.push_back(FrontierEntry{
            .path = std::move(candidates[i].path),
            .node = node_ptr,
        });
      }
    }
    frontier = std::move(next_frontier);
  }

  std::string out;
  out.reserve(std::min(max_chars, visible_entries * std::size_t{32}));
  render_nodes(roots, {}, out, max_chars, truncated);
  if (truncated) append_truncation_marker(out, max_chars);
  return out;
}

} // namespace core::utils
