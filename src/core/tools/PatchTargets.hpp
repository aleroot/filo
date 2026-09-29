#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace core::tools {

/// Paths named by the `---`/`+++` headers of a unified diff, as `patch -p1`
/// sees them: `/dev/null` skipped, `a/` and `b/` stripped, quotes removed,
/// relative to the patch working directory. In header order; may repeat.
[[nodiscard]] std::vector<std::string> patch_target_paths(std::string_view patch_text);

} // namespace core::tools
