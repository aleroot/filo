#include "PathDiagnostics.hpp"

#include "../ToolArgumentUtils.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <system_error>

namespace core::tools::read {

std::optional<std::string> misplaced_argument_error(std::span<const std::string> paths) {
    std::string listed;
    std::string_view first;
    std::size_t count = 0;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        if (std::ranges::find(kArgumentNames, paths[i]) == kArgumentNames.end()) continue;
        if (first.empty()) first = paths[i];
        if (count++) listed += ", ";
        listed += std::format("path[{}] \"{}\"", i, paths[i]);
    }
    if (!count) return std::nullopt;
    const bool one = count == 1;
    return std::format(
        "{} {} read argument {}, not {}: arguments go beside path, not inside its array "
        "(for example {{\"path\":\"a.txt\",\"{}\":...}}). To read a file with that name, write ./{}.",
        listed, one ? "is a" : "are", one ? "name" : "names", one ? "a path" : "paths", first, first);
}

std::string missing_path_reason(std::string_view path, const core::context::SessionContext& context) {
    constexpr std::string_view kReason = "path does not exist";
    const auto stripped = path.find_first_not_of(':');
    if (stripped == 0 || stripped == std::string_view::npos) return std::string(kReason);
    const std::string candidate(path.substr(stripped));
    std::filesystem::path resolved;
    if (detail::check_workspace_access(candidate, candidate, context, &resolved, names::kRead))
        return std::string(kReason);
    std::error_code ec;
    if (!std::filesystem::exists(resolved, ec)) return std::string(kReason);
    return std::format("{}; '{}' exists, so drop the leading ':'", kReason, candidate);
}

} // namespace core::tools::read
