#pragma once

#include "ReadTypes.hpp"

#include <optional>
#include <string_view>

namespace core::tools::read {

/**
 * Recover the intended read from a `path` argument that holds serialized
 * argument text instead of a path.
 *
 * Grammar-constrained tool samplers occasionally write JSON into the string
 * branch of `path`: a stringified path array (`"[\"a.cpp\",\"b.cpp\"]"`) or a
 * corrupted re-serialization of the whole argument object, line window
 * included (`[  "src/x.cpp",  "offset_line\": 90,  "  ]`). Such text always
 * carries JSON string quotes, which ordinary paths do not.
 *
 * The recovered path (or clean path array) and any embedded offset_line /
 * limit_lines are rebuilt into a canonical argument object and validated by
 * parse_options(), so a recovered read satisfies exactly the contract of a
 * well-formed one. Explicit arguments always win over embedded ones.
 *
 * Pure and filesystem-free: callers decide whether the literal path exists
 * and therefore takes precedence.
 *
 * @return The recovered options, or nullopt when @p json_args is not a read
 *         call whose single string path contains serialized text, or when
 *         nothing valid can be recovered from it.
 */
[[nodiscard]] std::optional<Options> recover_serialized_path(std::string_view json_args);

} // namespace core::tools::read
