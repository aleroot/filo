#pragma once

#include "ReadTypes.hpp"

#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace core::tools::read {

/**
 * Explain a `path` array that carries read's own arguments as entries.
 *
 * Models occasionally write the arguments that belong beside `path` into its
 * array (`["a.cpp","offset_line","40"]`). Reading such an entry as a file can
 * only report "path does not exist", which names nothing to fix and tends to
 * be repeated, so the call is rejected with the correction instead. Only
 * exact argument names match: `./offset_line` still names a file.
 *
 * Pure and filesystem-free.
 *
 * @return The error naming every misplaced entry by index, or nullopt when
 *         no entry is an argument name.
 */
[[nodiscard]] std::optional<std::string> misplaced_argument_error(std::span<const std::string> paths);

/**
 * The reason a missing @p path could not be read.
 *
 * When @p path carries stray leading ':' characters (a `path: value` rendering
 * leaking into the value) and the stripped path exists, the reason names it.
 * The stripped path must clear the same workspace gate as a read, so the
 * reason never reveals whether a file outside the session's scope exists.
 */
[[nodiscard]] std::string missing_path_reason(std::string_view path,
                                              const core::context::SessionContext& context);

} // namespace core::tools::read
