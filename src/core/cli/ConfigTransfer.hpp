#pragma once

#include "core/portability/ConfigPortability.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace core::cli {

/// `filo --export` / `filo --import` as parsed from the command line.
struct ConfigTransferRequest {
    std::optional<std::string> export_target;  ///< File path, or "-" for stdout.
    std::optional<std::string> import_source;  ///< File path, or "-" for stdin.
    std::vector<std::string> only;
    std::vector<std::string> except;
    std::vector<std::string> map_paths;
    bool include_secrets = false;
    /// Gzip the exported bundle even when the target does not end in `.gz`.
    bool compress = false;
    bool dry_run = false;
    /// Secrets are never written to an interactive terminal.
    bool stdout_is_terminal = false;
};

/// Runs one export or import and prints its report. The bundle itself goes
/// to `out` only for `--export -`; the report then moves to `err` so the two
/// never mix. Returns the process exit code.
[[nodiscard]] int run_config_transfer(const ConfigTransferRequest& request,
                                      const core::portability::PortabilityPaths& paths,
                                      std::istream& in,
                                      std::ostream& out,
                                      std::ostream& err);

/// Help text listing the selectable kinds, generated from item_kinds().
[[nodiscard]] std::string config_transfer_selector_help();

} // namespace core::cli
