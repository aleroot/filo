#pragma once

#include "../../config/ConfigManager.hpp"

#include <expected>
#include <string>

namespace core::tools::read {

/// Which provider/model backs the `read.question` worker.
struct ReaderProfileSelection {
    std::string provider;
    std::string model;
};

/// Pure resolution policy for the reader worker, kept free of singletons and
/// provider state so the decision is unit-testable on its own.
///
/// Resolution order:
///   1. `subagents.reader` — explicit worker configuration always wins.
///   2. The active routing policy's fast-tier candidate. A user who pinned
///      tiers already told Filo which model is cheap, so question answering
///      works out of the box wherever routing is configured.
///   3. Otherwise an error explaining both configuration paths.
[[nodiscard]] std::expected<ReaderProfileSelection, std::string>
resolve_reader_profile(const core::config::AppConfig& config);

} // namespace core::tools::read
