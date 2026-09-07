#pragma once
#include "ReadTypes.hpp"
#include <span>

namespace core::tools::read {
struct Evidence {
    std::string text;
    std::vector<std::vector<std::size_t>> supplied_lines;
    bool partial = false;
};
// Deterministic lexical selection, not semantic analysis. Returned line numbers
// are authoritative for validation; omitted text never counts as evidence.
[[nodiscard]] Evidence select_evidence(std::span<const Resource> sources,
    std::string_view question, std::size_t budget);
} // namespace core::tools::read
