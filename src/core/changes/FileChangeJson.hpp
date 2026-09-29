#pragma once

#include "FileChange.hpp"

#include <simdjson.h>

#include <string>

namespace core::changes {

/// Appends `changes` as a JSON object. The fidelity flags are written only
/// when set, so a complete summary costs nothing beyond its files.
void append_turn_changes_json(std::string& out, const TurnChanges& changes);

/// Reads an object written by `append_turn_changes_json`. Entries without a
/// path are skipped; unknown kinds read as Modified and absent flags as a
/// complete summary.
[[nodiscard]] TurnChanges parse_turn_changes_json(simdjson::dom::object object);

} // namespace core::changes
