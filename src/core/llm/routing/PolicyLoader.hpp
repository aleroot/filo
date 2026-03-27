#pragma once

#include "PolicyConfig.hpp"

#include <simdjson.h>
#include <string>

namespace core::llm::routing {

[[nodiscard]] RouterConfig make_default_router_config();

// Parse the "router" object from filo config.json.
// Returns false and fills error when the schema is invalid.
[[nodiscard]] bool parse_router_config(simdjson::dom::object router_obj,
                                       RouterConfig& out,
                                       std::string& error);

// Merge overlay router configuration into base.
void merge_router_config(RouterConfig& base, const RouterConfig& overlay);

} // namespace core::llm::routing
