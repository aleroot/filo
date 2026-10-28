#pragma once

#include "PolicyRules.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace core::policy {

/**
 * Parses a network policy document.
 *
 * The document is JSON:
 *
 *   {
 *     "version": 1,
 *     "default": "deny",
 *     "rules": [
 *       { "id": "read-github",
 *         "effect": "allow",
 *         "principal": "web_tool:fetch_url",
 *         "operation": "read",
 *         "flow": "inbound",
 *         "host": "*.github.com",
 *         "port": 443 }
 *     ]
 *   }
 *
 * "principal" is a principal kind, a kind and an identifier separated by a
 * colon, or "*". "operation", "flow" and "host" accept "*". "port" is a
 * number, a "low-high" range, or "*".
 *
 * Parsing is strict: an unknown key, an unknown value, or a present field of
 * the wrong type is an error. A wrong type is never treated as an omitted
 * field, because an omitted field is a wildcard. A policy that cannot be
 * understood must never be treated as an absent policy, so the caller is
 * expected to refuse to start rather than continue unprotected.
 */
[[nodiscard]] std::expected<PolicySet, std::string> parse_policy_document(std::string_view json_text);

// Reads and parses a policy file.
[[nodiscard]] std::expected<PolicySet, std::string> load_policy_file(const std::filesystem::path& path);

} // namespace core::policy
