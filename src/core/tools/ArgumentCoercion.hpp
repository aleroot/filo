#pragma once

/**
 * @file
 * @brief Schema-directed repair of model-supplied tool arguments.
 *
 * Models frequently emit a value whose *JSON type* does not match the tool
 * contract even though their intent is unambiguous: an array serialized as a
 * string ("[{\"old_string\":...}]"), a lone item where a list is expected, a
 * quoted number, or the fields of a single list item flattened onto the
 * argument root. Rejecting these costs a full model round trip and, worse,
 * tends to repeat.
 *
 * This unit implements the industry-standard remedy — coercion keyed off the
 * declared schema — under two hard rules learned from the failure modes of
 * other agents:
 *
 *  1. Try-validate-then-coerce. Coercion only ever runs after strict
 *     validation has already failed, so a conforming payload is never touched.
 *  2. Never guess. A rewritten payload is returned only when it validates
 *     completely against the same authoritative schema; a string that merely
 *     *looks* like JSON but does not parse is never wrapped, unwrapped, or
 *     otherwise smuggled through (the silent-fallback bug that turns a
 *     malformed argument into a confusing downstream error).
 *
 * Deliberately out of scope: repairing malformed JSON. Measured against the
 * real corpus of rejected calls, tolerant re-parsing recovered a minority of
 * payloads and silently corrupted replacement text in some of them, which for
 * file-mutating tools is worse than a clean rejection. Those payloads get a
 * precise explanation instead (explain_string_value()).
 */

#include <optional>
#include <string>
#include <string_view>

namespace core::tools::schema {

/**
 * Rewrites @p arguments_json so its value shapes satisfy @p schema_json.
 *
 * @param arguments_json Well-formed JSON object of tool arguments.
 * @param schema_json    Authoritative canonical input schema.
 * @return The rewritten arguments when they validate against the schema;
 *         nullopt when nothing could be coerced, when the rewrite still fails
 *         validation, or when either input is not parsable.
 */
[[nodiscard]] std::optional<std::string> coerce_arguments(
    std::string_view arguments_json,
    std::string_view schema_json);

/**
 * Explains, for a model, why the string @p value could not serve as a
 * @p expected_type value.
 *
 * Returns a sentence fragment that begins with " — " and is meant to be
 * appended to a type-mismatch message, or an empty string when there is
 * nothing useful to add beyond the mismatch itself.
 */
[[nodiscard]] std::string explain_string_value(std::string_view value,
                                               std::string_view expected_type);

} // namespace core::tools::schema
