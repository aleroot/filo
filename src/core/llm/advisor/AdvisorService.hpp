#pragma once

#include "../Advisor.hpp"
#include "../LLMProvider.hpp"

#include <string>
#include <string_view>

namespace core::llm::advisor {

/**
 * @brief Resolves the advisor for one turn.
 *
 * The provider owns the capability: a wire without one refuses, and a
 * provider that documents which advisor its executor accepts refuses the
 * pairings it does not. This function only decides, it does not know any
 * vendor's rules.
 *
 * @param provider       The provider that will serve the turn.
 * @param executor_model The model that will run the turn.
 * @param advisor_model  What the session asked for; empty means no advisor.
 */
[[nodiscard]] AdvisorPlan resolve(const LLMProvider& provider,
                                   std::string_view executor_model,
                                   std::string_view advisor_model);

} // namespace core::llm::advisor
