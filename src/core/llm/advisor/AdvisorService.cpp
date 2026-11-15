#include "AdvisorService.hpp"

namespace core::llm::advisor {

AdvisorPlan resolve(const LLMProvider& provider,
                    std::string_view executor_model,
                    std::string_view advisor_model) {
    AdvisorPlan plan;
    if (advisor_model.empty()) return plan;

    const auto support = provider.advisor_support();
    if (!support) {
        plan.refusal = "This provider cannot run an advisor.";
        return plan;
    }

    const AdvisorSpec spec{.model = std::string(advisor_model)};
    if (!support->supports(executor_model, spec)) {
        plan.refusal = support->unsupported_reason(executor_model, spec);
        if (plan.refusal.empty()) {
            plan.refusal = "The provider does not accept this advisor.";
        }
        return plan;
    }

    plan.spec = spec;
    return plan;
}

} // namespace core::llm::advisor
