#include "ModelGuidance.hpp"

#include <array>
#include <charconv>

namespace core::context {

namespace {

// ── OpenAI GPT ─────────────────────────────────────────────────────────────
// GPT-5 and later follow instructions literally. Read that way, "do not edit
// unless asked" turns a reported bug into a report, and "skipped steps go in
// the first sentence" headlines every answer with what was not done. The
// Codex client gives these models the opposite defaults (codex-rs
// models-manager models.json, model_messages.instructions_template:
// "Autonomy and persistence", "Rules for getting work done"); this profile
// carries those defaults in Filo's words and tool vocabulary.

[[nodiscard]] bool equals_ignoring_ascii_case(std::string_view text,
                                              std::string_view lower) noexcept {
    if (text.size() != lower.size()) return false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        const char folded = c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
        if (folded != lower[i]) return false;
    }
    return true;
}

/// gpt-5, gpt-5.6-terra, gpt-6.2-sol, openai/gpt-6...; not gpt-4o or gpt-oss.
[[nodiscard]] bool is_openai_gpt_5_or_later(std::string_view model_id) noexcept {
    if (const auto slash = model_id.rfind('/'); slash != std::string_view::npos) {
        model_id.remove_prefix(slash + 1);
    }
    constexpr std::string_view kPrefix = "gpt-";
    if (model_id.size() <= kPrefix.size()
        || !equals_ignoring_ascii_case(model_id.substr(0, kPrefix.size()), kPrefix)) {
        return false;
    }
    model_id.remove_prefix(kPrefix.size());
    int generation = 0;
    const auto parsed =
        std::from_chars(model_id.data(), model_id.data() + model_id.size(), generation);
    return parsed.ec == std::errc{} && generation >= 5;
}

constexpr std::array kOpenAIGptRules{
    WorkingRuleOverride{
        .rule = WorkingRule::ReportedProblems,
        .text = "When the user reports a bug, points out a mistake, or finds an unmet "
                "requirement, assume they want it fixed: investigate, fix, and verify. Answer "
                "without editing only when the user asks just for an explanation or a "
                "diagnosis, or tells you not to change anything.",
    },
    WorkingRuleOverride{
        .rule = WorkingRule::EvidenceBeforeClaims,
        .text = "\"Done\", \"fixed\", or \"verified\" requires evidence observed this "
                "session: tool output, the file as it now reads, a passing command; never "
                "present a workaround as a fix. Lead with what you did and found. State "
                "anything that failed or is still unverified once, plainly, after the "
                "result. Never add what you did not do or what remains unchanged, such as \"no "
                "code changed\" or \"no files were edited\", unless the user asked.",
    },
};

constexpr std::string_view kOpenAIGptInstructions =
    "\n\n[Model guidance]\n"
    "- When the user asks for new work or a fix, own it end to end: implement, verify, and "
    "report the result. Do not stop at a plan, a description of what you could do, an offer "
    "to do it, or a partial answer while authorized, in-scope work remains.\n"
    "- When the user follows up, questions your approach, or points out a mistake, carry the "
    "work through to a useful result instead of only acknowledging or explaining it.\n"
    "- Do not invent requirements, precautions, or approval steps that block or narrow the "
    "work. A precaution you chose, or a rule from an earlier task, binds only when the user, "
    "the project's instructions, or a real technical dependency requires it.\n"
    "- Before finishing, check the result against what the user asked for and complete "
    "anything still pending that you can do.\n"
    "- The current mode's limits still apply: in PLAN or RESEARCH mode, finishing the work "
    "means a complete plan or answer, not edits.";

// ── Registry ───────────────────────────────────────────────────────────────
// First match wins: list narrower profiles before broader ones.
constexpr std::array kProfiles{
    ModelGuidance{
        .name = "openai_gpt",
        .matches = is_openai_gpt_5_or_later,
        .rule_overrides = kOpenAIGptRules,
        .instructions = kOpenAIGptInstructions,
    },
};

} // namespace

std::span<const ModelGuidance> model_guidance_profiles() noexcept {
    return kProfiles;
}

const ModelGuidance* find_model_guidance(std::string_view model_id) noexcept {
    for (const auto& profile : model_guidance_profiles()) {
        if (profile.matches(model_id)) return &profile;
    }
    return nullptr;
}

} // namespace core::context
