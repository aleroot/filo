#include "AutoClassifier.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>

namespace core::llm::routing {

// ─── Task type base complexity factors ───────────────────────────────────────
// These are the dominant weights before length/content/turn boosts are applied.
static constexpr double kFactorTrivial      = 0.05;
static constexpr double kFactorSimple       = 0.15;
static constexpr double kFactorCodeGen      = 0.40;
static constexpr double kFactorDebugging    = 0.60;
static constexpr double kFactorArchitecture = 0.80;
static constexpr double kFactorReasoning    = 0.85;

namespace {

[[nodiscard]] std::string lower_string(std::string_view value) noexcept {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

[[nodiscard]] bool contains(std::string_view haystack, std::string_view needle) noexcept {
    return haystack.find(needle) != std::string_view::npos;
}

// Count how many of the given keywords appear in the lowercase text.
[[nodiscard]] int count_matches(std::string_view lower,
                                 std::initializer_list<std::string_view> keywords) noexcept {
    int n = 0;
    for (const auto kw : keywords) {
        if (contains(lower, kw)) ++n;
    }
    return n;
}

// Rough token count from raw character count (GPT-style: ~4 chars per token).
[[nodiscard]] std::size_t chars_to_tokens(std::size_t chars) noexcept {
    return chars / 4 + 1;
}

} // namespace

// ─── String helpers ───────────────────────────────────────────────────────────

std::string_view to_string(TaskType task_type) noexcept {
    switch (task_type) {
        case TaskType::Trivial:      return "Trivial";
        case TaskType::Simple:       return "Simple";
        case TaskType::CodeGen:      return "CodeGen";
        case TaskType::Debugging:    return "Debugging";
        case TaskType::Architecture: return "Architecture";
        case TaskType::Reasoning:    return "Reasoning";
    }
    return "Unknown";
}

std::string_view to_string(Tier tier) noexcept {
    switch (tier) {
        case Tier::Fast:     return "fast";
        case Tier::Balanced: return "balanced";
        case Tier::Powerful: return "powerful";
    }
    return "balanced";
}

// ─── AutoClassifier ───────────────────────────────────────────────────────────

AutoClassifier::AutoClassifier(AutoClassifierConfig config) noexcept
    : config_(config) {}

ClassificationInput AutoClassifier::extract_signals(std::string_view prompt,
                                                     std::size_t      history_tokens,
                                                     int              turn_count,
                                                     bool             has_tool_history) noexcept {
    ClassificationInput input;
    input.prompt         = prompt;
    input.history_tokens = history_tokens;
    input.turn_count     = turn_count;
    input.has_tool_history = has_tool_history;

    // Estimate prompt tokens from character count when caller didn't provide them.
    input.prompt_tokens = chars_to_tokens(prompt.size());

    // Content signal detection — scan raw prompt to keep it O(n) single pass.
    // We look for markers that strongly indicate a specific task type without
    // requiring an expensive NLP model.

    bool in_code_fence = false;
    std::size_t i = 0;
    while (i < prompt.size()) {
        // Code fence (``` or ~~~)
        if (i + 2 < prompt.size()
            && ((prompt[i] == '`' && prompt[i+1] == '`' && prompt[i+2] == '`')
             || (prompt[i] == '~' && prompt[i+1] == '~' && prompt[i+2] == '~'))) {
            in_code_fence = !in_code_fence;
            input.has_code_block = true;
            i += 3;
            continue;
        }

        // Stack trace markers (language-agnostic)
        if (!in_code_fence) {
            const std::string_view rest = prompt.substr(i);
            if (rest.starts_with("Traceback")
             || rest.starts_with("  at ")
             || rest.starts_with("\tat ")
             || rest.starts_with("    at ")
             || rest.starts_with("caused by")
             || rest.starts_with("Caused by")) {
                input.has_stack_trace = true;
            }

            // Error message markers
            if (rest.starts_with("Error:")
             || rest.starts_with("error:")
             || rest.starts_with("Exception:")
             || rest.starts_with("FATAL:")
             || rest.starts_with("PANIC:")
             || rest.starts_with("TypeError")
             || rest.starts_with("ValueError")
             || rest.starts_with("AttributeError")
             || rest.starts_with("RuntimeError")
             || rest.starts_with("NullPointerException")
             || rest.starts_with("SegmentationFault")
             || rest.starts_with("segfault")) {
                input.has_error_message = true;
            }
        }
        ++i;
    }

    return input;
}

ClassificationResult AutoClassifier::classify(const ClassificationInput& input) const noexcept {
    const std::string prompt_lower = lower_string(input.prompt);

    const TaskType task_type = classify_task_type(prompt_lower, input);
    const double complexity  = compute_complexity(task_type, input, prompt_lower);
    const Tier tier          = select_tier(complexity);

    ClassificationResult result;
    result.task_type  = task_type;
    result.tier       = tier;
    result.complexity = complexity;
    result.reason     = std::format("{}·{:.2f}", to_string(task_type), complexity);
    return result;
}

TaskType AutoClassifier::classify_task_type(std::string_view prompt_lower,
                                             const ClassificationInput& input) const noexcept {
    // Trivial: very short and contains common conversational filler only.
    if (prompt_lower.size() < 30
        && count_matches(prompt_lower, {"hello", "hi", "hey", "thanks", "thank you",
                                        "ok", "okay", "sure", "yes", "no", "great",
                                        "bye", "goodbye", "cool", "nice"}) >= 1) {
        return TaskType::Trivial;
    }

    // Reasoning: formal proofs, maths, deep analysis, comparisons.
    // Use word-bounded or specific phrases to avoid false positives:
    //  "prove" would match "improve" → use "prove that" / "can you prove"
    //  "lemma" would match "dilemma" → use "prove a lemma" / "the lemma"
    if (count_matches(prompt_lower, {
            "prove that", "prove it", "prove the", " prove ", "prove by",
            "the theorem", "by theorem", "mathematical proof", "mathematical induction",
            "the lemma", "a lemma", "formal proof",
            "step by step reasoning", "chain of thought", "analyze and compare",
            "root cause analysis", "logical deduction",
            "the hypothesis", "test the hypothesis"
        }) >= 1) {
        return TaskType::Reasoning;
    }

    // Architecture: system and API design, major structural decisions.
    if (count_matches(prompt_lower, {
            "architecture", "system design", "microservice", "monolith",
            "api design", "schema design", "database schema", "infrastructure",
            "design pattern", "migration plan", "refactor strategy",
            "high-level design", "production deployment", "scalability",
            "distributed system"
        }) >= 1) {
        return TaskType::Architecture;
    }

    // Debugging: anything that involves finding or fixing errors.
    if (input.has_stack_trace || input.has_error_message
        || count_matches(prompt_lower, {
               "debug", "fix this", "why does", "why is it", "not working",
               "broken", "crash", "segfault", "exception", "stack trace",
               "traceback", "assertion failed", "undefined behaviour",
               "memory leak", "race condition", "deadlock", "hang",
               "infinite loop", "null pointer", "out of bounds"
           }) >= 1) {
        return TaskType::Debugging;
    }

    // CodeGen: any request to write, extend, or explain specific code.
    if (input.has_code_block
        || count_matches(prompt_lower, {
               "write a function", "write a ", "write the ",
               "implement", "create a class", "add a method",
               "write code", "write a test", "generate code", "write unit test",
               "add endpoint", "write a script", "complete the following",
               "refactor", "optimise", "translate to", "convert this code",
               "add a ", "edit the ", "modify the ",
               "#include", "def ", "fn ", "func ", "class "
           }) >= 1) {
        return TaskType::CodeGen;
    }

    // Simple: short question or look-up.
    if (prompt_lower.size() < 200
        && count_matches(prompt_lower, {
               "what is", "what are", "who is", "when did", "where is",
               "list ", "show me", "tell me", "give me"
           }) >= 1) {
        return TaskType::Simple;
    }

    // Default: CodeGen for medium-length prompts; Simple for short ones.
    return (prompt_lower.size() > 150) ? TaskType::CodeGen : TaskType::Simple;
}

double AutoClassifier::compute_complexity(TaskType            task_type,
                                           const ClassificationInput& input,
                                           std::string_view    prompt_lower) const noexcept {
    // ── Base factor from task type ────────────────────────────────────────────
    double task_factor = kFactorSimple;
    switch (task_type) {
        case TaskType::Trivial:      task_factor = kFactorTrivial;      break;
        case TaskType::Simple:       task_factor = kFactorSimple;       break;
        case TaskType::CodeGen:      task_factor = kFactorCodeGen;      break;
        case TaskType::Debugging:    task_factor = kFactorDebugging;    break;
        case TaskType::Architecture: task_factor = kFactorArchitecture; break;
        case TaskType::Reasoning:    task_factor = kFactorReasoning;    break;
    }

    // ── Length score ─────────────────────────────────────────────────────────
    // Scale prompt tokens relative to the "powerful" threshold.
    const double token_ratio = static_cast<double>(
        input.prompt_tokens + input.history_tokens / std::size_t{2})
        / static_cast<double>(config_.powerful_token_threshold);
    const double length_score = std::clamp(token_ratio, 0.0, 1.0);

    // ── Content boosts ───────────────────────────────────────────────────────
    double content_boost = 0.0;
    if (input.has_stack_trace)    content_boost += 0.10;
    if (input.has_error_message)  content_boost += 0.05;
    if (input.has_code_block)     content_boost += 0.05;
    if (input.has_tool_history)   content_boost += 0.08;

    // ── Keyword complexity boost ──────────────────────────────────────────────
    // Each additional high-complexity keyword adds 0.05, capped at 0.15.
    const int kw_count = count_matches(prompt_lower, {
        "architecture", "root cause", "investigate", "complex",
        "multi-step", "reasoning", "design", "production",
        "distributed", "concurrent", "async", "race condition",
        "performance", "scalability", "security", "migration"
    });
    const double keyword_boost = std::min(static_cast<double>(kw_count) * 0.05, 0.15);

    // ── Conversation turn escalation ─────────────────────────────────────────
    // Long conversations accumulate context and typically become harder.
    const int excess_turns = std::max(0, input.turn_count - config_.escalation_turn_threshold);
    const double turn_boost = std::clamp(static_cast<double>(excess_turns) * 0.04, 0.0, 0.20);

    // ── Fuse signals ─────────────────────────────────────────────────────────
    // task_factor drives the signal; length_score adds up to 0.6 of its own weight,
    // but only if length dominates the task factor.
    const double base = std::max(task_factor, length_score * 0.6);
    const double raw  = base + content_boost + turn_boost + keyword_boost;
    return std::clamp(raw, 0.0, 1.0);
}

Tier AutoClassifier::select_tier(double complexity) const noexcept {
    // Thresholds shift with quality_bias:
    //   bias=0.0 (cost-first):  fast<0.35, balanced<0.70, powerful>=0.70
    //   bias=0.5 (default):     fast<0.22, balanced<0.60, powerful>=0.60
    //   bias=1.0 (quality):     fast<0.10, balanced<0.50, powerful>=0.50
    const double bias         = std::clamp(config_.quality_bias, 0.0, 1.0);
    const double fast_thresh  = 0.35 - 0.25 * bias;
    const double power_thresh = 0.70 - 0.20 * bias;

    if (complexity < fast_thresh)  return Tier::Fast;
    if (complexity < power_thresh) return Tier::Balanced;
    return Tier::Powerful;
}

} // namespace core::llm::routing
