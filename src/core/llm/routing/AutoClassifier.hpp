#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace core::llm::routing {

// ─── Task taxonomy ────────────────────────────────────────────────────────────
//
// Ordered roughly by expected model capability required.  The classifier maps
// each request onto one of these types and uses it as the dominant factor in
// the complexity score.
//
enum class TaskType {
    Trivial,       // greetings, one-word replies, status checks    → 0.05
    Simple,        // short look-ups, factual Q&A                   → 0.15
    CodeGen,       // write/modify/explain code                     → 0.40
    Debugging,     // trace bugs, read errors/stack traces           → 0.60
    Architecture,  // system design, multi-file planning            → 0.80
    Reasoning,     // formal logic, math, deep multi-step analysis  → 0.85
};

[[nodiscard]] std::string_view to_string(TaskType task_type) noexcept;

// ─── Tier ─────────────────────────────────────────────────────────────────────
//
// The three routing tiers that candidates are bucketed into.
// RouterEngine::choose_smart() picks the best available candidate in the
// target tier (or the next tier up/down when the preferred tier is empty).
//
enum class Tier { Fast, Balanced, Powerful };

[[nodiscard]] std::string_view to_string(Tier tier) noexcept;

// ─── Inputs ───────────────────────────────────────────────────────────────────

struct ClassificationInput {
    std::string_view prompt;

    // Token counts — rough estimates are fine; the classifier only cares about
    // order of magnitude (hundreds vs thousands).
    std::size_t prompt_tokens   = 0;
    std::size_t history_tokens  = 0;

    // Conversation state
    int  turn_count       = 0;
    bool has_tool_history = false;

    // Content signals extracted from the prompt text
    bool has_code_block     = false; // ```...``` block detected
    bool has_stack_trace    = false; // "at " / "Traceback" / stack frame pattern
    bool has_error_message  = false; // "Error:", "Exception:", "FATAL:", etc.
};

// ─── Config ───────────────────────────────────────────────────────────────────

struct AutoClassifierConfig {
    // 0.0 = always prefer cheapest/fastest model
    // 1.0 = always prefer highest quality model
    // Default 0.5 = balanced cost/quality
    double quality_bias = 0.5;

    // Prompt token count below which length alone does not push complexity up.
    std::size_t fast_token_threshold     = 400;
    // Above this threshold, length contributes significant complexity boost.
    std::size_t powerful_token_threshold = 2000;

    // After this many turns in a conversation the tier escalates by one level,
    // capturing longer-context requests that naturally grow in complexity.
    int escalation_turn_threshold = 6;
};

// ─── Result ───────────────────────────────────────────────────────────────────

struct ClassificationResult {
    TaskType task_type  = TaskType::Simple;
    Tier     tier       = Tier::Balanced;
    double   complexity = 0.0; // [0.0, 1.0]

    // Short human-readable explanation shown in the status bar.
    // E.g. "Debugging·0.72" or "CodeGen·0.41"
    std::string reason;
};

// ─── AutoClassifier ───────────────────────────────────────────────────────────
//
// Pure, stateless classification engine.  All methods are const-qualified and
// noexcept; the classifier never allocates beyond what std::string_view
// operations require, making it safe to call on every request without concern
// for latency jitter.
//
class AutoClassifier {
public:
    explicit AutoClassifier(AutoClassifierConfig config = {}) noexcept;

    // Main entry point.  classify() is pure and deterministic: same input →
    // same output, no side effects, no locking.
    [[nodiscard]] ClassificationResult classify(const ClassificationInput& input) const noexcept;

    // Convenience factory: extract content signals from a raw prompt string.
    // Returns a ClassificationInput ready to pass to classify().
    [[nodiscard]] static ClassificationInput extract_signals(
        std::string_view     prompt,
        std::size_t          history_tokens  = 0,
        int                  turn_count      = 0,
        bool                 has_tool_history = false) noexcept;

    [[nodiscard]] const AutoClassifierConfig& config() const noexcept { return config_; }

private:
    [[nodiscard]] TaskType classify_task_type(std::string_view prompt_lower,
                                              const ClassificationInput& input) const noexcept;

    [[nodiscard]] double compute_complexity(TaskType            task_type,
                                            const ClassificationInput& input,
                                            std::string_view    prompt_lower) const noexcept;

    [[nodiscard]] Tier select_tier(double complexity) const noexcept;

    AutoClassifierConfig config_;
};

} // namespace core::llm::routing
