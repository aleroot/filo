#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/llm/routing/AutoClassifier.hpp"

using namespace core::llm::routing;
using Catch::Matchers::WithinAbs;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static ClassificationResult classify(std::string_view prompt,
                                      AutoClassifierConfig cfg = {},
                                      int turn_count = 0,
                                      std::size_t history_tokens = 0,
                                      bool has_tool_history = false) {
    AutoClassifier ac{cfg};
    auto input = AutoClassifier::extract_signals(prompt, history_tokens, turn_count, has_tool_history);
    return ac.classify(input);
}

// ─── extract_signals ─────────────────────────────────────────────────────────

TEST_CASE("AutoClassifier: extract_signals detects code fences", "[auto_classifier]") {
    const auto input = AutoClassifier::extract_signals(
        "Here is my code:\n```cpp\nint x = 0;\n```\nPlease review.");
    REQUIRE(input.has_code_block);
    REQUIRE_FALSE(input.has_stack_trace);
    REQUIRE_FALSE(input.has_error_message);
}

TEST_CASE("AutoClassifier: extract_signals detects stack trace markers", "[auto_classifier]") {
    const auto input = AutoClassifier::extract_signals(
        "I got this error:\n  at MyClass.method(MyClass.java:42)\n  at main(Main.java:10)");
    REQUIRE(input.has_stack_trace);
}

TEST_CASE("AutoClassifier: extract_signals detects Python traceback", "[auto_classifier]") {
    const auto input = AutoClassifier::extract_signals(
        "Traceback (most recent call last):\n  File foo.py, line 5\nValueError: bad input");
    REQUIRE(input.has_stack_trace);
    REQUIRE(input.has_error_message);
}

TEST_CASE("AutoClassifier: extract_signals detects error message prefixes", "[auto_classifier]") {
    const auto input = AutoClassifier::extract_signals("Error: connection refused");
    REQUIRE(input.has_error_message);
    REQUIRE_FALSE(input.has_code_block);
}

TEST_CASE("AutoClassifier: extract_signals preserves turn_count and history_tokens",
          "[auto_classifier]") {
    const auto input = AutoClassifier::extract_signals("hello", 5000, 8, true);
    REQUIRE(input.history_tokens == 5000);
    REQUIRE(input.turn_count == 8);
    REQUIRE(input.has_tool_history);
}

// ─── Task type classification ─────────────────────────────────────────────────

TEST_CASE("AutoClassifier: trivial greeting classifies as Trivial", "[auto_classifier]") {
    const auto r = classify("Hello!");
    REQUIRE(r.task_type == TaskType::Trivial);
}

TEST_CASE("AutoClassifier: thank you classifies as Trivial", "[auto_classifier]") {
    const auto r = classify("thanks");
    REQUIRE(r.task_type == TaskType::Trivial);
}

TEST_CASE("AutoClassifier: debugging error request classifies as Debugging", "[auto_classifier]") {
    const auto r = classify("I keep getting a segfault when I call free(ptr), how do I debug it?");
    REQUIRE(r.task_type == TaskType::Debugging);
}

TEST_CASE("AutoClassifier: stack trace in prompt classifies as Debugging", "[auto_classifier]") {
    const auto input = AutoClassifier::extract_signals(
        "Why is this failing?\n  at Foo.bar(Foo.java:12)\n  at Main.main(Main.java:5)");
    AutoClassifier ac;
    const auto r = ac.classify(input);
    REQUIRE(r.task_type == TaskType::Debugging);
}

TEST_CASE("AutoClassifier: implement function classifies as CodeGen", "[auto_classifier]") {
    const auto r = classify("write a function that reverses a linked list in C++");
    REQUIRE(r.task_type == TaskType::CodeGen);
}

TEST_CASE("AutoClassifier: code block in prompt classifies as CodeGen", "[auto_classifier]") {
    const auto input = AutoClassifier::extract_signals(
        "Review this:\n```python\ndef foo():\n    pass\n```\nand suggest improvements");
    AutoClassifier ac;
    const auto r = ac.classify(input);
    REQUIRE(r.task_type == TaskType::CodeGen);
}

TEST_CASE("AutoClassifier: system design query classifies as Architecture", "[auto_classifier]") {
    const auto r = classify("design a microservice architecture for a high-traffic e-commerce system");
    REQUIRE(r.task_type == TaskType::Architecture);
}

TEST_CASE("AutoClassifier: formal proof classifies as Reasoning", "[auto_classifier]") {
    // "prove that" is a specific phrase that triggers Reasoning
    const auto r = classify("prove that there are infinitely many primes using Euclid's argument");
    REQUIRE(r.task_type == TaskType::Reasoning);
}

TEST_CASE("AutoClassifier: short factual question classifies as Simple", "[auto_classifier]") {
    const auto r = classify("what is the capital of France?");
    REQUIRE(r.task_type == TaskType::Simple);
}

// ─── Complexity values ────────────────────────────────────────────────────────

TEST_CASE("AutoClassifier: trivial prompt has very low complexity", "[auto_classifier]") {
    const auto r = classify("hi");
    REQUIRE(r.complexity < 0.20);
}

TEST_CASE("AutoClassifier: architecture prompt has high complexity", "[auto_classifier]") {
    const auto r = classify("design a distributed system architecture for a global banking platform with multi-region failover and GDPR compliance");
    REQUIRE(r.complexity > 0.60);
}

TEST_CASE("AutoClassifier: reasoning prompt has high complexity", "[auto_classifier]") {
    const auto r = classify("prove by mathematical induction that the sum of the first n odd numbers equals n squared");
    REQUIRE(r.complexity > 0.70);
}

TEST_CASE("AutoClassifier: complexity is bounded in [0, 1]", "[auto_classifier]") {
    AutoClassifier ac;
    // Maximally complex input
    ClassificationInput input;
    input.prompt          = "architecture distributed system design reasoning complex production migration refactor multi-step";
    input.prompt_tokens   = 100000;
    input.history_tokens  = 100000;
    input.turn_count      = 100;
    input.has_tool_history   = true;
    input.has_stack_trace    = true;
    input.has_error_message  = true;
    input.has_code_block     = true;
    const auto r = ac.classify(input);
    REQUIRE(r.complexity >= 0.0);
    REQUIRE(r.complexity <= 1.0);
}

// ─── Tier selection ───────────────────────────────────────────────────────────

TEST_CASE("AutoClassifier: trivial → Fast tier at default bias", "[auto_classifier]") {
    const auto r = classify("hi");
    REQUIRE(r.tier == Tier::Fast);
}

TEST_CASE("AutoClassifier: simple short question → Fast or Balanced at default bias",
          "[auto_classifier]") {
    const auto r = classify("what is the capital of Germany?");
    // Simple task with low complexity should land in Fast or Balanced
    REQUIRE((r.tier == Tier::Fast || r.tier == Tier::Balanced));
}

TEST_CASE("AutoClassifier: architecture prompt → Powerful tier at default bias",
          "[auto_classifier]") {
    const auto r = classify(
        "design a microservice architecture for a high-traffic distributed production system"
        " with CQRS, event sourcing, multi-region failover, and GDPR compliance");
    REQUIRE(r.tier == Tier::Powerful);
}

// ─── Quality bias ─────────────────────────────────────────────────────────────

TEST_CASE("AutoClassifier: quality_bias=0 (cost-first) keeps simple requests on Fast",
          "[auto_classifier]") {
    AutoClassifierConfig cfg;
    cfg.quality_bias = 0.0;
    const auto r = classify("what is 2 + 2?", cfg);
    // At bias=0 the fast threshold is 0.35, so Simple/Trivial should be Fast
    REQUIRE(r.tier == Tier::Fast);
}

TEST_CASE("AutoClassifier: quality_bias=1 (quality-first) escalates moderate complexity",
          "[auto_classifier]") {
    AutoClassifierConfig cfg;
    cfg.quality_bias = 1.0;
    // A CodeGen request that might be Balanced at bias=0.5 should become Powerful at bias=1
    const auto r = classify(
        "implement a thread-safe lock-free queue in C++ with detailed comments", cfg);
    // At bias=1 the powerful threshold is 0.50, so complexity >= 0.5 → Powerful
    REQUIRE((r.tier == Tier::Balanced || r.tier == Tier::Powerful));
    // At very high bias we should definitely see at least Balanced
    REQUIRE(r.tier != Tier::Fast);
}

TEST_CASE("AutoClassifier: quality_bias shifts fast threshold down",
          "[auto_classifier]") {
    AutoClassifierConfig low_bias;
    low_bias.quality_bias = 0.0;
    AutoClassifierConfig high_bias;
    high_bias.quality_bias = 1.0;

    // With a borderline Simple prompt, high bias → higher tier
    const std::string prompt = "list the SOLID principles";
    const auto r_low  = classify(prompt, low_bias);
    const auto r_high = classify(prompt, high_bias);

    // high bias complexity result should be >= low bias tier in terms of quality
    const auto tier_value = [](Tier t) { return static_cast<int>(t); };
    REQUIRE(tier_value(r_high.tier) >= tier_value(r_low.tier));
}

// ─── Turn count escalation ────────────────────────────────────────────────────

TEST_CASE("AutoClassifier: many turns escalate complexity", "[auto_classifier]") {
    // Same simple prompt, but after many turns
    const auto r_fresh = classify("fix this bug", {}, 0);
    const auto r_long  = classify("fix this bug", {}, 20);

    // After 20 turns the turn_boost should push complexity up
    REQUIRE(r_long.complexity > r_fresh.complexity);
}

TEST_CASE("AutoClassifier: turn escalation does not exceed 0.20 boost", "[auto_classifier]") {
    // Measure the difference between 0 and 100 turns
    const auto r0   = classify("hi", {}, 0);
    const auto r100 = classify("hi", {}, 100);
    REQUIRE(r100.complexity - r0.complexity <= 0.21); // ≤ 0.20 + floating point tolerance
}

// ─── Content boost signals ────────────────────────────────────────────────────

TEST_CASE("AutoClassifier: stack trace boosts complexity above baseline", "[auto_classifier]") {
    const auto plain = AutoClassifier::extract_signals("why does my code fail?");
    const auto with_trace = AutoClassifier::extract_signals(
        "why does my code fail?\n  at Foo.run(Foo.java:5)\n  at Main.main(Main.java:1)");

    AutoClassifier ac;
    const auto r_plain = ac.classify(plain);
    const auto r_trace = ac.classify(with_trace);

    REQUIRE(r_trace.complexity > r_plain.complexity);
}

TEST_CASE("AutoClassifier: tool history boosts complexity", "[auto_classifier]") {
    const auto no_tools = AutoClassifier::extract_signals("what is the next step?", 0, 3, false);
    const auto with_tools = AutoClassifier::extract_signals("what is the next step?", 0, 3, true);

    AutoClassifier ac;
    REQUIRE(ac.classify(with_tools).complexity > ac.classify(no_tools).complexity);
}

// ─── History token influence ─────────────────────────────────────────────────

TEST_CASE("AutoClassifier: large history tokens push complexity up", "[auto_classifier]") {
    const auto short_hist = classify("next step?", {}, 2, 100);
    const auto long_hist  = classify("next step?", {}, 2, 50000);

    REQUIRE(long_hist.complexity > short_hist.complexity);
}

// ─── Result formatting ────────────────────────────────────────────────────────

TEST_CASE("AutoClassifier: reason string contains task type and complexity", "[auto_classifier]") {
    const auto r = classify("write a recursive fibonacci function");
    REQUIRE(r.reason.find("CodeGen") != std::string::npos);
    // Must contain a decimal point (the complexity float)
    REQUIRE(r.reason.find('.') != std::string::npos);
}

// ─── to_string helpers ────────────────────────────────────────────────────────

TEST_CASE("AutoClassifier: to_string(TaskType) returns expected labels", "[auto_classifier]") {
    REQUIRE(to_string(TaskType::Trivial)      == "Trivial");
    REQUIRE(to_string(TaskType::Simple)       == "Simple");
    REQUIRE(to_string(TaskType::CodeGen)      == "CodeGen");
    REQUIRE(to_string(TaskType::Debugging)    == "Debugging");
    REQUIRE(to_string(TaskType::Architecture) == "Architecture");
    REQUIRE(to_string(TaskType::Reasoning)    == "Reasoning");
}

TEST_CASE("AutoClassifier: to_string(Tier) returns expected labels", "[auto_classifier]") {
    REQUIRE(to_string(Tier::Fast)     == "fast");
    REQUIRE(to_string(Tier::Balanced) == "balanced");
    REQUIRE(to_string(Tier::Powerful) == "powerful");
}
