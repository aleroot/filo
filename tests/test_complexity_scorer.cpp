#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/llm/routing/AutoClassifier.hpp"
#include "core/llm/routing/ComplexityScorer.hpp"

using namespace core::llm::routing;

namespace {

[[nodiscard]] std::uint32_t value(const ComplexityFeatureValues& values,
                                  ComplexityFeature feature) {
    return values[static_cast<std::size_t>(feature)];
}

} // namespace

TEST_CASE("ComplexityScorer extracts markdown structure outside code fences",
          "[complexity_scorer]") {
    ComplexityScorer scorer;
    const auto features = scorer.extract_features(R"(
---
title: ignored
---
# Plan

- first
- second

| a | b |
| - | - |

[docs](https://example.com)

```md
# not a heading
- not a list item
| not | a table |
```
)");

    REQUIRE(value(features, ComplexityFeature::HeadingCount) == 1);
    REQUIRE(value(features, ComplexityFeature::MaxHeadingDepth) == 1);
    REQUIRE(value(features, ComplexityFeature::ListItemCount) == 2);
    REQUIRE(value(features, ComplexityFeature::TableRowCount) == 2);
    REQUIRE(value(features, ComplexityFeature::LinkCount) == 1);
    REQUIRE(value(features, ComplexityFeature::CodeBlockCount) == 1);
}

TEST_CASE("ComplexityScorer strips frontmatter like Wayfinder artifacts",
          "[complexity_scorer]") {
    ComplexityScorer scorer;
    const std::string body = "# Task\n\nDo the thing.\n\n## Steps\n\n- one\n- two\n";
    const auto plain = scorer.extract_features(body);
    const auto with_frontmatter = scorer.extract_features(
        "---\nschema_version: 1\nid: WF-TEST-01\n---\n" + body);

    REQUIRE(with_frontmatter == plain);
}

TEST_CASE("ComplexityScorer leaves unterminated frontmatter in place",
          "[complexity_scorer]") {
    ComplexityScorer scorer;
    const auto with_unterminated = scorer.extract_features("---\nstill going\nno closer here\n");
    const auto body_only = scorer.extract_features("still going\nno closer here\n");

    REQUIRE(value(with_unterminated, ComplexityFeature::WordCount)
            > value(body_only, ComplexityFeature::WordCount));
}

TEST_CASE("ComplexityScorer keeps lexical features off by default",
          "[complexity_scorer]") {
    ComplexityScorer scorer;
    const auto easy = scorer.score("hello");
    const auto lexical = scorer.score(
        "prove the theorem exactly without using induction?");

    REQUIRE(value(lexical.features, ComplexityFeature::ReasoningTermCount) > 0);
    REQUIRE(value(lexical.features, ComplexityFeature::ConstraintTermCount) > 0);
    REQUIRE(value(lexical.features, ComplexityFeature::QuestionCount) > 0);
    REQUIRE(lexical.score == Catch::Approx(easy.score).margin(0.05));
}

TEST_CASE("ComplexityScorer counts reasoning terms case-insensitively as whole words",
          "[complexity_scorer]") {
    ComplexityScorer scorer;

    const auto proof = scorer.extract_features(
        "Prove that the square root of 2 is irrational.");
    const auto upper = scorer.extract_features("PROVE THE THEOREM");
    const auto substrings = scorer.extract_features("approve the proverbial change");

    REQUIRE(value(proof, ComplexityFeature::ReasoningTermCount) == 2);
    REQUIRE(value(upper, ComplexityFeature::ReasoningTermCount) == 2);
    REQUIRE(value(substrings, ComplexityFeature::ReasoningTermCount) == 0);
}

TEST_CASE("ComplexityScorer counts Wayfinder math glyphs and LaTeX tokens",
          "[complexity_scorer]") {
    ComplexityScorer scorer;

    const auto latex = scorer.extract_features(R"(Show that $\int x\,dx \le 5$ and \frac{1}{2}.)");
    const auto glyphs = scorer.extract_features("Bound it by ∑ and ∫ where x ≤ y.");

    REQUIRE(value(latex, ComplexityFeature::MathSymbolCount) == 3);
    REQUIRE(value(glyphs, ComplexityFeature::MathSymbolCount) == 3);
}

TEST_CASE("ComplexityScorer counts constraint terms and questions",
          "[complexity_scorer]") {
    ComplexityScorer scorer;
    const auto features = scorer.extract_features(
        "It must run without locks, only once. Done? Sure?");

    REQUIRE(value(features, ComplexityFeature::ConstraintTermCount) == 3);
    REQUIRE(value(features, ComplexityFeature::QuestionCount) == 2);
}

TEST_CASE("ComplexityScorer matches Wayfinder markdown link shape",
          "[complexity_scorer]") {
    ComplexityScorer scorer;
    const auto features = scorer.extract_features(
        "[valid](https://example.com) []() [](https://example.com) [empty]()");

    REQUIRE(value(features, ComplexityFeature::LinkCount) == 1);
}

TEST_CASE("ComplexityScorer accepts calibrated lexical weights",
          "[complexity_scorer]") {
    ComplexityScoringConfig config;
    config.weights = default_complexity_weights();
    config.weights[static_cast<std::size_t>(ComplexityFeature::ReasoningTermCount)] = 5.0;
    config.weights[static_cast<std::size_t>(ComplexityFeature::ConstraintTermCount)] = 2.0;
    config.weights[static_cast<std::size_t>(ComplexityFeature::QuestionCount)] = 1.0;

    ComplexityScorer scorer{config};
    const auto simple = scorer.score("summarize this short paragraph");
    const auto hard = scorer.score(
        "prove the theorem exactly without using induction?");

    REQUIRE(hard.score > simple.score);
}

TEST_CASE("AutoClassifier structural-only mode can route by prompt structure",
          "[complexity_scorer][auto_classifier]") {
    AutoClassifierConfig config;
    config.quality_bias = 0.0;
    config.scoring.mode = ComplexityScoringMode::StructuralOnly;

    AutoClassifier classifier{config};
    const auto input = AutoClassifier::extract_signals(R"(
# Migration Plan

## Constraints

1. Preserve compatibility.
2. Avoid downtime.
3. Add observability.
4. Keep rollback possible.
5. Measure costs.

```cpp
void example();
```

| phase | owner |
| - | - |
| one | platform |
| two | product |

Explain the exact sequence and risks.
)");

    const auto result = classifier.classify(input);
    REQUIRE(result.structural_complexity > 0.10);
    REQUIRE(result.complexity == Catch::Approx(result.structural_complexity));
    REQUIRE(result.reason.find("/s") != std::string::npos);
}

TEST_CASE("AutoClassifier task-only mode skips structural scoring",
          "[complexity_scorer][auto_classifier]") {
    AutoClassifierConfig config;
    config.scoring.mode = ComplexityScoringMode::TaskOnly;

    AutoClassifier classifier{config};
    const auto input = AutoClassifier::extract_signals(R"(
# Heavy Prompt

## Section

1. first
2. second
3. third

```cpp
void example();
```

| a | b |
| - | - |
| c | d |
)");

    const auto result = classifier.classify(input);
    REQUIRE(result.structural_complexity == 0.0);
    REQUIRE(value(result.structural_features, ComplexityFeature::HeadingCount) == 0);
    REQUIRE(result.complexity == Catch::Approx(result.task_complexity));
}
