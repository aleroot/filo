#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace core::llm::routing {

enum class ComplexityFeature : std::uint8_t {
    WordCount,
    HeadingCount,
    MaxHeadingDepth,
    ListItemCount,
    LinkCount,
    CodeBlockCount,
    TableRowCount,
    ReasoningTermCount,
    MathSymbolCount,
    ConstraintTermCount,
    QuestionCount,
    Count,
};

inline constexpr std::size_t kComplexityFeatureCount =
    static_cast<std::size_t>(ComplexityFeature::Count);

using ComplexityFeatureValues = std::array<std::uint32_t, kComplexityFeatureCount>;
using ComplexityWeights = std::array<double, kComplexityFeatureCount>;

[[nodiscard]] std::string_view to_string(ComplexityFeature feature) noexcept;
[[nodiscard]] std::optional<ComplexityFeature>
complexity_feature_from_string(std::string_view value) noexcept;

enum class ComplexityScoringMode {
    TaskOnly,
    StructuralOnly,
    Hybrid,
};

[[nodiscard]] std::string_view to_string(ComplexityScoringMode mode) noexcept;
[[nodiscard]] std::optional<ComplexityScoringMode>
complexity_scoring_mode_from_string(std::string_view value) noexcept;

struct ComplexityFeatureContribution {
    ComplexityFeature feature = ComplexityFeature::WordCount;
    std::uint32_t value = 0;
    double normalized = 0.0;
    double weight = 0.0;
    double contribution = 0.0;
};

struct ComplexityScoringConfig {
    ComplexityScoringMode mode = ComplexityScoringMode::Hybrid;
    ComplexityWeights weights{};
};

struct ComplexityScore {
    ComplexityFeatureValues features{};
    double score = 0.0;
};

[[nodiscard]] ComplexityWeights default_complexity_weights() noexcept;
[[nodiscard]] std::span<const double, kComplexityFeatureCount>
complexity_feature_saturation() noexcept;

class ComplexityScorer {
public:
    explicit ComplexityScorer(ComplexityScoringConfig config = {}) noexcept;

    [[nodiscard]] ComplexityScore score(std::string_view text) const;
    [[nodiscard]] ComplexityFeatureValues extract_features(std::string_view text) const;
    [[nodiscard]] double scalar_score(const ComplexityFeatureValues& features) const noexcept;

    [[nodiscard]] std::array<ComplexityFeatureContribution, kComplexityFeatureCount>
    explain(const ComplexityFeatureValues& features) const noexcept;

    [[nodiscard]] const ComplexityScoringConfig& config() const noexcept { return config_; }

private:
    ComplexityScoringConfig config_;
};

} // namespace core::llm::routing
