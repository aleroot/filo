#include "ComplexityScorer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>

namespace core::llm::routing {

namespace {

using enum ComplexityFeature;

constexpr std::array<std::string_view, kComplexityFeatureCount> kFeatureNames{
    "word_count",
    "heading_count",
    "max_heading_depth",
    "list_item_count",
    "link_count",
    "code_block_count",
    "table_row_count",
    "reasoning_term_count",
    "math_symbol_count",
    "constraint_term_count",
    "question_count",
};

constexpr ComplexityWeights kDefaultWeights{
    3.0, // word_count
    1.5, // heading_count
    1.0, // max_heading_depth
    2.0, // list_item_count
    1.0, // link_count
    1.5, // code_block_count
    1.0, // table_row_count
    0.0, // reasoning_term_count, opt-in after calibration
    0.0, // math_symbol_count, opt-in after calibration
    0.0, // constraint_term_count, opt-in after calibration
    0.0, // question_count, opt-in after calibration
};

constexpr std::array<double, kComplexityFeatureCount> kSaturation{
    400.0, // word_count
    8.0,   // heading_count
    4.0,   // max_heading_depth
    15.0,  // list_item_count
    10.0,  // link_count
    4.0,   // code_block_count
    12.0,  // table_row_count
    2.0,   // reasoning_term_count
    6.0,   // math_symbol_count
    3.0,   // constraint_term_count
    3.0,   // question_count
};

constexpr auto kReasoningTerms = std::to_array<std::string_view>({
    "asymptotic", "axiom", "axioms", "bijection", "combinatorial",
    "complexity", "concurrency", "concurrent", "contradiction", "corollary",
    "deadlock", "decidable", "derivation", "derivative", "derive", "derives",
    "eigenvalue", "eigenvalues", "halting", "induction", "infinitely",
    "injective", "integral", "invariant", "invariants", "irrational",
    "isomorphism", "lemma", "lemmas", "maximise", "maximize", "minimise",
    "minimize", "modulo", "monotonic", "optimal", "optimality", "optimise",
    "optimize", "polynomial", "prime", "primes", "proof", "proofs",
    "prove", "proven", "recurrence", "surjective", "theorem", "theorems",
    "undecidability", "undecidable",
});

constexpr auto kConstraintTerms = std::to_array<std::string_view>({
    "constraint", "constraints", "ensure", "exactly", "guarantee", "must",
    "only", "preserve", "preserving", "subject", "without",
});

[[nodiscard]] constexpr std::size_t index(ComplexityFeature feature) noexcept {
    return static_cast<std::size_t>(feature);
}

[[nodiscard]] bool is_space(char ch) noexcept {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

[[nodiscard]] bool is_alpha(char ch) noexcept {
    return std::isalpha(static_cast<unsigned char>(ch)) != 0;
}

[[nodiscard]] bool is_digit(char ch) noexcept {
    return std::isdigit(static_cast<unsigned char>(ch)) != 0;
}

[[nodiscard]] std::string normalize_token(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        if (ch == '-' || ch == '_' || ch == ' ') continue;
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

[[nodiscard]] char ascii_lower(char ch) noexcept {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
}

[[nodiscard]] int compare_ascii_ci(std::string_view lhs, std::string_view rhs) noexcept {
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < limit; ++i) {
        const char left = ascii_lower(lhs[i]);
        const char right = ascii_lower(rhs[i]);
        if (left < right) return -1;
        if (left > right) return 1;
    }
    if (lhs.size() < rhs.size()) return -1;
    if (lhs.size() > rhs.size()) return 1;
    return 0;
}

[[nodiscard]] std::string_view trim_line(std::string_view line) noexcept {
    std::size_t begin = 0;
    while (begin < line.size() && is_space(line[begin])) ++begin;
    std::size_t end = line.size();
    while (end > begin && is_space(line[end - 1])) --end;
    return line.substr(begin, end - begin);
}

[[nodiscard]] bool starts_code_fence(std::string_view trimmed) noexcept {
    return trimmed.starts_with("```") || trimmed.starts_with("~~~");
}

[[nodiscard]] std::optional<std::uint32_t> markdown_heading_depth(
    std::string_view trimmed) noexcept {
    std::uint32_t depth = 0;
    while (depth < trimmed.size() && depth < 6 && trimmed[depth] == '#') {
        ++depth;
    }
    if (depth == 0 || depth >= trimmed.size() || !is_space(trimmed[depth])) {
        return std::nullopt;
    }
    for (std::size_t i = depth + 1; i < trimmed.size(); ++i) {
        if (!is_space(trimmed[i])) return depth;
    }
    return std::nullopt;
}

[[nodiscard]] bool is_list_item(std::string_view trimmed) noexcept {
    if (trimmed.size() < 3) return false;

    const char first = trimmed.front();
    if ((first == '-' || first == '*' || first == '+')
        && is_space(trimmed[1])
        && !trim_line(trimmed.substr(2)).empty()) {
        return true;
    }

    std::size_t i = 0;
    while (i < trimmed.size() && is_digit(trimmed[i])) ++i;
    if (i == 0 || i + 1 >= trimmed.size()) return false;
    if (trimmed[i] != '.' && trimmed[i] != ')') return false;
    return is_space(trimmed[i + 1]) && !trim_line(trimmed.substr(i + 2)).empty();
}

[[nodiscard]] bool is_table_row(std::string_view trimmed) noexcept {
    return trimmed.size() >= 2 && trimmed.front() == '|' && trimmed.back() == '|';
}

[[nodiscard]] std::uint32_t count_markdown_links(std::string_view line) noexcept {
    std::uint32_t count = 0;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] != '[') continue;
        const auto close_bracket = line.find(']', i + 1);
        if (close_bracket == std::string_view::npos
            || close_bracket == i + 1
            || close_bracket + 1 >= line.size()
            || line[close_bracket + 1] != '(') {
            continue;
        }
        const auto close_paren = line.find(')', close_bracket + 2);
        if (close_paren == std::string_view::npos || close_paren == close_bracket + 2) {
            continue;
        }
        ++count;
        i = close_paren;
    }
    return count;
}

[[nodiscard]] std::string_view strip_frontmatter(std::string_view text) noexcept {
    const auto first_line_end = text.find('\n');
    const auto first_line = text.substr(
        0,
        first_line_end == std::string_view::npos ? text.size() : first_line_end);
    if (trim_line(first_line) != "---") return text;

    std::size_t line_begin =
        first_line_end == std::string_view::npos ? text.size() : first_line_end + 1;
    while (line_begin < text.size()) {
        const auto line_end = text.find('\n', line_begin);
        const auto line = text.substr(
            line_begin,
            line_end == std::string_view::npos ? std::string_view::npos : line_end - line_begin);
        const auto trimmed = trim_line(line);
        if (trimmed == "---" || trimmed == "...") {
            return line_end == std::string_view::npos
                       ? std::string_view{}
                       : text.substr(line_end + 1);
        }
        if (line_end == std::string_view::npos) break;
        line_begin = line_end + 1;
    }
    return text;
}

template <std::size_t N>
[[nodiscard]] bool contains_term(const std::array<std::string_view, N>& terms,
                                 std::string_view token) noexcept {
    const auto it = std::lower_bound(
        terms.begin(),
        terms.end(),
        token,
        [](std::string_view term, std::string_view value) noexcept {
            return compare_ascii_ci(term, value) < 0;
        });
    return it != terms.end() && compare_ascii_ci(*it, token) == 0;
}

[[nodiscard]] bool is_math_symbol_start(unsigned char ch) noexcept {
    // LaTeX commands plus UTF-8 lead bytes used by Wayfinder's math/logic glyph set.
    return ch == '\\' || ch == 0xCE || ch == 0xCF || ch == 0xE2;
}

[[nodiscard]] std::size_t math_glyph_length(std::string_view text,
                                            std::size_t offset) noexcept {
    static constexpr std::array<std::string_view, 24> kMathGlyphs{
        "∑", "∫", "√", "≤", "≥", "≠", "≈", "∞", "∂", "∈", "∉", "∀",
        "∃", "⊆", "⊂", "∪", "∩", "∇", "±", "×", "÷", "π", "θ", "λ",
    };
    for (const auto glyph : kMathGlyphs) {
        if (offset + glyph.size() <= text.size()
            && std::memcmp(text.data() + offset, glyph.data(), glyph.size()) == 0) {
            return glyph.size();
        }
    }
    // μ, σ, Σ, Π are two-byte UTF-8 sequences and share the Greek lead byte.
    static constexpr std::array<std::string_view, 4> kTwoByteGreekMathGlyphs{
        "μ", "σ", "Σ", "Π",
    };
    for (const auto glyph : kTwoByteGreekMathGlyphs) {
        if (offset + glyph.size() <= text.size()
            && std::memcmp(text.data() + offset, glyph.data(), glyph.size()) == 0) {
            return glyph.size();
        }
    }
    return 0;
}

void scan_text_features(std::string_view body, ComplexityFeatureValues& out) {
    std::size_t i = 0;
    bool in_word = false;
    while (i < body.size()) {
        const unsigned char ch = static_cast<unsigned char>(body[i]);

        if (is_space(body[i])) {
            in_word = false;
        } else if (!in_word) {
            in_word = true;
            ++out[index(WordCount)];
        }

        if (body[i] == '?') {
            ++out[index(QuestionCount)];
            ++i;
            continue;
        }

        if (is_math_symbol_start(ch)) {
            if (body[i] == '\\') {
                std::size_t j = i + 1;
                while (j < body.size() && is_alpha(body[j])) ++j;
                if (j > i + 1) ++out[index(MathSymbolCount)];
                i = std::max(j, i + 1);
                continue;
            }
            if (const std::size_t glyph_len = math_glyph_length(body, i); glyph_len > 0) {
                ++out[index(MathSymbolCount)];
                i += glyph_len;
                continue;
            }
        }

        if (!is_alpha(body[i])) {
            ++i;
            continue;
        }

        const std::size_t start = i;
        while (i < body.size()
               && (is_alpha(body[i]) || body[i] == '\'' || body[i] == '-')) {
            ++i;
        }
        const std::string_view token = body.substr(start, i - start);
        if (contains_term(kReasoningTerms, token)) {
            ++out[index(ReasoningTermCount)];
        }
        if (contains_term(kConstraintTerms, token)) {
            ++out[index(ConstraintTermCount)];
        }
    }
}

} // namespace

std::string_view to_string(ComplexityFeature feature) noexcept {
    const auto i = index(feature);
    if (i < kFeatureNames.size()) return kFeatureNames[i];
    return "unknown";
}

std::optional<ComplexityFeature> complexity_feature_from_string(std::string_view value) noexcept {
    const std::string token = normalize_token(value);
    for (std::size_t i = 0; i < kFeatureNames.size(); ++i) {
        if (normalize_token(kFeatureNames[i]) == token) {
            return static_cast<ComplexityFeature>(i);
        }
    }
    return std::nullopt;
}

std::string_view to_string(ComplexityScoringMode mode) noexcept {
    switch (mode) {
        case ComplexityScoringMode::TaskOnly: return "task";
        case ComplexityScoringMode::StructuralOnly: return "structural";
        case ComplexityScoringMode::Hybrid: return "hybrid";
    }
    return "hybrid";
}

std::optional<ComplexityScoringMode>
complexity_scoring_mode_from_string(std::string_view value) noexcept {
    const std::string token = normalize_token(value);
    if (token == "task" || token == "taskonly") return ComplexityScoringMode::TaskOnly;
    if (token == "structural" || token == "structuralonly" || token == "wayfinder") {
        return ComplexityScoringMode::StructuralOnly;
    }
    if (token == "hybrid") return ComplexityScoringMode::Hybrid;
    return std::nullopt;
}

ComplexityWeights default_complexity_weights() noexcept {
    return kDefaultWeights;
}

std::span<const double, kComplexityFeatureCount> complexity_feature_saturation() noexcept {
    return std::span<const double, kComplexityFeatureCount>{kSaturation};
}

ComplexityScorer::ComplexityScorer(ComplexityScoringConfig config) noexcept
    : config_(config) {
    if (std::ranges::all_of(config_.weights, [](double weight) { return weight == 0.0; })) {
        config_.weights = kDefaultWeights;
    }
}

ComplexityScore ComplexityScorer::score(std::string_view text) const {
    ComplexityScore result;
    result.features = extract_features(text);
    result.score = scalar_score(result.features);
    return result;
}

ComplexityFeatureValues ComplexityScorer::extract_features(std::string_view text) const {
    ComplexityFeatureValues out{};
    const std::string_view body = strip_frontmatter(text);

    bool in_fence = false;
    std::size_t line_begin = 0;
    while (line_begin <= body.size()) {
        const auto line_end = body.find('\n', line_begin);
        const auto line = body.substr(
            line_begin,
            line_end == std::string_view::npos ? body.size() - line_begin : line_end - line_begin);
        const auto trimmed = trim_line(line);

        if (starts_code_fence(trimmed)) {
            if (!in_fence) ++out[index(CodeBlockCount)];
            in_fence = !in_fence;
        } else if (!in_fence) {
            if (const auto depth = markdown_heading_depth(trimmed); depth.has_value()) {
                ++out[index(HeadingCount)];
                out[index(MaxHeadingDepth)] = std::max(out[index(MaxHeadingDepth)], *depth);
            } else if (is_list_item(trimmed)) {
                ++out[index(ListItemCount)];
            } else if (is_table_row(trimmed)) {
                ++out[index(TableRowCount)];
            }
            out[index(LinkCount)] += count_markdown_links(line);
        }

        if (line_end == std::string_view::npos) break;
        line_begin = line_end + 1;
    }

    scan_text_features(body, out);
    return out;
}

double ComplexityScorer::scalar_score(const ComplexityFeatureValues& features) const noexcept {
    double total_weight = 0.0;
    double accumulated = 0.0;
    for (std::size_t i = 0; i < features.size(); ++i) {
        const double weight = std::max(0.0, config_.weights[i]);
        total_weight += weight;
        const double normalized =
            std::min(static_cast<double>(features[i]) / kSaturation[i], 1.0);
        accumulated += weight * normalized;
    }
    if (total_weight <= 0.0) return 0.0;
    return std::round((accumulated / total_weight) * 100.0) / 100.0;
}

std::array<ComplexityFeatureContribution, kComplexityFeatureCount>
ComplexityScorer::explain(const ComplexityFeatureValues& features) const noexcept {
    std::array<ComplexityFeatureContribution, kComplexityFeatureCount> out{};
    double total_weight = 0.0;
    for (const double weight : config_.weights) {
        total_weight += std::max(0.0, weight);
    }

    for (std::size_t i = 0; i < out.size(); ++i) {
        const double weight = std::max(0.0, config_.weights[i]);
        const double normalized =
            std::min(static_cast<double>(features[i]) / kSaturation[i], 1.0);
        out[i] = ComplexityFeatureContribution{
            .feature = static_cast<ComplexityFeature>(i),
            .value = features[i],
            .normalized = normalized,
            .weight = weight,
            .contribution = total_weight > 0.0 ? (weight * normalized / total_weight) : 0.0,
        };
    }
    return out;
}

} // namespace core::llm::routing
