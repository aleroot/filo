#include "Findings.hpp"

#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <simdjson.h>
#include <unordered_map>
#include <utility>

namespace core::review {
namespace {

using core::utils::str::trim_ascii_copy;
using core::utils::str::to_lower_ascii_copy;

[[nodiscard]] std::string join(std::string_view separator,
                               const std::vector<std::string>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += separator;
        out += values[i];
    }
    return out;
}

[[nodiscard]] std::optional<std::string>
object_string_field(const simdjson::dom::object& object, const char* key) {
    std::string_view value;
    if (object[key].get(value) == simdjson::SUCCESS) {
        return std::string(value);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::int64_t>
element_to_int64(const simdjson::dom::element& element) {
    std::int64_t i = 0;
    if (element.get(i) == simdjson::SUCCESS) {
        return i;
    }

    std::uint64_t u = 0;
    if (element.get(u) == simdjson::SUCCESS) {
        if (u <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return static_cast<std::int64_t>(u);
        }
        return std::nullopt;
    }

    double d = 0.0;
    if (element.get(d) == simdjson::SUCCESS) {
        return static_cast<std::int64_t>(d);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<double>
element_to_double(const simdjson::dom::element& element) {
    double d = 0.0;
    if (element.get(d) == simdjson::SUCCESS) {
        return d;
    }

    std::int64_t i = 0;
    if (element.get(i) == simdjson::SUCCESS) {
        return static_cast<double>(i);
    }

    std::uint64_t u = 0;
    if (element.get(u) == simdjson::SUCCESS) {
        return static_cast<double>(u);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::int64_t>
object_int_field(const simdjson::dom::object& object, const char* key) {
    simdjson::dom::element element;
    if (object[key].get(element) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return element_to_int64(element);
}

[[nodiscard]] std::optional<double>
object_double_field(const simdjson::dom::object& object, const char* key) {
    simdjson::dom::element element;
    if (object[key].get(element) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return element_to_double(element);
}

void apply_defaults(Finding& finding) {
    if (!finding.severity.has_value() && finding.priority.has_value()) {
        finding.severity = severity_from_priority(*finding.priority);
    }
    if (!finding.priority.has_value() && finding.severity.has_value()) {
        finding.priority = severity_rank(*finding.severity);
    }
    if (!finding.category.has_value()) {
        finding.category = Category::Other;
    }
}

[[nodiscard]] bool confidence_too_low(const Finding& finding) noexcept {
    return finding.confidence.has_value()
        && *finding.confidence < kMinFindingConfidence;
}

[[nodiscard]] std::optional<Report> try_parse_report_json(std::string_view text) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(text).get(root) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    simdjson::dom::object object;
    if (root.get(object) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    Report out;
    if (const auto value = object_string_field(object, "overall_correctness"); value.has_value()) {
        out.overall_correctness = *value;
    }
    if (const auto value = object_string_field(object, "overall_explanation"); value.has_value()) {
        out.overall_explanation = *value;
    }
    out.overall_confidence = object_double_field(object, "overall_confidence_score");

    simdjson::dom::array findings;
    if (object["findings"].get(findings) == simdjson::SUCCESS) {
        for (const auto finding_value : findings) {
            simdjson::dom::object finding_object;
            if (finding_value.get(finding_object) != simdjson::SUCCESS) {
                continue;
            }

            Finding finding;
            if (const auto value = object_string_field(finding_object, "title"); value.has_value()) {
                finding.title = *value;
            }
            if (const auto value = object_string_field(finding_object, "body"); value.has_value()) {
                finding.body = *value;
            }
            if (const auto value = object_int_field(finding_object, "priority"); value.has_value()) {
                finding.priority = static_cast<int>(*value);
            }
            finding.confidence = object_double_field(finding_object, "confidence_score");
            if (const auto value = object_string_field(finding_object, "severity"); value.has_value()) {
                finding.severity = severity_from_string(to_lower_ascii_copy(*value));
            }
            if (const auto value = object_string_field(finding_object, "category"); value.has_value()) {
                finding.category = category_from_string(to_lower_ascii_copy(*value));
            }
            simdjson::dom::array steering_references;
            if (finding_object["steering_references"].get(steering_references)
                == simdjson::SUCCESS) {
                constexpr std::size_t kMaxSteeringReferencesPerFinding = 4;
                for (const auto reference_value : steering_references) {
                    if (finding.steering_references.size()
                        >= kMaxSteeringReferencesPerFinding) {
                        break;
                    }
                    simdjson::dom::object reference_object;
                    if (reference_value.get(reference_object) != simdjson::SUCCESS) {
                        continue;
                    }
                    const auto source_id = object_string_field(reference_object, "source_id");
                    const auto rule_excerpt = object_string_field(reference_object, "rule_excerpt");
                    if (!source_id.has_value() || !rule_excerpt.has_value()) continue;
                    finding.steering_references.push_back(SteeringReference{
                        .source_id = *source_id,
                        .rule_excerpt = *rule_excerpt,
                    });
                }
            }

            simdjson::dom::object location_object;
            if (finding_object["code_location"].get(location_object) == simdjson::SUCCESS) {
                if (const auto value = object_string_field(location_object, "absolute_file_path");
                    value.has_value()) {
                    finding.absolute_file_path = *value;
                }

                simdjson::dom::object line_range;
                if (location_object["line_range"].get(line_range) == simdjson::SUCCESS) {
                    if (const auto value = object_int_field(line_range, "start"); value.has_value()) {
                        finding.line_start = static_cast<int>(*value);
                    }
                    if (const auto value = object_int_field(line_range, "end"); value.has_value()) {
                        finding.line_end = static_cast<int>(*value);
                    }
                }
            }

            apply_defaults(finding);
            if (confidence_too_low(finding)) {
                continue;
            }
            out.findings.push_back(std::move(finding));
        }
    }

    return out;
}

[[nodiscard]] std::string normalize_path(std::string_view path) {
    auto out = to_lower_ascii_copy(path);
    std::replace(out.begin(), out.end(), '\\', '/');
    return out;
}

[[nodiscard]] std::string finding_dedupe_key(const Finding& finding) {
    return std::format(
        "{}|{}|{}|{}",
        normalize_path(finding.absolute_file_path),
        finding.line_start,
        finding.line_end,
        to_lower_ascii_copy(trim_ascii_copy(finding.title)));
}

void merge_steering_references(Finding& target, const Finding& source) {
    for (const auto& reference : source.steering_references) {
        const bool present = std::ranges::any_of(
            target.steering_references,
            [&](const SteeringReference& existing) {
                return existing.source_id == reference.source_id
                    && existing.rule_excerpt == reference.rule_excerpt;
            });
        if (!present) target.steering_references.push_back(reference);
    }
}

[[nodiscard]] bool is_incorrect(std::string_view verdict) noexcept {
    const auto lower = to_lower_ascii_copy(verdict);
    return lower.find("incorrect") != std::string::npos
        || lower.find("not correct") != std::string::npos;
}

[[nodiscard]] std::string format_location(const Finding& finding) {
    const std::string path = finding.absolute_file_path.empty()
        ? std::string("<unknown-file>")
        : finding.absolute_file_path;
    return std::format("{}:{}-{}", path, finding.line_start, finding.line_end);
}

[[nodiscard]] std::string format_findings_block(const std::vector<Finding>& findings) {
    std::vector<std::string> lines;
    lines.reserve(findings.size() * 5);

    for (const auto& finding : findings) {
        // Location first, on its own rule, so the reader anchors the comment to
        // a file and line range before reading the prose.
        lines.push_back("");
        lines.push_back(std::format("── {} ──", format_location(finding)));

        std::vector<std::string> tags;
        tags.emplace_back(to_string(effective_severity(finding)));
        if (finding.category.has_value() && *finding.category != Category::Other) {
            tags.emplace_back(to_string(*finding.category));
        }
        const std::string title =
            finding.title.empty() ? std::string("Untitled finding") : finding.title;
        lines.push_back(std::format("[{}] {}", join(" · ", tags), title));

        if (!finding.body.empty()) {
            std::string_view body_view(finding.body);
            std::size_t cursor = 0;
            while (cursor <= body_view.size()) {
                const std::size_t end = body_view.find('\n', cursor);
                const std::string_view line = end == std::string_view::npos
                    ? body_view.substr(cursor)
                    : body_view.substr(cursor, end - cursor);
                lines.emplace_back(line);
                if (end == std::string_view::npos) break;
                cursor = end + 1;
            }
        }
        for (const auto& reference : finding.steering_references) {
            lines.push_back(std::format(
                "Project guidance [{}] ({}): “{}”",
                reference.source_id,
                reference.source_label,
                reference.rule_excerpt));
        }
    }
    return join("\n", lines);
}

} // namespace

std::optional<Report> parse_review_json(std::string_view text) {
    if (auto parsed = try_parse_report_json(text); parsed.has_value()) {
        return parsed;
    }

    if (const auto start = text.find('{'); start != std::string_view::npos) {
        if (const auto end = text.rfind('}'); end != std::string_view::npos && start < end) {
            if (auto parsed = try_parse_report_json(text.substr(start, end - start + 1));
                parsed.has_value()) {
                return parsed;
            }
        }
    }
    return std::nullopt;
}

Report parse_review_output(std::string_view text) {
    if (auto parsed = parse_review_json(text); parsed.has_value()) {
        return *parsed;
    }

    Report fallback;
    fallback.overall_explanation = std::string(text);
    return fallback;
}

std::vector<RiskItem> parse_risk_output(std::string_view text) {
    auto try_parse = [](std::string_view json) -> std::vector<RiskItem> {
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        if (parser.parse(json).get(root) != simdjson::SUCCESS) {
            return {};
        }
        simdjson::dom::object object;
        if (root.get(object) != simdjson::SUCCESS) {
            return {};
        }
        simdjson::dom::array risks;
        if (object["risks"].get(risks) != simdjson::SUCCESS) {
            return {};
        }
        std::vector<RiskItem> out;
        for (const auto value : risks) {
            simdjson::dom::object item;
            if (value.get(item) != simdjson::SUCCESS) continue;
            RiskItem risk;
            if (const auto title = object_string_field(item, "title"); title.has_value()) {
                risk.title = *title;
            }
            if (const auto why = object_string_field(item, "why"); why.has_value()) {
                risk.why = *why;
            }
            if (const auto path = object_string_field(item, "path"); path.has_value()) {
                risk.path = *path;
            }
            if (!risk.title.empty()) {
                out.push_back(std::move(risk));
            }
        }
        return out;
    };

    if (auto parsed = try_parse(text); !parsed.empty()) {
        return parsed;
    }
    if (const auto start = text.find('{'); start != std::string_view::npos) {
        if (const auto end = text.rfind('}'); end != std::string_view::npos && start < end) {
            return try_parse(text.substr(start, end - start + 1));
        }
    }
    return {};
}

Report aggregate_reports(std::span<const Report> reports,
                         std::span<const std::string> skipped_paths,
                         std::span<const std::string> warnings) {
    Report out;
    out.skipped_paths.assign(skipped_paths.begin(), skipped_paths.end());
    out.warnings.assign(warnings.begin(), warnings.end());
    for (const auto& report : reports) {
        out.warnings.insert(out.warnings.end(),
                            report.warnings.begin(),
                            report.warnings.end());
    }
    out.groups_reviewed = static_cast<int>(reports.size());

    if (reports.empty()) {
        out.overall_correctness = "patch is correct";
        if (out.skipped_paths.empty()) {
            out.overall_explanation = "There are no git changes in the selected review target.";
        } else {
            out.overall_explanation = std::format(
                "No files could be reviewed; {} file(s) exceeded the per-group diff budget.",
                out.skipped_paths.size());
        }
        return out;
    }

    if (reports.size() == 1) {
        auto single = reports.front();
        single.skipped_paths = out.skipped_paths;
        single.warnings = out.warnings;
        single.groups_reviewed = 1;
        if (trim_ascii_copy(single.overall_explanation).empty()) {
            single.overall_explanation = single.findings.empty()
                ? "No actionable findings were reported across 1 reviewed file-group."
                : std::format(
                    "{} actionable finding(s) were reported across 1 reviewed file-group.",
                    single.findings.size());
        }
        return single;
    }

    std::vector<Finding> merged;
    bool any_incorrect = false;
    double confidence_sum = 0.0;
    int confidence_count = 0;
    std::vector<std::string> explanations;

    for (const auto& report : reports) {
        if (is_incorrect(report.overall_correctness)) {
            any_incorrect = true;
        }
        if (report.overall_confidence.has_value()) {
            confidence_sum += *report.overall_confidence;
            ++confidence_count;
        }
        if (const auto explanation = trim_ascii_copy(report.overall_explanation);
            !explanation.empty()) {
            explanations.push_back(explanation);
        }
        for (const auto& finding : report.findings) {
            merged.push_back(finding);
        }
    }

    // Hash the location/title key once per finding: aggregation runs over every
    // group's output, so a quadratic scan with per-comparison formatting would
    // dominate the cost of merging a large review.
    std::vector<Finding> deduped;
    deduped.reserve(merged.size());
    std::unordered_map<std::string, std::size_t> index_by_key;
    index_by_key.reserve(merged.size());
    for (auto& finding : merged) {
        auto key = finding_dedupe_key(finding);
        const auto [slot, inserted] = index_by_key.try_emplace(std::move(key), deduped.size());
        if (inserted) {
            deduped.push_back(std::move(finding));
            continue;
        }
        auto& existing = deduped[slot->second];
        if (severity_rank(effective_severity(finding))
            < severity_rank(effective_severity(existing))) {
            merge_steering_references(finding, existing);
            existing = std::move(finding);
        } else {
            merge_steering_references(existing, finding);
        }
    }

    std::ranges::sort(deduped, [](const Finding& a, const Finding& b) {
        const auto sa = severity_rank(effective_severity(a));
        const auto sb = severity_rank(effective_severity(b));
        if (sa != sb) return sa < sb;
        if (a.absolute_file_path != b.absolute_file_path) {
            return a.absolute_file_path < b.absolute_file_path;
        }
        if (a.line_start != b.line_start) return a.line_start < b.line_start;
        return a.title < b.title;
    });

    for (const auto& finding : deduped) {
        const auto severity = effective_severity(finding);
        if (severity == Severity::Critical || severity == Severity::High) {
            any_incorrect = true;
        }
    }

    out.findings = std::move(deduped);
    out.overall_correctness = any_incorrect ? "patch is incorrect" : "patch is correct";
    if (confidence_count > 0) {
        out.overall_confidence = confidence_sum / static_cast<double>(confidence_count);
    }
    if (!explanations.empty()) {
        out.overall_explanation = explanations.front();
        out.overall_explanation += std::format(
            " Combined {} file-group review(s)", out.groups_reviewed);
        if (!out.skipped_paths.empty()) {
            out.overall_explanation += std::format(
                "; skipped {} oversized file(s)", out.skipped_paths.size());
        }
        out.overall_explanation += ".";
    } else if (out.findings.empty()) {
        out.overall_explanation = std::format(
            "No actionable findings were reported across {} reviewed file-group(s).",
            out.groups_reviewed);
    } else {
        out.overall_explanation = std::format(
            "{} actionable finding(s) were reported across {} reviewed file-group(s).",
            out.findings.size(),
            out.groups_reviewed);
    }
    return out;
}

std::string render_report(const Report& report, bool include_omitted_low_count) {
    std::vector<std::string> sections;
    sections.emplace_back("# Review");

    std::string summary = "## Summary";
    const auto verdict = trim_ascii_copy(report.overall_correctness);
    if (!verdict.empty()) {
        summary += "\n\n**Overall correctness:** ";
        summary += verdict;
    }
    const std::string explanation = trim_ascii_copy(report.overall_explanation);
    if (!explanation.empty()) {
        summary += "\n\n";
        summary += explanation;
    }
    sections.push_back(std::move(summary));

    std::vector<Finding> visible;
    visible.reserve(report.findings.size());
    int hidden_low = 0;
    for (const auto& finding : report.findings) {
        if (effective_severity(finding) == Severity::Low) {
            ++hidden_low;
            continue;
        }
        visible.push_back(finding);
    }
    sections.emplace_back("## Comments");
    if (visible.empty()) {
        sections.emplace_back("No actionable comments found.");
    } else {
        sections.push_back(format_findings_block(visible));
    }
    if (hidden_low > 0 && include_omitted_low_count) {
        sections.push_back(std::format(
            "{} low-severity note(s) omitted.", hidden_low));
    }
    if (!report.warnings.empty()) {
        std::string notes = "Review notes:\n";
        for (const auto& warning : report.warnings) {
            notes += "- ";
            notes += warning;
            notes.push_back('\n');
        }
        sections.push_back(std::move(notes));
    }
    return join("\n\n", sections);
}

std::string render_low_severity_findings(const Report& report) {
    std::vector<Finding> low;
    for (const auto& finding : report.findings) {
        if (effective_severity(finding) == Severity::Low) {
            low.push_back(finding);
        }
    }
    return format_findings_block(low);
}

} // namespace core::review
