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
            if (const auto value = object_string_field(finding_object, "rationale"); value.has_value()) {
                finding.rationale = trim_ascii_copy(*value);
            }
            if (const auto value = object_string_field(finding_object, "evidence"); value.has_value()) {
                finding.evidence = trim_ascii_copy(*value);
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

void merge_supporting_details(Finding& target, const Finding& source) {
    // A duplicate that loses on severity may still be the one that explained
    // itself; keep its reasons when the survivor has none.
    if (target.rationale.empty()) target.rationale = source.rationale;
    if (target.evidence.empty()) target.evidence = source.evidence;
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

// One finding, split the way every surface needs it: the comment a reader
// meets first, and the reasons behind it.
struct FindingText {
    std::string comment;
    std::string rationale;
    std::string evidence;
    std::vector<std::string> guidance;
    std::string id;

    [[nodiscard]] bool has_reasons() const noexcept {
        return !rationale.empty() || !evidence.empty() || !guidance.empty();
    }
};

[[nodiscard]] FindingText describe_finding(const Finding& finding) {
    std::vector<std::string> tags;
    tags.emplace_back(to_string(effective_severity(finding)));
    if (finding.category.has_value() && *finding.category != Category::Other) {
        tags.emplace_back(to_string(*finding.category));
    }
    const std::string_view title =
        finding.title.empty() ? std::string_view("Untitled finding") : finding.title;

    // Location first, on its own rule, so the reader anchors the comment to a
    // file and line range before reading the prose.
    FindingText text{
        .comment = std::format("── {} ──\n[{}] {}", format_location(finding), join(" · ", tags), title),
        .rationale = finding.rationale,
        .evidence = finding.evidence,
        .guidance = {},
        .id = finding_id(finding),
    };
    if (!finding.body.empty()) {
        text.comment += '\n';
        text.comment += finding.body;
    }
    for (const auto& reference : finding.steering_references) {
        text.guidance.push_back(std::format(
            "[{}] ({}): “{}”",
            reference.source_id,
            reference.source_label,
            reference.rule_excerpt));
    }
    return text;
}

// The plain form: what the model reads back from history, and what a host
// without display sections shows. Every reason stays, on labelled lines.
[[nodiscard]] std::string plain_finding(const FindingText& text) {
    std::string out = text.comment;
    const auto append_line = [&out](std::string_view label, std::string_view value) {
        out += '\n';
        out += label;
        out += value;
    };
    if (!text.rationale.empty()) append_line("Why flagged: ", text.rationale);
    if (!text.evidence.empty()) append_line("Checked: ", text.evidence);
    for (const auto& guidance : text.guidance) append_line("Project guidance ", guidance);
    append_line("Finding ID: ", text.id);
    return out;
}

// The body of the "Why this was flagged" disclosure.
[[nodiscard]] std::string reasons_markdown(const FindingText& text) {
    std::vector<std::string> paragraphs;
    if (!text.rationale.empty()) paragraphs.push_back(text.rationale);
    if (!text.evidence.empty()) paragraphs.push_back("**Checked:** " + text.evidence);
    for (const auto& guidance : text.guidance) {
        paragraphs.push_back("**Project guidance** " + guidance);
    }
    paragraphs.push_back(std::format("Finding ID: `{}`", text.id));
    return join("\n\n", paragraphs);
}

using FindingRefs = std::vector<const Finding*>;

[[nodiscard]] std::string format_findings_block(const FindingRefs& findings) {
    std::vector<std::string> blocks;
    blocks.reserve(findings.size());
    for (const Finding* finding : findings) {
        blocks.push_back(plain_finding(describe_finding(*finding)));
    }
    return join("\n\n", blocks);
}

// Low-severity notes are kept out of the main comment list on every surface.
struct FindingsByVisibility {
    FindingRefs visible;
    FindingRefs low;
};

[[nodiscard]] FindingsByVisibility split_by_visibility(const Report& report) {
    FindingsByVisibility out;
    for (const auto& finding : report.findings) {
        (effective_severity(finding) == Severity::Low ? out.low : out.visible)
            .push_back(&finding);
    }
    return out;
}

[[nodiscard]] std::string report_summary(const Report& report) {
    std::string summary = "# Review\n\n## Summary";
    if (const auto verdict = trim_ascii_copy(report.overall_correctness); !verdict.empty()) {
        summary += "\n\n**Overall correctness:** ";
        summary += verdict;
    }
    if (!report.reviewer.empty()) {
        summary += "\n\n**Reviewed by:** ";
        summary += report.reviewer;
    }
    if (const auto explanation = trim_ascii_copy(report.overall_explanation);
        !explanation.empty()) {
        summary += "\n\n";
        summary += explanation;
    }
    return summary;
}

[[nodiscard]] std::string review_notes(const Report& report) {
    if (report.warnings.empty()) return {};
    std::string notes = "Review notes:";
    for (const auto& warning : report.warnings) {
        notes += "\n- ";
        notes += warning;
    }
    return notes;
}

[[nodiscard]] std::string low_severity_label(std::size_t count) {
    return count == 1 ? std::string("1 low-severity note")
                      : std::format("{} low-severity notes", count);
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
            merge_supporting_details(finding, existing);
            existing = std::move(finding);
        } else {
            merge_supporting_details(existing, finding);
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

void drop_unchecked_evidence(Report& report, int tools_run) {
    if (tools_run > 0) return;
    for (auto& finding : report.findings) finding.evidence.clear();
}

std::string finding_id(const Finding& finding) {
    // FNV-1a over the same key that deduplicates findings: two findings the
    // review would merge share an id, and nothing else does in practice.
    std::uint32_t hash = 2166136261U;
    for (const unsigned char byte : finding_dedupe_key(finding)) {
        hash ^= byte;
        hash *= 16777619U;
    }
    return std::format("f-{:08x}", hash);
}

std::string render_report(const Report& report) {
    const auto [visible, low] = split_by_visibility(report);
    std::vector<std::string> sections{report_summary(report), "## Comments"};
    sections.push_back(visible.empty() ? std::string("No actionable comments found.")
                                       : format_findings_block(visible));
    if (!low.empty()) {
        sections.emplace_back("### Low-severity notes");
        sections.push_back(format_findings_block(low));
    }
    if (auto notes = review_notes(report); !notes.empty()) {
        sections.push_back(std::move(notes));
    }
    return join("\n\n", sections);
}

std::vector<core::llm::DisplaySection> render_report_sections(const Report& report,
                                                              std::string_view trailer) {
    const auto [visible, low] = split_by_visibility(report);
    std::vector<core::llm::DisplaySection> sections;
    sections.reserve(visible.size() + 2);

    std::string head = report_summary(report) + "\n\n## Comments";
    if (visible.empty()) head += "\n\nNo actionable comments found.";
    sections.push_back({.markdown = std::move(head)});

    for (const Finding* finding : visible) {
        auto text = describe_finding(*finding);
        if (text.has_reasons()) {
            sections.push_back({
                .markdown = text.comment,
                .details_summary = "Why this was flagged",
                .details_markdown = reasons_markdown(text),
            });
        } else {
            // Nothing to disclose: a box holding only the id would be noise.
            sections.push_back({
                .markdown = std::format("{}\n\nFinding ID: `{}`", text.comment, text.id),
            });
        }
    }

    std::vector<std::string> tail;
    if (auto notes = review_notes(report); !notes.empty()) tail.push_back(std::move(notes));
    if (auto extra = trim_ascii_copy(trailer); !extra.empty()) tail.push_back(std::move(extra));
    if (!tail.empty() || !low.empty()) {
        core::llm::DisplaySection last{.markdown = join("\n\n", tail)};
        if (!low.empty()) {
            last.details_summary = low_severity_label(low.size());
            last.details_markdown = format_findings_block(low);
        }
        sections.push_back(std::move(last));
    }
    return sections;
}

} // namespace core::review
