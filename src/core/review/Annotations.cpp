#include "Annotations.hpp"

#include "../utils/JsonWriter.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <unordered_set>

namespace core::review {
namespace {

constexpr std::size_t kMaxAnnotations = 10'000;
constexpr std::size_t kMaxTextBytes = 1024 * 1024;
constexpr std::size_t kMaxPathBytes = 4096;

using core::utils::JsonWriter;

[[nodiscard]] bool unique_keys(simdjson::dom::object object) {
    std::unordered_set<std::string_view> keys;
    for (const auto field : object) {
        if (!keys.insert(field.key).second) return false;
    }
    return true;
}

[[nodiscard]] bool string_field(simdjson::dom::object object,
                                std::string_view key, std::string& target,
                                std::size_t max_bytes) {
    std::string_view value;
    if (object[key].get(value) != simdjson::SUCCESS || value.size() > max_bytes) {
        return false;
    }
    target = value;
    return true;
}

[[nodiscard]] bool line_field(simdjson::dom::object object,
                              std::string_view key, int& target) {
    std::int64_t value;
    if (object[key].get(value) != simdjson::SUCCESS
        || value <= 0 || value > std::numeric_limits<int>::max()) {
        return false;
    }
    target = static_cast<int>(value);
    return true;
}

[[nodiscard]] bool version_one(simdjson::dom::object object) {
    std::int64_t version;
    return object["version"].get(version) == simdjson::SUCCESS && version == 1;
}

[[nodiscard]] bool context_field(simdjson::dom::object object,
                                 std::string_view key,
                                 std::vector<std::string>& target,
                                 std::size_t& context_bytes) {
    simdjson::dom::array lines;
    if (object[key].get(lines) != simdjson::SUCCESS || lines.size() > 3) return false;
    for (const auto line : lines) {
        std::string_view text;
        if (line.get(text) != simdjson::SUCCESS || text.size() > kMaxTextBytes - context_bytes
            || text.find_first_of("\r\n") != std::string_view::npos) {
            return false;
        }
        context_bytes += text.size();
        target.emplace_back(text);
    }
    return true;
}

[[nodiscard]] bool normalize_uuid(std::string& id) {
    if (id.size() != 36) return false;
    for (std::size_t i = 0; i < id.size(); ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (id[i] != '-') return false;
        } else if (id[i] >= 'A' && id[i] <= 'F') {
            id[i] = static_cast<char>(id[i] - 'A' + 'a');
        } else if (!((id[i] >= '0' && id[i] <= '9')
                     || (id[i] >= 'a' && id[i] <= 'f'))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool valid_path(std::string_view path) {
    return !path.empty() && std::ranges::none_of(path, [](unsigned char c) {
        return c < 0x20 || c == 0x7f;
    });
}

[[nodiscard]] bool valid_tone(std::string_view tone) {
    return tone == "neutral" || tone == "success" || tone == "warning"
        || tone == "critical" || tone == "pending";
}

[[nodiscard]] std::expected<std::vector<Annotation>, std::string>
parse_annotations(simdjson::dom::object object) {
    simdjson::dom::array entries;
    if (object["annotations"].get(entries) != simdjson::SUCCESS) {
        return std::unexpected("Review response must contain an annotations array.");
    }
    if (entries.size() > kMaxAnnotations) {
        return std::unexpected("Review response has too many annotations.");
    }

    std::vector<Annotation> annotations;
    std::unordered_set<std::string> ids;
    annotations.reserve(entries.size());
    for (const auto entry : entries) {
        simdjson::dom::object annotation;
        Annotation value;
        std::size_t context_bytes = 0;
        if (entry.get(annotation) != simdjson::SUCCESS || !unique_keys(annotation)
            || !string_field(annotation, "id", value.id, 36)
            || !normalize_uuid(value.id)
            || !string_field(annotation, "path", value.path, kMaxPathBytes)
            || !valid_path(value.path)
            || !string_field(annotation, "side", value.side, 3)
            || (value.side != "old" && value.side != "new")
            || !line_field(annotation, "start_line", value.start_line)
            || !line_field(annotation, "end_line", value.end_line)
            || value.end_line < value.start_line
            || !string_field(annotation, "selected_text", value.selected_text, kMaxTextBytes)
            || !string_field(annotation, "comment", value.comment, kMaxTextBytes)
            || !string_field(annotation, "tone", value.tone, 8)
            || !valid_tone(value.tone)
            || !string_field(annotation, "revision", value.revision, 256)
            || value.revision.empty()
            || !string_field(annotation, "anchor_state", value.anchor_state, 8)
            || (value.anchor_state != "attached" && value.anchor_state != "outdated")
            || !context_field(annotation, "context_before", value.context_before, context_bytes)
            || !context_field(annotation, "context_after", value.context_after, context_bytes)) {
            return std::unexpected("Review response contains an invalid annotation.");
        }
        if (!ids.insert(value.id).second) {
            return std::unexpected("Review response contains duplicate annotation IDs.");
        }
        annotations.push_back(std::move(value));
    }
    return annotations;
}

void write_strings(JsonWriter& writer, std::span<const std::string> strings) {
    auto array = writer.array();
    bool first = true;
    for (const auto& string : strings) {
        if (!first) writer.comma();
        first = false;
        writer.str(string);
    }
}

void write_annotations(JsonWriter& writer, std::span<const Annotation> annotations) {
    auto array = writer.array();
    bool first = true;
    for (const auto& annotation : annotations) {
        if (!first) writer.comma();
        first = false;
        auto object = writer.object();
        writer.kv_str("id", annotation.id).comma()
            .kv_str("path", annotation.path).comma()
            .kv_str("side", annotation.side).comma()
            .kv_num("start_line", annotation.start_line).comma()
            .kv_num("end_line", annotation.end_line).comma()
            .kv_str("selected_text", annotation.selected_text).comma()
            .kv_str("comment", annotation.comment).comma()
            .kv_str("tone", annotation.tone).comma()
            .kv_str("revision", annotation.revision).comma()
            .kv_str("anchor_state", annotation.anchor_state).comma()
            .key("context_before");
        write_strings(writer, annotation.context_before);
        writer.comma().key("context_after");
        write_strings(writer, annotation.context_after);
    }
}

[[nodiscard]] std::string code_fence(std::string_view text) {
    std::size_t longest = 0;
    std::size_t run = 0;
    for (const auto c : text) {
        run = c == '`' ? run + 1 : 0;
        longest = std::max(longest, run);
    }
    return std::string(std::max<std::size_t>(3, longest + 1), '`');
}

} // namespace

std::string encode_review_request(std::string_view patch,
                                  std::span<const Annotation> annotations) {
    JsonWriter writer;
    {
        auto object = writer.object();
        writer.kv_num("version", 1).comma().kv_str("patch", patch).comma()
            .key("annotations");
        write_annotations(writer, annotations);
    }
    return std::move(writer).take();
}

std::expected<std::vector<Annotation>, std::string>
parse_review_response(std::string_view json) {
    if (json.size() > kMaxReviewDocumentBytes) {
        return std::unexpected("Review response exceeds 16 MiB.");
    }
    simdjson::dom::parser parser;
    simdjson::dom::object object;
    if (parser.parse(json).get(object) != simdjson::SUCCESS || !unique_keys(object)) {
        return std::unexpected("Invalid review response JSON.");
    }
    if (!version_one(object)) {
        return std::unexpected("Unsupported review response version.");
    }
    return parse_annotations(object);
}

std::string render_annotation_prompt(std::span<const Annotation> annotations) {
    std::string prompt;
    for (const auto& annotation : annotations) {
        if (!prompt.empty()) prompt += "\n\n";
        prompt += "### Review comment\nFile: ";
        JsonWriter path;
        path.str(annotation.path);
        prompt += path.view();
        prompt += " (" + annotation.side + " side, lines "
            + std::to_string(annotation.start_line) + "-"
            + std::to_string(annotation.end_line) + ")\n";
        prompt += "Tone: " + annotation.tone + "\n\nSelected code:\n";
        const auto fence = code_fence(annotation.selected_text);
        prompt += fence + "\n" + annotation.selected_text;
        if (!annotation.selected_text.ends_with('\n')) prompt += '\n';
        prompt += fence + "\n\nComment:\n"
            + (annotation.comment.empty() ? "(no comment)" : annotation.comment);
    }
    return prompt;
}

ReviewMerge merge_review(
    std::span<const Annotation> held,
    std::span<const Annotation> shown,
    std::vector<Annotation> returned) {
    const auto holds = [](std::span<const Annotation> annotations, const Annotation& annotation) {
        return std::ranges::any_of(annotations, [&](const auto& other) { return other.id == annotation.id; });
    };
    ReviewMerge merge;
    std::ranges::copy_if(held, std::back_inserter(merge.kept),
        [&](const auto& annotation) { return !holds(shown, annotation); });
    std::ranges::copy_if(returned, std::back_inserter(merge.added),
        [&](const auto& annotation) { return !holds(held, annotation); });
    std::ranges::move(returned, std::back_inserter(merge.kept));
    return merge;
}

void append_annotations_json(std::string& out, std::span<const Annotation> annotations) {
    JsonWriter writer;
    write_annotations(writer, annotations);
    out += std::move(writer).take();
}

std::vector<Annotation> parse_saved_annotations(simdjson::dom::object object) {
    if (simdjson::dom::array entries; object["annotations"].get(entries) != simdjson::SUCCESS) {
        return {};
    }
    auto annotations = parse_annotations(object);
    return annotations ? std::move(*annotations) : std::vector<Annotation>{};
}

} // namespace core::review
