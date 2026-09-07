#include "ReadTypes.hpp"
#include "../ToolArgumentUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <format>

namespace core::tools::read {
std::string read_prefix(std::istream& stream, std::size_t bytes) {
    std::array<char, 16384> buffer;
    std::string result;
    while (result.size() < bytes && stream) {
        const auto count = std::min(buffer.size(), bytes - result.size());
        stream.read(buffer.data(), static_cast<std::streamsize>(count));
        result.append(buffer.data(), static_cast<std::size_t>(stream.gcount()));
    }
    return result;
}
std::string bounded_prefix(std::string_view text, std::size_t bytes) {
    if (text.size() <= bytes) return std::string(text);
    while (bytes && (static_cast<unsigned char>(text[bytes]) & 0xc0U) == 0x80U) --bytes;
    return std::string(text.substr(0, bytes));
}
std::string digest(std::string_view text) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char c : text) { hash ^= c; hash *= 1099511628211ULL; }
    return std::format("{:016x}", hash);
}
std::vector<std::string_view> lines(std::string_view text) {
    std::vector<std::string_view> result;
    while (!text.empty()) {
        const auto newline = text.find('\n');
        result.push_back(text.substr(0, newline));
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    return result;
}
std::size_t line_count(std::string_view text) {
    if (text.empty()) return 0;
    const auto newlines = static_cast<std::size_t>(std::ranges::count(text, '\n'));
    return text.ends_with('\n') ? newlines : newlines + 1;
}
std::string slice(const Resource& source, int first, int count) {
    // Find just the requested span; avoid allocating a vector for every line
    // in a large file when recovering a short citation.
    std::size_t start = 0;
    for (int line = 1; line < first; ++line) {
        const auto end = source.text.find('\n', start);
        if (end == std::string::npos) return {};
        start = end + 1;
    }
    auto end = start;
    if (!count) end = source.text.size();
    else for (int line = 0; line < count; ++line) {
        const auto newline = source.text.find('\n', end);
        if (newline == std::string::npos) { end = source.text.size(); break; }
        end = newline + 1;
    }
    return source.text.substr(start, end - start);
}
bool is_instruction_resource(std::string_view uri) {
    const auto name = std::filesystem::path(uri).filename().string();
    return name == "AGENTS.md" || name == "CLAUDE.md" || name == "SKILL.md";
}
std::expected<Options, std::string> parse_options(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded{json};
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc)) return std::unexpected("Invalid JSON arguments for read.");
    if (auto error = detail::validate_object_arguments(doc, "read",
            {"path", "view", "question", "offset_line", "limit_lines", "select", "expected_digest"}))
        return std::unexpected(*error);
    Options options;
    simdjson::dom::element path;
    if (doc["path"].get(path)) return std::unexpected("read requires path.");
    std::string_view text;
    if (!path.get(text)) options.paths.emplace_back(text);
    else {
        simdjson::dom::array paths;
        if (path.get(paths)) return std::unexpected("path must be a string or an array of strings.");
        for (auto item : paths) {
            if (item.get(text)) return std::unexpected("Every path must be a string.");
            options.paths.emplace_back(text);
        }
    }
    if (options.paths.empty() || options.paths.size() > kMaxResources)
        return std::unexpected("read accepts 1 to 8 resources.");
    for (const auto& value : options.paths)
        if (value.empty() || value.size() > 1024) return std::unexpected("Paths must contain 1 to 1024 bytes.");
    auto string_field = [&](const char* key, std::string& target) {
        simdjson::dom::element value;
        if (doc[key].get(value)) return true;
        if (value.get(text)) return false;
        target = text;
        return true;
    };
    if (!string_field("view", options.view) || !string_field("question", options.question)
        || !string_field("expected_digest", options.expected_digest))
        return std::unexpected("view, question, and expected_digest must be strings.");
    const bool explicit_view = !doc["view"].error();
    if (!explicit_view && !options.question.empty()) options.view = "auto";
    if (options.view != "auto" && options.view != "exact" && options.view != "outline")
        return std::unexpected("view must be auto, exact, or outline.");
    if (options.question.size() > 8192) return std::unexpected("Question exceeds 8192 bytes.");
    if (options.expected_digest.size() > 64) return std::unexpected("Invalid expected_digest.");
    if (!options.question.empty() && options.view != "auto")
        return std::unexpected("A question requires view auto; use exact separately for source recovery.");
    auto integer_field = [&](const char* key, int& target, int maximum) {
        simdjson::dom::element value;
        if (doc[key].get(value)) return true;
        int64_t n = 0;
        if (value.get(n) || n < 1 || n > maximum) return false;
        target = static_cast<int>(n); options.sliced = true; return true;
    };
    if (!integer_field("offset_line", options.offset_line, 10000000)
        || !integer_field("limit_lines", options.limit_lines, 10000))
        return std::unexpected("offset_line must be positive; limit_lines must be between 1 and 10000.");
    if (options.sliced && !options.question.empty())
        return std::unexpected("Use question and exact line ranges in separate reads.");
    simdjson::dom::element selection;
    if (!doc["select"].get(selection)) {
        if (auto error = detail::validate_object_arguments(selection, "read.select", {"cell"}))
            return std::unexpected(*error);
        simdjson::dom::object object;
        if (selection.get(object) || object.size() != 1)
            return std::unexpected("select requires a notebook cell id or 1-based index.");
        for (auto field : object) {
            if (field.value.get(text) || text.empty() || text.size() > 1024)
                return std::unexpected("Cell selectors must be nonempty strings of at most 1024 bytes.");
            options.cell = text;
        }
    }
    if (options.paths.size() > 1 && (options.sliced || !options.cell.empty() || !options.expected_digest.empty()))
        return std::unexpected("Selections and expected_digest require one resource.");
    return options;
}

std::string outline(const Resource& source, std::size_t budget) {
    const auto entries = lines(source.text);
    std::string out = "[Structural preview; omitted bodies are not analyzed]\n";
    // Deterministic line selection is deliberately conservative and language
    // agnostic. It does not claim AST parsing or semantic completeness.
    for (std::size_t i = 0; i < entries.size(); ++i) {
        auto line = core::utils::str::trim_ascii_view(entries[i]);
        const bool declaration = line.starts_with("#") || line.starts_with("import ")
            || line.starts_with("from ") || line.starts_with("class ") || line.starts_with("struct ")
            || line.starts_with("enum ") || line.starts_with("def ") || line.starts_with("async def ")
            || line.starts_with("fn ") || line.starts_with("pub ") || line.starts_with("export ")
            || line.starts_with("function ") || line.starts_with("interface ")
            || (line.find('(') != std::string_view::npos && (line.ends_with('{') || line.ends_with(';')));
        if (i >= 8 && !declaration) continue;
        const auto row = std::format("{}: {}\n", i + 1, bounded_prefix(entries[i], 240));
        if (out.size() + row.size() + 160 > budget) break;
        out += row;
    }
    out += "Use view=exact with offset_line/limit_lines to recover source.\n";
    return bounded_prefix(out, budget);
}
} // namespace core::tools::read
