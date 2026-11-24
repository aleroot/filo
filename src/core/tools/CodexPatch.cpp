#include "CodexPatch.hpp"

#include <algorithm>
#include <format>
#include <map>
#include <utility>

namespace core::tools {
namespace {

constexpr std::string_view kBegin = "*** Begin Patch";
constexpr std::string_view kEnd = "*** End Patch";
constexpr std::string_view kEndOfFile = "*** End of File";
constexpr std::string_view kAdd = "*** Add File: ";
constexpr std::string_view kDelete = "*** Delete File: ";
constexpr std::string_view kUpdate = "*** Update File: ";
constexpr std::string_view kMove = "*** Move to: ";

enum class LineKind { Context, Delete, Insert };

struct DiffLine {
    LineKind kind = LineKind::Context;
    std::string text;
};

/// One `@@` section. `at_eof` requires the context to match through the last line.
struct Segment {
    std::vector<DiffLine> lines;
    bool at_eof = false;
};

struct UpdateHunk {
    std::string path;
    std::optional<std::string> move_to;
    std::vector<Segment> segments;
};

struct AddHunk {
    std::string path;
    std::string content;
};

struct DeleteHunk {
    std::string path;
};

struct Envelope {
    struct Item {
        enum class Kind { Add, Delete, Update };
        Kind kind = Kind::Update;
        AddHunk add;
        DeleteHunk deletion;
        UpdateHunk update;
    };
    std::vector<Item> items;
};

struct Document {
    std::vector<std::string> lines;
    bool trailing_newline = false;
    bool crlf = false;
};

[[nodiscard]] std::string_view trim_trailing(std::string_view line) noexcept {
    while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) {
        line.remove_suffix(1);
    }
    return line;
}

[[nodiscard]] bool marker_is(std::string_view line, std::string_view marker) noexcept {
    return trim_trailing(line) == marker;
}

[[nodiscard]] std::optional<std::string> marker_argument(std::string_view line, std::string_view prefix) {
    const auto trimmed = trim_trailing(line);
    if (!trimmed.starts_with(prefix)) return std::nullopt;
    auto path = trimmed.substr(prefix.size());
    while (!path.empty() && (path.front() == ' ' || path.front() == '\t')) path.remove_prefix(1);
    if (path.empty()) return std::nullopt;
    return std::string(path);
}

[[nodiscard]] bool is_hunk_boundary(std::string_view line) noexcept {
    return marker_is(line, kEnd)
        || marker_argument(line, kAdd)
        || marker_argument(line, kDelete)
        || marker_argument(line, kUpdate);
}

class Lines {
public:
    explicit Lines(std::string_view text) {
        while (!text.empty()) {
            const auto newline = text.find('\n');
            auto line = text.substr(0, newline == std::string_view::npos ? text.size() : newline);
            if (line.ends_with('\r')) line.remove_suffix(1);
            lines_.emplace_back(line);
            if (newline == std::string_view::npos) break;
            text.remove_prefix(newline + 1);
        }
    }

    [[nodiscard]] bool done() const noexcept { return index_ >= lines_.size(); }

    [[nodiscard]] std::string_view peek() const noexcept {
        return done() ? std::string_view{} : lines_[index_];
    }

    std::string_view take() {
        return done() ? std::string_view{} : lines_[index_++];
    }

private:
    std::vector<std::string_view> lines_;
    std::size_t index_ = 0;
};

[[nodiscard]] Document split_document(std::string_view text) {
    Document document;
    if (text.empty()) return document;
    document.crlf = text.find("\r\n") != std::string_view::npos;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto newline = text.find('\n', start);
        if (newline == std::string_view::npos) {
            auto tail = text.substr(start);
            if (!tail.empty() && tail.back() == '\r') tail.remove_suffix(1);
            if (!tail.empty()) document.lines.emplace_back(tail);
            document.trailing_newline = false;
            break;
        }
        auto line = text.substr(start, newline - start);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        document.lines.emplace_back(line);
        document.trailing_newline = true;
        start = newline + 1;
        if (start == text.size()) break;
    }
    return document;
}

[[nodiscard]] std::string join_document(const Document& document) {
    std::string out;
    const auto newline = document.crlf ? std::string_view{"\r\n"} : std::string_view{"\n"};
    for (std::size_t i = 0; i < document.lines.size(); ++i) {
        if (i != 0) out.append(newline);
        out += document.lines[i];
    }
    if (document.trailing_newline && !document.lines.empty()) out.append(newline);
    if (document.lines.empty() && document.trailing_newline) out.append(newline);
    return out;
}

enum class Fuzz { Exact, Rstrip, Trim };

[[nodiscard]] std::string_view strip_right(std::string_view text) noexcept {
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) text.remove_suffix(1);
    return text;
}

[[nodiscard]] std::string_view strip_left(std::string_view text) noexcept {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    return text;
}

[[nodiscard]] bool lines_equal(std::string_view file_line, std::string_view patch_line, Fuzz fuzz) noexcept {
    switch (fuzz) {
        case Fuzz::Exact: return file_line == patch_line;
        case Fuzz::Rstrip: return strip_right(file_line) == strip_right(patch_line);
        case Fuzz::Trim: return strip_left(strip_right(file_line)) == strip_left(strip_right(patch_line));
    }
    return false;
}

struct Seek {
    enum class Status { Found, Missing, Ambiguous };
    Status status = Status::Missing;
    std::size_t index = 0;
};

[[nodiscard]] Seek seek_pattern(const std::vector<std::string>& lines,
                                const std::vector<std::string>& pattern,
                                Fuzz fuzz,
                                std::size_t begin,
                                std::size_t end,
                                bool at_eof) {
    Seek seek;
    if (pattern.empty() || begin > end || end > lines.size()) return seek;
    const auto last = lines.size() - pattern.size();
    if (pattern.size() > lines.size() || begin > last) return seek;
    const auto stop = std::min(end, last + 1);
    if (at_eof) {
        if (last < begin || last >= stop) return seek;
        bool match = true;
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            if (!lines_equal(lines[last + i], pattern[i], fuzz)) {
                match = false;
                break;
            }
        }
        if (!match) return seek;
        seek.status = Seek::Status::Found;
        seek.index = last;
        return seek;
    }
    for (std::size_t start = begin; start < stop; ++start) {
        bool match = true;
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            if (!lines_equal(lines[start + i], pattern[i], fuzz)) {
                match = false;
                break;
            }
        }
        if (!match) continue;
        if (seek.status == Seek::Status::Found) {
            seek.status = Seek::Status::Ambiguous;
            return seek;
        }
        seek.status = Seek::Status::Found;
        seek.index = start;
    }
    return seek;
}

[[nodiscard]] Seek locate(const std::vector<std::string>& lines,
                          const std::vector<std::string>& pattern,
                          bool at_eof,
                          std::size_t cursor) {
    // Exact text, then trailing whitespace, then surrounding whitespace.
    // A looser match is used only when every tighter one missed, and a pattern
    // that matches twice at that looseness is rejected rather than guessed.
    for (const auto fuzz : {Fuzz::Exact, Fuzz::Rstrip, Fuzz::Trim}) {
        auto forward = seek_pattern(lines, pattern, fuzz, cursor, lines.size(), at_eof);
        if (forward.status == Seek::Status::Ambiguous) return forward;
        if (forward.status == Seek::Status::Found) return forward;
        if (cursor == 0) continue;
        auto earlier = seek_pattern(lines, pattern, fuzz, 0, cursor, at_eof);
        if (earlier.status != Seek::Status::Missing) return earlier;
    }
    return {};
}

[[nodiscard]] std::string show_segment(const Segment& segment) {
    std::string out;
    std::size_t shown = 0;
    for (const auto& line : segment.lines) {
        if (shown == 12) {
            out += "...\n";
            break;
        }
        const char prefix = line.kind == LineKind::Insert ? '+'
            : line.kind == LineKind::Delete ? '-' : ' ';
        out += prefix;
        out += line.text;
        out += '\n';
        ++shown;
    }
    return out;
}

[[nodiscard]] std::optional<std::string> apply_segments(Document& document,
                                                        const std::vector<Segment>& segments,
                                                        std::string_view path) {
    std::size_t cursor = 0;
    for (const auto& segment : segments) {
        if (segment.lines.empty()) continue;
        std::vector<std::string> pattern;
        std::vector<std::string> replacement;
        pattern.reserve(segment.lines.size());
        replacement.reserve(segment.lines.size());
        for (const auto& line : segment.lines) {
            if (line.kind != LineKind::Insert) pattern.push_back(line.text);
            if (line.kind != LineKind::Delete) replacement.push_back(line.text);
        }
        if (pattern.empty()) {
            if (!segment.at_eof) {
                return std::format(
                    "Update hunk for {} inserts lines without context to locate them.", path);
            }
            document.lines.insert(document.lines.end(), replacement.begin(), replacement.end());
            document.trailing_newline = true;
            cursor = document.lines.size();
            continue;
        }
        const auto found = locate(document.lines, pattern, segment.at_eof, cursor);
        if (found.status == Seek::Status::Ambiguous) {
            return std::format(
                "The context for {} matches more than once. Add surrounding lines so it is unique.\n{}",
                path, show_segment(segment));
        }
        if (found.status == Seek::Status::Missing) {
            return std::format(
                "Could not find this context in {}:\n{}", path, show_segment(segment));
        }
        document.lines.erase(
            document.lines.begin() + static_cast<std::ptrdiff_t>(found.index),
            document.lines.begin() + static_cast<std::ptrdiff_t>(found.index + pattern.size()));
        document.lines.insert(
            document.lines.begin() + static_cast<std::ptrdiff_t>(found.index),
            replacement.begin(), replacement.end());
        cursor = found.index + replacement.size();
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> parse_update_body(Lines& lines, UpdateHunk& hunk) {
    Segment segment;
    bool any_change = false;
    auto push_segment = [&] {
        if (segment.lines.empty() && !segment.at_eof) return;
        hunk.segments.push_back(std::move(segment));
        segment = {};
    };
    while (!lines.done()) {
        const auto line = lines.peek();
        if (is_hunk_boundary(line)) break;
        lines.take();
        if (marker_is(line, kEndOfFile)) {
            segment.at_eof = true;
            push_segment();
            continue;
        }
        if (marker_argument(line, kMove)) {
            return std::format("*** Move to must follow *** Update File, before the hunk for {}.", hunk.path);
        }
        const auto trimmed = trim_trailing(line);
        if (trimmed == "@@" || trimmed.starts_with("@@ ")) {
            push_segment();
            continue;
        }
        if (line.empty()) {
            // A blank line between file operations is padding. One inside a
            // hunk is an empty context line the file has to contain.
            if (!segment.lines.empty()) segment.lines.push_back({LineKind::Context, {}});
            continue;
        }
        const char prefix = line.front();
        if (prefix != ' ' && prefix != '+' && prefix != '-') {
            return std::format(
                "Unexpected line in the update of {}: {}", hunk.path, trim_trailing(line));
        }
        const auto kind = prefix == '+' ? LineKind::Insert
            : prefix == '-' ? LineKind::Delete : LineKind::Context;
        if (kind != LineKind::Context) any_change = true;
        segment.lines.push_back({kind, std::string(line.substr(1))});
    }
    push_segment();
    if (!any_change && !hunk.move_to) {
        return std::format("Update file hunk for {} is empty.", hunk.path);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<Envelope> parse_envelope(std::string_view text, std::string& error) {
    Lines lines(text);
    while (!lines.done() && trim_trailing(lines.peek()).empty()) lines.take();
    if (lines.done() || !marker_is(lines.peek(), kBegin)) {
        error = "A Codex patch must start with *** Begin Patch.";
        return std::nullopt;
    }
    lines.take();

    Envelope envelope;
    while (!lines.done()) {
        while (!lines.done() && trim_trailing(lines.peek()).empty()) lines.take();
        if (lines.done()) break;
        const auto line = lines.peek();
        if (marker_is(line, kEnd)) {
            lines.take();
            while (!lines.done()) {
                if (!trim_trailing(lines.take()).empty()) {
                    error = "Unexpected text after *** End Patch.";
                    return std::nullopt;
                }
            }
            if (envelope.items.empty()) {
                error = "The Codex patch contains no file operations.";
                return std::nullopt;
            }
            return envelope;
        }
        if (auto path = marker_argument(line, kAdd)) {
            lines.take();
            std::string content;
            bool any = false;
            while (!lines.done() && !is_hunk_boundary(lines.peek())) {
                const auto body = lines.take();
                if (body.empty() || body.front() != '+') {
                    error = std::format("Add file {} lines must start with '+'.", *path);
                    return std::nullopt;
                }
                if (any) content += '\n';
                content.append(body.substr(1));
                any = true;
            }
            if (!any) {
                error = std::format("Add file {} has no content lines.", *path);
                return std::nullopt;
            }
            content += '\n';
            Envelope::Item item;
            item.kind = Envelope::Item::Kind::Add;
            item.add = AddHunk{std::move(*path), std::move(content)};
            envelope.items.push_back(std::move(item));
            continue;
        }
        if (auto path = marker_argument(line, kDelete)) {
            lines.take();
            Envelope::Item item;
            item.kind = Envelope::Item::Kind::Delete;
            item.deletion = DeleteHunk{std::move(*path)};
            envelope.items.push_back(std::move(item));
            continue;
        }
        if (auto path = marker_argument(line, kUpdate)) {
            lines.take();
            UpdateHunk hunk;
            hunk.path = std::move(*path);
            if (!lines.done()) {
                if (auto destination = marker_argument(lines.peek(), kMove)) {
                    lines.take();
                    hunk.move_to = std::move(*destination);
                }
            }
            if (auto body_error = parse_update_body(lines, hunk)) {
                error = std::move(*body_error);
                return std::nullopt;
            }
            Envelope::Item item;
            item.kind = Envelope::Item::Kind::Update;
            item.update = std::move(hunk);
            envelope.items.push_back(std::move(item));
            continue;
        }
        error = std::format("Unexpected line in Codex patch: {}", trim_trailing(line));
        return std::nullopt;
    }
    error = "The Codex patch is missing *** End Patch.";
    return std::nullopt;
}

struct Node {
    bool existed = false;
    std::string original;
    bool exists = false;
    Document document;
};

} // namespace

std::optional<CodexPlan> plan_codex_patch(
    std::string_view text,
    const std::function<std::optional<std::string>(std::string_view)>& read_file,
    std::string& error)
{
    auto envelope = parse_envelope(text, error);
    if (!envelope) return std::nullopt;

    std::map<std::string, Node> files;
    auto load = [&](const std::string& path) -> Node& {
        if (const auto it = files.find(path); it != files.end()) return it->second;
        Node node;
        if (auto content = read_file(path)) {
            node.existed = true;
            node.exists = true;
            node.original = std::move(*content);
            node.document = split_document(node.original);
        }
        return files.emplace(path, std::move(node)).first->second;
    };

    CodexPlan plan;
    for (const auto& item : envelope->items) {
        if (item.kind == Envelope::Item::Kind::Add) {
            auto& node = load(item.add.path);
            if (node.exists) {
                error = std::format("Cannot add {}: the file already exists.", item.add.path);
                return std::nullopt;
            }
            node.exists = true;
            node.document = split_document(item.add.content);
            node.document.trailing_newline = true;
            plan.output += std::format("Added {}\n", item.add.path);
            continue;
        }
        if (item.kind == Envelope::Item::Kind::Delete) {
            auto& node = load(item.deletion.path);
            if (!node.exists) {
                error = std::format("Cannot delete {}: the file does not exist.", item.deletion.path);
                return std::nullopt;
            }
            node.exists = false;
            node.document = {};
            plan.output += std::format("Deleted {}\n", item.deletion.path);
            continue;
        }

        const auto& hunk = item.update;
        auto& source = load(hunk.path);
        if (!source.exists) {
            error = std::format("Cannot update {}: the file does not exist.", hunk.path);
            return std::nullopt;
        }
        if (auto apply_error = apply_segments(source.document, hunk.segments, hunk.path)) {
            error = std::move(*apply_error);
            return std::nullopt;
        }
        if (!hunk.move_to) {
            plan.output += std::format("Updated {}\n", hunk.path);
            continue;
        }
        if (*hunk.move_to == hunk.path) {
            plan.output += std::format("Updated {}\n", hunk.path);
            continue;
        }
        auto& destination = load(*hunk.move_to);
        if (destination.exists) {
            error = std::format(
                "Cannot move {} to {}: the destination already exists.", hunk.path, *hunk.move_to);
            return std::nullopt;
        }
        destination.exists = true;
        destination.document = std::move(source.document);
        source.exists = false;
        source.document = {};
        const bool changed = !hunk.segments.empty();
        plan.output += changed
            ? std::format("Updated and moved {} to {}\n", hunk.path, *hunk.move_to)
            : std::format("Moved {} to {}\n", hunk.path, *hunk.move_to);
    }

    for (auto& [path, node] : files) {
        if (node.exists) {
            const auto content = join_document(node.document);
            if (!node.existed || content != node.original) {
                plan.actions.push_back(CodexAction{CodexAction::Kind::Write, path, content});
            }
            continue;
        }
        if (node.existed) {
            plan.actions.push_back(CodexAction{CodexAction::Kind::Remove, path, {}});
        }
    }
    // Writes land before removals so a move still has its source if the
    // destination write fails.
    std::stable_partition(plan.actions.begin(), plan.actions.end(), [](const CodexAction& action) {
        return action.kind == CodexAction::Kind::Write;
    });
    return plan;
}

} // namespace core::tools
