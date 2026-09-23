#include <catch2/catch_test_macros.hpp>

#include "core/tools/ToolDiffUtils.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct FileLine {
    std::string text;
    bool        terminated = false;
};

struct ChangeCounts {
    std::size_t deleted = 0;
    std::size_t added = 0;
};

ChangeCounts count_changes(std::string_view patch) {
    ChangeCounts counts;
    std::size_t start = 0;
    while (start < patch.size()) {
        const std::size_t end = patch.find('\n', start);
        const std::string_view line = end == std::string_view::npos
            ? patch.substr(start)
            : patch.substr(start, end - start);
        if (line.starts_with("-") && !line.starts_with("--- ")) {
            ++counts.deleted;
        } else if (line.starts_with("+") && !line.starts_with("+++ ")) {
            ++counts.added;
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return counts;
}

std::size_t count_hunks(std::string_view patch) {
    std::size_t count = 0;
    std::size_t start = 0;
    while (start < patch.size()) {
        const std::size_t end = patch.find('\n', start);
        const std::string_view line = end == std::string_view::npos
            ? patch.substr(start)
            : patch.substr(start, end - start);
        count += line.starts_with("@@ ") ? 1 : 0;
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return count;
}

std::size_t reference_edit_distance(const std::vector<std::string>& old_lines,
                                    const std::vector<std::string>& new_lines) {
    std::vector<std::size_t> previous(new_lines.size() + 1, 0);
    std::vector<std::size_t> current(new_lines.size() + 1, 0);
    for (const auto& old_line : old_lines) {
        for (std::size_t j = 0; j < new_lines.size(); ++j) {
            current[j + 1] = old_line == new_lines[j]
                ? previous[j] + 1
                : std::max(previous[j + 1], current[j]);
        }
        std::swap(previous, current);
        std::ranges::fill(current, 0);
    }
    return old_lines.size() + new_lines.size() - 2 * previous.back();
}

std::string join_lines(const std::vector<std::string>& lines) {
    std::string text;
    for (const auto& line : lines) {
        text += line;
        text.push_back('\n');
    }
    return text;
}

std::vector<FileLine> split_file(std::string_view text) {
    std::vector<FileLine> lines;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t newline = text.find('\n', start);
        if (newline == std::string_view::npos) {
            lines.push_back({std::string(text.substr(start)), false});
            break;
        }
        lines.push_back({std::string(text.substr(start, newline - start)), true});
        start = newline + 1;
    }
    return lines;
}

bool parse_number(std::string_view text, std::size_t& position, std::size_t& value) {
    const char* begin = text.data() + position;
    const char* end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || parsed_end == begin) {
        return false;
    }
    position = static_cast<std::size_t>(parsed_end - text.data());
    return true;
}

struct HunkHeader {
    std::size_t old_start = 0;
    std::size_t old_count = 0;
    std::size_t new_start = 0;
    std::size_t new_count = 0;
};

bool parse_hunk_header(std::string_view line, HunkHeader& header) {
    if (!line.starts_with("@@ -")) {
        return false;
    }
    std::size_t position = 4;
    if (!parse_number(line, position, header.old_start)
        || position >= line.size() || line[position++] != ','
        || !parse_number(line, position, header.old_count)
        || position >= line.size() || line[position++] != ' '
        || position >= line.size() || line[position++] != '+'
        || !parse_number(line, position, header.new_start)
        || position >= line.size() || line[position++] != ','
        || !parse_number(line, position, header.new_count)) {
        return false;
    }
    return line.substr(position) == " @@";
}

std::optional<std::string> apply_unified_diff(std::string_view before,
                                              std::string_view patch) {
    const auto original = split_file(before);
    std::vector<FileLine> result;
    std::vector<std::string_view> patch_lines;
    std::size_t start = 0;
    while (start < patch.size()) {
        const std::size_t newline = patch.find('\n', start);
        patch_lines.push_back(newline == std::string_view::npos
            ? patch.substr(start)
            : patch.substr(start, newline - start));
        if (newline == std::string_view::npos) {
            break;
        }
        start = newline + 1;
    }

    std::size_t input_position = 0;
    std::size_t old_position = 0;
    while (input_position < patch_lines.size()) {
        HunkHeader header;
        if (!parse_hunk_header(patch_lines[input_position], header)) {
            ++input_position;
            continue;
        }
        ++input_position;

        const std::size_t old_hunk_start = header.old_count == 0
            ? header.old_start
            : header.old_start - 1;
        const std::size_t new_hunk_start = header.new_count == 0
            ? header.new_start
            : header.new_start - 1;
        if (old_hunk_start < old_position || old_hunk_start > original.size()) {
            return std::nullopt;
        }
        while (old_position < old_hunk_start) {
            result.push_back(original[old_position++]);
        }
        if (new_hunk_start != result.size()) {
            return std::nullopt;
        }

        std::size_t consumed_old = 0;
        std::size_t consumed_new = 0;
        char previous_operation = '\0';
        while (consumed_old < header.old_count
               || consumed_new < header.new_count
               || (input_position < patch_lines.size()
                   && patch_lines[input_position]
                       == "\\ No newline at end of file")) {
            if (input_position >= patch_lines.size()) {
                return std::nullopt;
            }
            const std::string_view line = patch_lines[input_position++];
            if (line == "\\ No newline at end of file") {
                if (previous_operation == ' ' || previous_operation == '+') {
                    if (result.empty()) {
                        return std::nullopt;
                    }
                    result.back().terminated = false;
                }
                if (previous_operation == ' ' || previous_operation == '-') {
                    if (old_position == 0 || original[old_position - 1].terminated) {
                        return std::nullopt;
                    }
                }
                continue;
            }
            if (line.empty()) {
                return std::nullopt;
            }

            const char operation = line.front();
            const std::string_view content = line.substr(1);
            if (operation == ' ' || operation == '-') {
                if (consumed_old >= header.old_count
                    || old_position >= original.size()
                    || original[old_position].text != content) {
                    return std::nullopt;
                }
                if (operation == ' ') {
                    if (consumed_new >= header.new_count) {
                        return std::nullopt;
                    }
                    result.push_back(original[old_position]);
                    ++consumed_new;
                }
                ++old_position;
                ++consumed_old;
            } else if (operation == '+') {
                if (consumed_new >= header.new_count) {
                    return std::nullopt;
                }
                result.push_back({std::string(content), true});
                ++consumed_new;
            } else {
                return std::nullopt;
            }
            previous_operation = operation;
        }
    }

    while (old_position < original.size()) {
        result.push_back(original[old_position++]);
    }

    std::string output;
    for (const FileLine& line : result) {
        output += line.text;
        if (line.terminated) {
            output.push_back('\n');
        }
    }
    return output;
}

} // namespace

TEST_CASE("tool diff keeps unchanged lines as context around a local edit",
          "[tools][diff][regression]") {
    std::string before;
    std::string after;
    for (std::size_t i = 0; i < 100; ++i) {
        before += std::format("row {}\n", i);
        after += std::format("{} {}\n", i == 49 ? "updated" : "row", i);
    }

    const auto diff = core::tools::detail::build_unified_diff(
        "large.cpp", before, after);
    REQUIRE(diff.has_value());
    CHECK(diff->find("@@ -47,7 +47,7 @@\n") != std::string::npos);
    CHECK(diff->find(" row 48\n") != std::string::npos);
    CHECK(diff->find("-row 49\n") != std::string::npos);
    CHECK(diff->find("+updated 49\n") != std::string::npos);
    CHECK(diff->find("-row 0\n") == std::string::npos);
    CHECK(diff->find("+row 0\n") == std::string::npos);
    const auto applied = apply_unified_diff(before, *diff);
    REQUIRE(applied.has_value());
    CHECK(*applied == after);
    const auto counts = count_changes(*diff);
    CHECK(counts.deleted == 1);
    CHECK(counts.added == 1);
}

TEST_CASE("tool diff preserves CRLF and missing-final-newline semantics",
          "[tools][diff][regression]") {
    const std::string before = "first\r\nkeep\r\ntarget\r\nlast";
    const std::string after = "first\r\nkeep\r\nchanged\r\nlast";
    const auto diff = core::tools::detail::build_unified_diff(
        "windows.txt", before, after);

    REQUIRE(diff.has_value());
    CHECK(diff->find(" first\r\n") != std::string::npos);
    CHECK(diff->find("-target\r\n") != std::string::npos);
    CHECK(diff->find("+changed\r\n") != std::string::npos);
    CHECK(diff->find(" last\n\\ No newline at end of file\n") != std::string::npos);
    const auto applied = apply_unified_diff(before, *diff);
    REQUIRE(applied.has_value());
    CHECK(*applied == after);
    const auto counts = count_changes(*diff);
    CHECK(counts.deleted == 1);
    CHECK(counts.added == 1);

    // A final newline is part of the line record and must not be silently
    // normalized into an unchanged line.
    const auto terminator_change =
        core::tools::detail::build_unified_diff("eof.txt", "line", "line\n");
    REQUIRE(terminator_change.has_value());
    CHECK(terminator_change->find("-line\n\\ No newline at end of file\n")
          != std::string::npos);
    CHECK(terminator_change->find("+line\n") != std::string::npos);
}

TEST_CASE("tool diff separates distant edits into correctly numbered hunks",
          "[tools][diff][regression]") {
    std::string before;
    std::string after;
    for (std::size_t i = 0; i < 40; ++i) {
        before += std::format("row {}\n", i);
        after += std::format("{} {}\n",
                             (i == 4 || i == 35) ? "updated" : "row",
                             i);
    }

    const auto diff = core::tools::detail::build_unified_diff(
        "separated.cpp", before, after);
    REQUIRE(diff.has_value());
    CHECK(count_hunks(*diff) == 2);
    CHECK(diff->find("@@ -2,7 +2,7 @@\n") != std::string::npos);
    CHECK(diff->find("@@ -33,7 +33,7 @@\n") != std::string::npos);
    const auto applied = apply_unified_diff(before, *diff);
    REQUIRE(applied.has_value());
    CHECK(*applied == after);
    const auto counts = count_changes(*diff);
    CHECK(counts.deleted == 2);
    CHECK(counts.added == 2);
}

TEST_CASE("Myers tool diff matches shortest edit distance for repeated lines",
          "[tools][diff][property]") {
    std::uint32_t state = 0xC0FFEEu;
    const auto next_value = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };

    for (std::size_t sample = 0; sample < 160; ++sample) {
        std::vector<std::string> old_lines;
        std::vector<std::string> new_lines;
        const std::size_t old_count = next_value() % 10;
        const std::size_t new_count = next_value() % 10;
        for (std::size_t i = 0; i < old_count; ++i) {
            old_lines.push_back(std::format("v{}", next_value() % 5));
        }
        for (std::size_t i = 0; i < new_count; ++i) {
            new_lines.push_back(std::format("v{}", next_value() % 5));
        }

        const std::string before = join_lines(old_lines);
        const std::string after = join_lines(new_lines);
        if (before == after) {
            CHECK_FALSE(core::tools::detail::build_unified_diff(
                "random.txt", before, after));
            continue;
        }

        const auto diff = core::tools::detail::build_unified_diff(
            "random.txt", before, after);
        REQUIRE(diff.has_value());
        const auto counts = count_changes(*diff);
        CHECK(counts.deleted + counts.added
              == reference_edit_distance(old_lines, new_lines));
        const auto applied = apply_unified_diff(before, *diff);
        REQUIRE(applied.has_value());
        CHECK(*applied == after);
    }

    std::vector<std::vector<std::string>> sequences{{}};
    for (std::size_t length = 1; length <= 4; ++length) {
        std::size_t sequence_count = 1;
        for (std::size_t i = 0; i < length; ++i) {
            sequence_count *= 3;
        }
        for (std::size_t encoded = 0; encoded < sequence_count; ++encoded) {
            std::size_t value = encoded;
            std::vector<std::string> sequence(length);
            for (std::size_t i = 0; i < length; ++i) {
                sequence[i] = std::format("v{}", value % 3);
                value /= 3;
            }
            sequences.push_back(std::move(sequence));
        }
    }

    for (const auto& old_lines : sequences) {
        for (const auto& new_lines : sequences) {
            const std::string before = join_lines(old_lines);
            const std::string after = join_lines(new_lines);
            if (before == after) {
                continue;
            }
            const auto diff = core::tools::detail::build_unified_diff(
                "exhaustive.txt", before, after);
            REQUIRE(diff.has_value());
            const auto counts = count_changes(*diff);
            CHECK(counts.deleted + counts.added
                  == reference_edit_distance(old_lines, new_lines));
            const auto applied = apply_unified_diff(before, *diff);
            REQUIRE(applied.has_value());
            CHECK(*applied == after);
        }
    }
}

TEST_CASE("tool diff omits pathological comparisons within a fixed work budget",
          "[tools][diff][limits]") {
    std::string before;
    std::string after;
    for (std::size_t i = 0; i < 2'000; ++i) {
        before += std::format("old-{}\n", i);
        after += std::format("new-{}\n", i);
    }

    CHECK(before.size() + after.size()
          < core::tools::detail::kMaxToolDiffInputBytes);
    CHECK_FALSE(core::tools::detail::build_unified_diff(
        "adversarial.txt", before, after));
}

TEST_CASE("tool diff rejects oversized unchanged inputs before comparing them",
          "[tools][diff][limits]") {
    const std::string oversized(
        core::tools::detail::kMaxToolDiffInputBytes + 1, 'x');
    CHECK_FALSE(core::tools::detail::build_unified_diff(
        "oversized.txt", oversized, oversized));
}

TEST_CASE("tool diff omits output that exceeds the result-size budget",
          "[tools][diff][limits]") {
    const std::string before(262'143, 'a');
    const std::string after(262'143, 'b');
    CHECK(before.size() + after.size()
          <= core::tools::detail::kMaxToolDiffInputBytes);
    CHECK_FALSE(core::tools::detail::build_unified_diff(
        "large-lines.txt", before, after));
}
