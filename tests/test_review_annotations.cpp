#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/review/Annotations.hpp"

#include <array>
#include <simdjson.h>

using namespace core::review;
using Catch::Matchers::ContainsSubstring;

namespace {

Annotation comment() {
    return {
        .id = "ccbce2b6-4eb9-4ba3-b0b7-a34614970d9a",
        .path = "src/parser.cpp",
        .side = "new",
        .start_line = 12,
        .end_line = 13,
        .selected_text = "return parse(config);\n",
        .comment = "Keep the old configuration format.",
        .tone = "warning",
        .revision = "opaque-revision-1",
        .context_before = {"if (config.version == 1) {"},
        .context_after = {"}"},
    };
}

std::string replace_once(std::string text, std::string_view before,
                         std::string_view after) {
    const auto at = text.find(before);
    REQUIRE(at != std::string::npos);
    text.replace(at, before.size(), after);
    return text;
}

std::vector<Annotation> read_saved(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::object object;
    REQUIRE(parser.parse(json).get(object) == simdjson::SUCCESS);
    return parse_saved_annotations(object);
}

Annotation comment(std::string id, std::string path, std::string text) {
    auto annotation = comment();
    annotation.id = std::move(id);
    annotation.path = std::move(path);
    annotation.comment = std::move(text);
    return annotation;
}

} // namespace

TEST_CASE("Review annotation wire preserves structured feedback", "[review_annotations]") {
    auto annotation = comment();
    annotation.path = "src/配置.cpp";
    annotation.selected_text = "std::string note = \"naïve\";\n```\n````\n";
    annotation.comment = "  Keep \"配置\" exactly.\n\nAnd these spaces.  ";
    const std::array annotations{annotation};
    const std::string patch = "diff --git a/配置.cpp b/配置.cpp\n+```\n";
    const auto wire = encode_review_request(patch, annotations);

    simdjson::dom::parser parser;
    simdjson::dom::object object;
    REQUIRE(parser.parse(wire).get(object) == simdjson::SUCCESS);
    std::string_view encoded_patch;
    REQUIRE(object["patch"].get(encoded_patch) == simdjson::SUCCESS);
    CHECK(encoded_patch == patch);

    const auto parsed = parse_review_response(wire);
    REQUIRE(parsed);
    CHECK(*parsed == std::vector<Annotation>{annotation});
}

TEST_CASE("Review annotation response requires a valid versioned snapshot",
          "[review_annotations]") {
    const auto wire = encode_review_request("patch", std::array{comment()});
    const std::array invalid{
        std::string("Please fix the parser."),
        std::string("{broken"),
        std::string("[]"),
        std::string(R"({"annotations":[]})"),
        std::string(R"({"version":1,"annotations":null})"),
        std::string(R"({"version":1,"annotations":[null]})"),
        std::string(R"({"version":1,"version":1,"annotations":[]})"),
        replace_once(wire, "\"version\":1", "\"version\":2"),
        replace_once(wire, "\"version\":1", "\"version\":1.0"),
        replace_once(wire, "\"version\":1", "\"version\":\"1\""),
        replace_once(wire, "\"start_line\":12", "\"start_line\":0"),
        replace_once(wire, "\"start_line\":12", "\"start_line\":14"),
        replace_once(wire, "\"start_line\":12", "\"start_line\":12.5"),
        replace_once(wire, "\"start_line\":12", "\"start_line\":2147483648"),
        replace_once(wire, "\"side\":\"new\"", "\"side\":\"right\""),
        replace_once(wire, "\"tone\":\"warning\"", "\"tone\":\"note\""),
        replace_once(wire, "\"anchor_state\":\"attached\"", "\"anchor_state\":\"missing\""),
        replace_once(wire, "\"id\":\"ccbce2b6-4eb9-4ba3-b0b7-a34614970d9a\"", "\"id\":\"bad-id\""),
        replace_once(wire, "\"revision\":\"opaque-revision-1\"", "\"revision\":\"\""),
        replace_once(wire, "\"context_before\":[\"if (config.version == 1) {\"]", "\"context_before\":null"),
        replace_once(wire, "\"context_before\":[\"if (config.version == 1) {\"]", "\"context_before\":[42]"),
        replace_once(wire, "\"context_before\":[\"if (config.version == 1) {\"]", "\"context_before\":[\"a\\nb\"]"),
        replace_once(wire, "\"context_after\":[\"}\"]", "\"context_after\":[\"a\",\"b\",\"c\",\"d\"]"),
        replace_once(wire, "\"comment\":\"Keep the old configuration format.\"", "\"comment\":42"),
        replace_once(wire, "\"side\":\"new\"", "\"side\":\"old\",\"side\":\"new\""),
    };
    for (const auto& response : invalid) {
        INFO(response);
        CHECK_FALSE(parse_review_response(response));
    }
    CHECK_FALSE(parse_review_response(std::string(16 * 1024 * 1024 + 1, ' ')));
    auto large_context = comment();
    large_context.context_before = {std::string(600'000, 'a')};
    large_context.context_after = {std::string(600'000, 'b')};
    CHECK_FALSE(parse_review_response(encode_review_request("", std::array{large_context})));

    const auto empty = parse_review_response(R"({"version":1,"annotations":[]})");
    REQUIRE(empty);
    CHECK(empty->empty());
}

TEST_CASE("Review annotations reject invalid paths and duplicate UUIDs",
          "[review_annotations]") {
    auto annotation = comment();
    for (const std::string path : {"", "src/\nparser.cpp", "src/\tparser.cpp"}) {
        annotation.path = path;
        INFO(path);
        CHECK_FALSE(parse_review_response(encode_review_request("", std::array{annotation})));
    }
    for (const std::string path : {"/tmp/parser.cpp", "../parser.cpp", "src/配置.cpp"}) {
        annotation.path = path;
        const auto parsed = parse_review_response(encode_review_request("", std::array{annotation}));
        REQUIRE(parsed);
        CHECK(parsed->front().path == path);
    }

    auto uppercase = comment();
    uppercase.id = "CCBCE2B6-4EB9-4BA3-B0B7-A34614970D9A";
    const auto normalized = parse_review_response(encode_review_request("", std::array{uppercase}));
    REQUIRE(normalized);
    CHECK(normalized->front().id == comment().id);
    CHECK_FALSE(parse_review_response(encode_review_request("", std::array{comment(), uppercase})));
}

TEST_CASE("Review prompt renders the given comments and no others",
          "[review_annotations]") {
    CHECK(render_annotation_prompt({}).empty());
    const std::array annotations{comment()};
    const auto first_prompt = render_annotation_prompt(annotations);
    CHECK_FALSE(first_prompt.empty());
    CHECK(render_annotation_prompt(annotations) == first_prompt);

    auto second = comment();
    second.id = "9b786cc1-c084-4c38-b0fb-bbd482f7e600";
    second.path = "tests/parser.cpp";
    second.comment = "Cover the legacy format here.";
    const auto prompt = render_annotation_prompt(std::array{comment(), second});
    CHECK_THAT(prompt, ContainsSubstring("tests/parser.cpp"));
    CHECK_THAT(prompt, ContainsSubstring(second.comment));
    CHECK_THAT(prompt, ContainsSubstring("src/parser.cpp"));
    CHECK_THAT(prompt, ContainsSubstring(comment().comment));

    const auto remaining_prompt = render_annotation_prompt(std::array{second});
    CHECK_THAT(remaining_prompt, ContainsSubstring(second.comment));
    CHECK(remaining_prompt.find(comment().comment) == std::string::npos);
}

TEST_CASE("Review prompt preserves multiline Unicode quotes and fence boundaries",
          "[review_annotations]") {
    auto annotation = comment();
    annotation.side = "old";
    annotation.selected_text = "\t配置 = \"naïve\";\n```\n````\n  ";
    annotation.comment = "  Preserve \"配置\".\n\n```exact```  ";
    const auto prompt = render_annotation_prompt(std::array{annotation});
    CHECK_THAT(prompt, ContainsSubstring("old side, lines 12-13"));
    CHECK_THAT(prompt, ContainsSubstring("`````\n" + annotation.selected_text + "\n`````"));
    CHECK(prompt.ends_with(annotation.comment));
}

TEST_CASE("Review annotations preserve selections without a comment", "[review_annotations]") {
    auto annotation = comment();
    annotation.comment.clear();
    const auto parsed = parse_review_response(encode_review_request("", std::array{annotation}));
    REQUIRE(parsed);
    CHECK(parsed->front() == annotation);
    CHECK_THAT(render_annotation_prompt(*parsed), ContainsSubstring("(no comment)"));
}

TEST_CASE("A returned review delivers only the comments its turn did not hold",
          "[review_annotations]") {
    const auto shown = comment("0f6f8a39-1a7e-4c8b-9a51-2b1f5c3d7e10", "src/parser.cpp", "Keep the old format.");
    const auto elsewhere = comment("5d0e2c71-8b3a-4f69-a1c4-7e2d9b6f0a35", "docs/guide.md", "Mention the flag.");
    const auto removed = comment("9b786cc1-c084-4c38-b0fb-bbd482f7e600", "src/parser.cpp", "Drop this branch.");
    auto edited = shown;
    edited.comment = "Keep the old format, version 1 only.";
    const auto added = comment("ccbce2b6-4eb9-4ba3-b0b7-a34614970d9b", "src/parser.cpp", "Cover the legacy path.");

    const std::array held{shown, elsewhere, removed};
    const std::array sent{shown, removed};
    const auto review = merge_review(held, sent, {edited, added});

    CHECK(review.kept == std::vector{elsewhere, edited, added});
    CHECK(review.added == std::vector{added});
    CHECK(merge_review(review.kept, review.kept, review.kept).added.empty());
    CHECK(merge_review({}, {}, {added}).added == std::vector{added});
}

TEST_CASE("Saved review comments round-trip, and a malformed list is dropped whole",
          "[review_annotations]") {
    std::string saved = "{\"annotations\":";
    append_annotations_json(saved, std::array{comment()});
    saved += '}';
    CHECK(read_saved(saved) == std::vector{comment()});

    CHECK(read_saved("{}").empty());
    CHECK(read_saved(replace_once(saved, "\"side\":\"new\"", "\"side\":\"other\"")).empty());
}
