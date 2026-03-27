#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/ContextMentions.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path make_temp_dir(const std::string& label) {
    const auto stamp = std::to_string(static_cast<long long>(std::rand()));
    fs::path path = fs::temp_directory_path() / (label + "_" + stamp);
    fs::create_directories(path);
    return path;
}

void write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << text;
}

} // namespace

TEST_CASE("Context mentions expand file contents with line numbers", "[context]") {
    const fs::path sandbox = make_temp_dir("filo_context_file");
    write_text(sandbox / "src" / "main.cpp", "int main() {\n  return 0;\n}\n");

    const std::string expanded =
        core::context::expand_mentions("Inspect @src/main.cpp please", sandbox);

    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("[Context for src/main.cpp]"));
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("1 | int main() {"));
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("2 |   return 0;"));

    fs::remove_all(sandbox);
}

TEST_CASE("Context mentions support quoted paths and ignore email addresses", "[context]") {
    const fs::path sandbox = make_temp_dir("filo_context_quotes");
    write_text(sandbox / "notes" / "debug log.txt", "trace line\n");

    const std::string expanded = core::context::expand_mentions(
        R"(Mail test@example.com and inspect @"notes/debug log.txt")",
        sandbox);

    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("test@example.com"));
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("[Context for notes/debug log.txt]"));

    fs::remove_all(sandbox);
}

TEST_CASE("Context mentions expand directories, preserve punctuation, and truncate large files", "[context]") {
    const fs::path sandbox = make_temp_dir("filo_context_dir");
    write_text(sandbox / "docs" / "README.md", "hello world from filo\n");
    write_text(sandbox / "big.txt", "0123456789abcdef");

    const auto options = core::context::MentionExpansionOptions{
        .max_file_bytes = 8,
        .max_directory_entries = 8
    };

    const std::string expanded = core::context::expand_mentions(
        "Look at @docs, then @big.txt.",
        sandbox,
        options);

    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("Directory listing:"));
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("README.md"));
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("[truncated after 8 bytes]"));
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("[/Context]."));

    fs::remove_all(sandbox);
}

TEST_CASE("Context mentions detect the active unquoted mention under the cursor", "[context]") {
    const auto mention = core::context::find_active_mention("Inspect @src/ma please", 15);
    REQUIRE(mention.has_value());
    REQUIRE(mention->raw_path == "src/ma");
    REQUIRE_FALSE(mention->quoted);
}

TEST_CASE("Context mentions detect quoted mentions while typing", "[context]") {
    const auto mention = core::context::find_active_mention(
        R"(Inspect @"notes/debug l)",
        std::string_view{R"(Inspect @"notes/debug l)"}.size());
    REQUIRE(mention.has_value());
    REQUIRE(mention->raw_path == "notes/debug l");
    REQUIRE(mention->quoted);
}

TEST_CASE("Context mentions keep reserved subagent tags untouched", "[context]") {
    const fs::path sandbox = make_temp_dir("filo_context_subagent_tag");

    const std::string expanded =
        core::context::expand_mentions("Please ask @general to research this.", sandbox);

    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("@general"));
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("research this."));
    REQUIRE_FALSE(expanded.find("[Context for general]") != std::string::npos);

    fs::remove_all(sandbox);
}

TEST_CASE("Context mentions do not treat emails as active mentions", "[context]") {
    const auto mention = core::context::find_active_mention("mail test@example.com", 21);
    REQUIRE_FALSE(mention.has_value());
}

TEST_CASE("Context mention completion wraps paths with spaces", "[context]") {
    const std::string input = "Inspect @not";
    const auto mention = core::context::find_active_mention(input, input.size());
    REQUIRE(mention.has_value());

    const auto completed = core::context::apply_mention_completion(
        input, *mention, "notes/debug log.txt");

    REQUIRE(completed.text == R"(Inspect @"notes/debug log.txt")");
    REQUIRE(completed.cursor == completed.text.size());
}

TEST_CASE("Context mention completion keeps directory quotes open for continued typing", "[context]") {
    const std::string input = R"(Inspect @"notes/de)";
    const auto mention = core::context::find_active_mention(input, input.size());
    REQUIRE(mention.has_value());

    const auto completed = core::context::apply_mention_completion(
        input, *mention, "notes/debug log/");

    REQUIRE(completed.text == R"(Inspect @"notes/debug log/)");
    REQUIRE(completed.cursor == completed.text.size());
}
