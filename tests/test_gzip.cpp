#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/utils/Gzip.hpp"

#include <zlib.h>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

namespace {

namespace gzip = core::utils::gzip;

[[nodiscard]] std::string random_bytes(std::size_t size) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> byte(0, 255);
    std::string out(size, '\0');
    for (auto& c : out) c = static_cast<char>(byte(rng));
    return out;
}

[[nodiscard]] std::string repetitive_text(std::size_t size) {
    std::string out;
    while (out.size() < size) out += R"({"providers":{"claude":{"model":"claude-sonnet-4-6"}}},)";
    out.resize(size);
    return out;
}

/// The one-shot deflate the Claude client used before core::utils::gzip
/// existed: header OS byte patched after compression.
[[nodiscard]] std::string reference_one_shot(std::string_view text, int level, unsigned char os) {
    z_stream stream{};
    REQUIRE(deflateInit2(&stream, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) == Z_OK);
    std::string out(deflateBound(&stream, static_cast<uLong>(text.size())), '\0');
    stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(text.data()));
    stream.avail_in = static_cast<uInt>(text.size());
    stream.next_out = reinterpret_cast<Bytef*>(out.data());
    stream.avail_out = static_cast<uInt>(out.size());
    REQUIRE(deflate(&stream, Z_FINISH) == Z_STREAM_END);
    out.resize(stream.total_out);
    deflateEnd(&stream);
    out[9] = static_cast<char>(os);
    return out;
}

} // namespace

TEST_CASE("gzip round-trips empty, text and incompressible data", "[gzip]") {
    for (const auto& input : {std::string{}, std::string("hello"), repetitive_text(300'000),
                              random_bytes(200'000)}) {
        const auto packed = gzip::compress(input);
        CHECK(gzip::is_compressed(packed));
        const auto unpacked = gzip::decompress(packed);
        REQUIRE(unpacked.has_value());
        CHECK(*unpacked == input);
    }
    CHECK(gzip::compress(repetitive_text(300'000)).size() < 30'000);
}

TEST_CASE("gzip output is deterministic and carries the requested header", "[gzip]") {
    const auto text = repetitive_text(10'000);
    const auto packed = gzip::compress(text, {.level = 9});
    CHECK(packed == gzip::compress(text, {.level = 9}));
    REQUIRE(packed.size() > 10);
    CHECK(static_cast<unsigned char>(packed[2]) == 8);  // deflate
    CHECK(packed.substr(4, 4) == std::string(4, '\0')); // mtime 0
    CHECK(static_cast<unsigned char>(packed[9]) == gzip::kUnknownOperatingSystem);
    CHECK(static_cast<unsigned char>(gzip::compress(text, {.operating_system = 3})[9]) == 3);
}

TEST_CASE("gzip matches zlib's one-shot output byte for byte", "[gzip]") {
    // Claude request bodies moved onto this helper; the wire bytes must not change.
    const auto text = repetitive_text(50'000) + random_bytes(5'000);
    CHECK(gzip::compress(text, {.level = 6, .operating_system = 0xff})
          == reference_one_shot(text, 6, 0xff));
}

TEST_CASE("gzip reads every member of a concatenated stream", "[gzip]") {
    // What `cat a.gz b.gz` and parallel compressors produce; gunzip accepts it.
    const auto joined = gzip::decompress(gzip::compress("first ") + gzip::compress("second"));
    REQUIRE(joined.has_value());
    CHECK(*joined == "first second");
    // The output limit covers the members together, not each one.
    const auto half = repetitive_text(1'000);
    const auto two = gzip::compress(half) + gzip::compress(half);
    CHECK(gzip::decompress(two, 2'000).has_value());
    CHECK_FALSE(gzip::decompress(two, 1'999));
}

TEST_CASE("gzip rejects an out-of-range compression level", "[gzip]") {
    CHECK_THROWS_AS(gzip::compress("x", {.level = gzip::kMaxLevel + 1}), std::invalid_argument);
    CHECK_THROWS_AS(gzip::compress("x", {.level = -1}), std::invalid_argument);
    CHECK(gzip::decompress(gzip::compress("x", {.level = gzip::kMinLevel})).value_or("") == "x");
}

TEST_CASE("gzip rejects damaged, foreign and oversized streams", "[gzip]") {
    const auto text = repetitive_text(100'000);
    const auto packed = gzip::compress(text);

    SECTION("truncated") {
        const auto result = gzip::decompress(std::string_view(packed).substr(0, packed.size() / 2));
        REQUIRE_FALSE(result.has_value());
        CHECK_THAT(result.error(), Catch::Matchers::ContainsSubstring("truncated"));
        CHECK_FALSE(gzip::decompress(std::string_view(packed).substr(0, packed.size() - 1)));
        CHECK_FALSE(gzip::decompress(""));
    }
    SECTION("corrupt checksum") {
        auto corrupt = packed;
        corrupt[corrupt.size() - 5] = static_cast<char>(corrupt[corrupt.size() - 5] ^ 0x01);
        const auto result = gzip::decompress(corrupt);
        REQUIRE_FALSE(result.has_value());
        CHECK_THAT(result.error(), Catch::Matchers::ContainsSubstring("corrupt"));
    }
    SECTION("trailing data") {
        const auto result = gzip::decompress(packed + "junk");
        REQUIRE_FALSE(result.has_value());
        CHECK_THAT(result.error(), Catch::Matchers::ContainsSubstring("after the gzip stream"));
        CHECK_FALSE(gzip::decompress(packed + packed + "junk"));
        CHECK_FALSE(gzip::decompress(packed + packed.substr(0, 10)));  // truncated second member
    }
    SECTION("not gzip") {
        CHECK_FALSE(gzip::is_compressed(R"({"format":"filo-config-bundle"})"));
        CHECK_FALSE(gzip::is_compressed("\x1f"));
        CHECK_FALSE(gzip::decompress(R"({"format":"filo-config-bundle"})"));
    }
    SECTION("output limit") {
        CHECK(gzip::decompress(packed, text.size()).has_value());
        const auto result = gzip::decompress(packed, text.size() - 1);
        REQUIRE_FALSE(result.has_value());
        CHECK_THAT(result.error(), Catch::Matchers::ContainsSubstring("exceeds"));
        // A small stream that expands enormously stops at the limit.
        const auto bomb = gzip::compress(std::string(8u << 20, '\0'), {.level = 9});
        CHECK(bomb.size() < 64 * 1024);
        CHECK_FALSE(gzip::decompress(bomb, 1u << 20));
    }
}
