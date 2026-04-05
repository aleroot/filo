#include <catch2/catch_test_macros.hpp>

#include "core/utils/Base64.hpp"

#include <array>
#include <string>
#include <string_view>

TEST_CASE("Base64 standard encode matches RFC 4648 vectors", "[utils][base64]") {
    REQUIRE(core::utils::Base64::encode("") == "");
    REQUIRE(core::utils::Base64::encode("f") == "Zg==");
    REQUIRE(core::utils::Base64::encode("fo") == "Zm8=");
    REQUIRE(core::utils::Base64::encode("foo") == "Zm9v");
    REQUIRE(core::utils::Base64::encode("hello world") == "aGVsbG8gd29ybGQ=");
}

TEST_CASE("Base64 standard decode parses valid payloads", "[utils][base64]") {
    const auto empty = core::utils::Base64::decode("");
    REQUIRE(empty.has_value());
    REQUIRE(*empty == "");

    const auto one = core::utils::Base64::decode("Zg==");
    REQUIRE(one.has_value());
    REQUIRE(*one == "f");

    const auto hello = core::utils::Base64::decode("aGVsbG8gd29ybGQ=");
    REQUIRE(hello.has_value());
    REQUIRE(*hello == "hello world");
}

TEST_CASE("Base64 standard decode rejects malformed input", "[utils][base64]") {
    REQUIRE_FALSE(core::utils::Base64::decode("Zg=").has_value());
    REQUIRE_FALSE(core::utils::Base64::decode("Z===").has_value());
    REQUIRE_FALSE(core::utils::Base64::decode("Zm$=").has_value());
    REQUIRE_FALSE(core::utils::Base64::decode("Zm=9").has_value());
}

TEST_CASE("Base64URL encode defaults to no padding", "[utils][base64]") {
    const std::array<std::uint8_t, 1> one_byte{0xFF};
    REQUIRE(core::utils::Base64::encode_url(one_byte) == "_w");
    REQUIRE(core::utils::Base64::encode_url(one_byte, true) == "_w==");

    const std::array<std::uint8_t, 3> three_bytes{0x00, 0x01, 0xFF};
    REQUIRE(core::utils::Base64::encode_url(three_bytes) == "AAH_");
}

TEST_CASE("Base64URL decode accepts padded and unpadded input", "[utils][base64]") {
    const auto no_padding = core::utils::Base64::decode_url("eyJmb28iOiJiYXIifQ");
    REQUIRE(no_padding.has_value());
    REQUIRE(*no_padding == R"({"foo":"bar"})");

    const auto padded = core::utils::Base64::decode_url("eyJmb28iOiJiYXIifQ==");
    REQUIRE(padded.has_value());
    REQUIRE(*padded == R"({"foo":"bar"})");
}

TEST_CASE("Base64 handles binary round-trip", "[utils][base64]") {
    const std::string input("\x00\x01\xFF\x7F", 4);
    const std::string encoded = core::utils::Base64::encode(input);
    const auto decoded = core::utils::Base64::decode(encoded);
    REQUIRE(decoded.has_value());
    REQUIRE(*decoded == input);
}

