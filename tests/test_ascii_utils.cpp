#include <catch2/catch_test_macros.hpp>

#include "core/utils/AsciiUtils.hpp"

TEST_CASE("ascii::to_lower lowercases only ASCII uppercase letters", "[utils][ascii]") {
    REQUIRE(core::utils::ascii::to_lower('A') == 'a');
    REQUIRE(core::utils::ascii::to_lower('Z') == 'z');
    REQUIRE(core::utils::ascii::to_lower('m') == 'm');
    REQUIRE(core::utils::ascii::to_lower('0') == '0');
    REQUIRE(core::utils::ascii::to_lower('-') == '-');
}

TEST_CASE("ascii::is_alnum recognizes ASCII letters and digits", "[utils][ascii]") {
    REQUIRE(core::utils::ascii::is_alnum(static_cast<unsigned char>('a')));
    REQUIRE(core::utils::ascii::is_alnum(static_cast<unsigned char>('Z')));
    REQUIRE(core::utils::ascii::is_alnum(static_cast<unsigned char>('9')));

    REQUIRE_FALSE(core::utils::ascii::is_alnum(static_cast<unsigned char>('_')));
    REQUIRE_FALSE(core::utils::ascii::is_alnum(static_cast<unsigned char>('-')));
    REQUIRE_FALSE(core::utils::ascii::is_alnum(static_cast<unsigned char>(0xC0)));
}

TEST_CASE("ascii::iequals is ASCII case-insensitive", "[utils][ascii]") {
    REQUIRE(core::utils::ascii::iequals("file", "FILE"));
    REQUIRE(core::utils::ascii::iequals("LocalHost", "localhost"));
    REQUIRE_FALSE(core::utils::ascii::iequals("file", "files"));
    REQUIRE_FALSE(core::utils::ascii::iequals("http", "https"));
}

TEST_CASE("ascii::istarts_with matches prefix ignoring ASCII case", "[utils][ascii]") {
    REQUIRE(core::utils::ascii::istarts_with("FILE://tmp/path", "file://"));
    REQUIRE(core::utils::ascii::istarts_with("localHOST", "LOCAL"));
    REQUIRE_FALSE(core::utils::ascii::istarts_with("abc", "abcd"));
    REQUIRE_FALSE(core::utils::ascii::istarts_with("http://x", "file://"));
}
