#include <catch2/catch_test_macros.hpp>

#include "core/utils/UriUtils.hpp"

#include <string>

TEST_CASE("uri::percent_decode decodes valid percent-encoded bytes", "[utils][uri]") {
    std::string decoded;
    REQUIRE(core::utils::uri::percent_decode("file%3A%2F%2Ftmp%2Fhello%20world", decoded));
    REQUIRE(decoded == "file://tmp/hello world");
}

TEST_CASE("uri::percent_decode rejects malformed encodings", "[utils][uri]") {
    std::string decoded;
    REQUIRE_FALSE(core::utils::uri::percent_decode("%", decoded));
    REQUIRE_FALSE(core::utils::uri::percent_decode("%A", decoded));
    REQUIRE_FALSE(core::utils::uri::percent_decode("%ZZ", decoded));
}

TEST_CASE("uri::percent_encode_uri_path escapes reserved bytes and keeps path chars",
          "[utils][uri]") {
    const auto encoded =
        core::utils::uri::percent_encode_uri_path("/tmp/hello world?#.txt");
    REQUIRE(encoded == "/tmp/hello%20world%3F%23.txt");
}

TEST_CASE("uri percent encode/decode round-trip for path text", "[utils][uri]") {
    constexpr std::string_view raw = "/root/space dir/a+b@c";
    const std::string encoded = core::utils::uri::percent_encode_uri_path(raw);

    std::string decoded;
    REQUIRE(core::utils::uri::percent_decode(encoded, decoded));
    REQUIRE(decoded == raw);
}

TEST_CASE("uri::extract_http_host parses authority safely", "[utils][uri]") {
    REQUIRE(core::utils::uri::extract_http_host("https://localhost:11434/v1").value() == "localhost");
    REQUIRE(core::utils::uri::extract_http_host("http://[::1]:8080").value() == "::1");
    REQUIRE(core::utils::uri::extract_http_host("https://localhost.run/v1").value() == "localhost.run");
    REQUIRE_FALSE(core::utils::uri::extract_http_host("file:///tmp/test").has_value());
    REQUIRE_FALSE(core::utils::uri::extract_http_host("http://user@localhost:11434").has_value());
}

TEST_CASE("uri::is_loopback_http_url requires exact loopback host", "[utils][uri]") {
    REQUIRE(core::utils::uri::is_loopback_http_url("http://localhost:11434"));
    REQUIRE(core::utils::uri::is_loopback_http_url("http://127.0.0.1:11434"));
    REQUIRE(core::utils::uri::is_loopback_http_url("http://[::1]:11434"));

    REQUIRE_FALSE(core::utils::uri::is_loopback_http_url("https://localhost.run/v1"));
    REQUIRE_FALSE(core::utils::uri::is_loopback_http_url("https://127.0.0.1.example.com/v1"));
}
