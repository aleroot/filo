#include <catch2/catch_test_macros.hpp>

#include "core/utils/MimeUtils.hpp"

#include <filesystem>

TEST_CASE("mime::guess_type returns octet-stream for binary payloads", "[utils][mime]") {
    const auto mime = core::utils::mime::guess_type("any.unknown", true);
    REQUIRE(mime == "application/octet-stream");
}

TEST_CASE("mime::guess_type uses cpp-httplib mapping for common extensions", "[utils][mime]") {
    REQUIRE(core::utils::mime::guess_type("doc.json", false) == "application/json");
    REQUIRE(core::utils::mime::guess_type("index.html", false) == "text/html");
}

TEST_CASE("mime::guess_type keeps project code-type overrides", "[utils][mime]") {
    REQUIRE(core::utils::mime::guess_type("main.cpp", false) == "text/x-c++src");
    REQUIRE(core::utils::mime::guess_type("script.py", false) == "text/x-python");
}

TEST_CASE("mime::guess_type normalizes extension case", "[utils][mime]") {
    REQUIRE(core::utils::mime::guess_type("MAIN.CPP", false) == "text/x-c++src");
    REQUIRE(core::utils::mime::guess_type("README.MD", false) == "text/markdown");
}
