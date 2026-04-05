#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/utils/JsonWriter.hpp"
#include "core/utils/JsonUtils.hpp"

#include <cmath>
#include <limits>

using namespace core::utils;

// -----------------------------------------------------------------------------
// Basic primitive emitters
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: raw() appends unmodified text", "[json_writer]") {
    JsonWriter w;
    w.raw("hello").raw(" ").raw("world");
    CHECK(w.view() == "hello world");
}

TEST_CASE("JsonWriter: comma() appends comma", "[json_writer]") {
    JsonWriter w;
    w.raw("a").comma().raw("b");
    CHECK(w.view() == "a,b");
}

TEST_CASE("JsonWriter: null_val() writes null", "[json_writer]") {
    JsonWriter w;
    w.null_val();
    CHECK(w.view() == "null");
}

TEST_CASE("JsonWriter: boolean() writes true/false", "[json_writer]") {
    JsonWriter w1;
    w1.boolean(true);
    CHECK(w1.view() == "true");

    JsonWriter w2;
    w2.boolean(false);
    CHECK(w2.view() == "false");
}

// -----------------------------------------------------------------------------
// Integer number serialization
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: number() handles positive integers", "[json_writer]") {
    JsonWriter w;
    w.number(42);
    CHECK(w.view() == "42");
}

TEST_CASE("JsonWriter: number() handles zero", "[json_writer]") {
    JsonWriter w;
    w.number(0);
    CHECK(w.view() == "0");
}

TEST_CASE("JsonWriter: number() handles negative integers", "[json_writer]") {
    JsonWriter w;
    w.number(-123);
    CHECK(w.view() == "-123");
}

TEST_CASE("JsonWriter: number() handles large integers", "[json_writer]") {
    JsonWriter w;
    w.number(9223372036854775807LL);  // max int64_t
    CHECK(w.view() == "9223372036854775807");

    JsonWriter w2;
    w2.number(-9223372036854775807LL - 1);  // min int64_t
    CHECK(w2.view() == "-9223372036854775808");
}

TEST_CASE("JsonWriter: number() handles different integral types", "[json_writer]") {
    JsonWriter w;
    w.number(static_cast<int8_t>(-128));
    CHECK(w.view() == "-128");

    JsonWriter w2;
    w2.number(static_cast<uint8_t>(255));
    CHECK(w2.view() == "255");

    JsonWriter w3;
    w3.number(static_cast<int16_t>(32767));
    CHECK(w3.view() == "32767");

    JsonWriter w4;
    w4.number(static_cast<uint32_t>(4294967295U));
    CHECK(w4.view() == "4294967295");
}

// -----------------------------------------------------------------------------
// Floating-point number serialization (primary new functionality)
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: number() handles positive doubles", "[json_writer]") {
    JsonWriter w;
    w.number(3.14159);
    auto result = std::string(w.view());
    // %.6g format - should be compact representation
    CHECK((result == "3.14159" || result.find("3.1415") == 0));
}

TEST_CASE("JsonWriter: number() handles negative doubles", "[json_writer]") {
    JsonWriter w;
    w.number(-273.15);
    auto result = std::string(w.view());
    CHECK(result.find("-273.15") != std::string::npos);
}

TEST_CASE("JsonWriter: number() handles zero as double", "[json_writer]") {
    JsonWriter w;
    w.number(0.0);
    auto result = std::string(w.view());
    CHECK((result == "0" || result == "0.0"));
}

TEST_CASE("JsonWriter: number() handles very small doubles", "[json_writer]") {
    JsonWriter w;
    w.number(1e-10);
    auto result = std::string(w.view());
    // Scientific notation or decimal
    CHECK((result.find('e') != std::string::npos || result.find('E') != std::string::npos ||
           std::stod(result) == Catch::Approx(1e-10)));
}

TEST_CASE("JsonWriter: number() handles very large doubles", "[json_writer]") {
    JsonWriter w;
    w.number(1e20);
    auto result = std::string(w.view());
    CHECK(std::stod(result) == Catch::Approx(1e20));
}

TEST_CASE("JsonWriter: number() handles scientific notation values", "[json_writer]") {
    JsonWriter w;
    w.number(6.022e23);
    auto result = std::string(w.view());
    CHECK(std::stod(result) == Catch::Approx(6.022e23));
}

TEST_CASE("JsonWriter: number() handles double with specified precision", "[json_writer]") {
    JsonWriter w;
    w.number(3.141592653589793, 3);
    auto result = std::string(w.view());
    // With precision 3, should round to ~3.14
    CHECK(std::stod(result) == Catch::Approx(3.14).epsilon(0.01));
}

TEST_CASE("JsonWriter: number() handles high precision doubles", "[json_writer]") {
    JsonWriter w;
    w.number(3.141592653589793, 15);
    auto result = std::string(w.view());
    // Should preserve more digits
    CHECK(std::stod(result) == Catch::Approx(3.141592653589793));
}

TEST_CASE("JsonWriter: number() handles precision of 1", "[json_writer]") {
    JsonWriter w;
    w.number(123.456, 1);
    auto result = std::string(w.view());
    // With 1 significant digit, should be around 100
    double val = std::stod(result);
    CHECK(val >= 50.0);
    CHECK(val <= 200.0);
}

// -----------------------------------------------------------------------------
// Key-value floating-point helpers (new functionality)
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: kv_float() writes key with double value", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_float("temperature", 0.7);
    }
    auto result = std::string(w.view());
    CHECK(result.find("\"temperature\":") != std::string::npos);
    // Extract the value after colon but before closing brace
    auto colon_pos = result.find(':');
    REQUIRE(colon_pos != std::string::npos);
    auto brace_pos = result.find('}', colon_pos);
    double value = std::stod(result.substr(colon_pos + 1, brace_pos - colon_pos - 1));
    CHECK(value == Catch::Approx(0.7));
}

TEST_CASE("JsonWriter: kv_float() with custom precision", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_float("pi", 3.14159265359, 4);
    }
    auto result = std::string(w.view());
    CHECK(result.find("\"pi\":") != std::string::npos);
    auto colon_pos = result.find(':');
    REQUIRE(colon_pos != std::string::npos);
    auto brace_pos = result.find('}', colon_pos);
    double value = std::stod(result.substr(colon_pos + 1, brace_pos - colon_pos - 1));
    CHECK(value == Catch::Approx(3.142).epsilon(0.001));
}

TEST_CASE("JsonWriter: kv_float() handles zero", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_float("zero", 0.0);
    }
    auto result = std::string(w.view());
    CHECK(result.find("\"zero\":") != std::string::npos);
}

TEST_CASE("JsonWriter: kv_float() handles negative values", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_float("negative", -273.15);
    }
    auto result = std::string(w.view());
    auto colon_pos = result.find(':');
    REQUIRE(colon_pos != std::string::npos);
    auto brace_pos = result.find('}', colon_pos);
    double value = std::stod(result.substr(colon_pos + 1, brace_pos - colon_pos - 1));
    CHECK(value == Catch::Approx(-273.15));
}

// -----------------------------------------------------------------------------
// String escaping
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: str() wraps in quotes", "[json_writer]") {
    JsonWriter w;
    w.str("hello");
    CHECK(w.view() == "\"hello\"");
}

TEST_CASE("JsonWriter: str() escapes double quotes", "[json_writer]") {
    JsonWriter w;
    w.str(R"(say "hello")");
    CHECK(w.view() == R"("say \"hello\"")");
}

TEST_CASE("JsonWriter: str() escapes backslash", "[json_writer]") {
    JsonWriter w;
    w.str(R"(C:\path\to\file)");
    CHECK(w.view() == R"("C:\\path\\to\\file")");
}

TEST_CASE("JsonWriter: str() escapes control characters", "[json_writer]") {
    JsonWriter w;
    w.str("line1\nline2\ttab");
    auto result = std::string(w.view());
    CHECK(result.find("\\n") != std::string::npos);
    CHECK(result.find("\\t") != std::string::npos);
}

TEST_CASE("JsonWriter: str() handles empty string", "[json_writer]") {
    JsonWriter w;
    w.str("");
    CHECK(w.view() == "\"\"");
}

TEST_CASE("JsonWriter: str() handles unicode/UTF-8 content", "[json_writer]") {
    JsonWriter w;
    w.str("Hello 世界 🌍");
    auto result = std::string(w.view());
    // UTF-8 content should pass through unescaped
    CHECK(result == "\"Hello 世界 🌍\"");
}

// -----------------------------------------------------------------------------
// Key methods
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: key() writes quoted key with colon", "[json_writer]") {
    JsonWriter w;
    w.key("name");
    CHECK(w.view() == "\"name\":");
}

TEST_CASE("JsonWriter: key() escapes special characters in key", "[json_writer]") {
    JsonWriter w;
    w.key("key\"with\\quotes");
    CHECK(w.view() == "\"key\\\"with\\\\quotes\":");
}

// -----------------------------------------------------------------------------
// Key-value shortcuts
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: kv_str() writes string key-value", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_str("name", "Alice");
    }
    CHECK(w.view() == R"({"name":"Alice"})");
}

TEST_CASE("JsonWriter: kv_raw() writes raw key-value", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_raw("value", "123");
    }
    CHECK(w.view() == R"({"value":123})");
}

TEST_CASE("JsonWriter: kv_bool() writes boolean key-value", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_bool("active", true);
    }
    CHECK(w.view() == R"({"active":true})");
}

TEST_CASE("JsonWriter: kv_num() writes integer key-value", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_num("count", 42);
    }
    CHECK(w.view() == R"({"count":42})");
}

TEST_CASE("JsonWriter: kv_null() writes null key-value", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_null("data");
    }
    CHECK(w.view() == R"({"data":null})");
}

// -----------------------------------------------------------------------------
// RAII scope guards
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: object() scope writes braces", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
    }
    CHECK(w.view() == "{}");
}

TEST_CASE("JsonWriter: array() scope writes brackets", "[json_writer]") {
    JsonWriter w;
    {
        auto _arr = w.array();
    }
    CHECK(w.view() == "[]");
}

TEST_CASE("JsonWriter: nested scopes work correctly", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_str("name", "outer");
        w.comma();
        w.key("inner");
        {
            auto _inner = w.object();
            w.kv_num("value", 42);
        }
    }
    CHECK(w.view() == R"({"name":"outer","inner":{"value":42}})");
}

TEST_CASE("JsonWriter: multiple nested levels", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.key("level1");
        {
            auto _l1 = w.object();
            w.key("level2");
            {
                auto _l2 = w.object();
                w.key("level3");
                {
                    auto _l3 = w.array();
                    w.number(1).comma().number(2).comma().number(3);
                }
            }
        }
    }
    CHECK(w.view() == R"({"level1":{"level2":{"level3":[1,2,3]}}})");
}

TEST_CASE("JsonWriter: array with multiple elements", "[json_writer]") {
    JsonWriter w;
    {
        auto _arr = w.array();
        w.str("a").comma().str("b").comma().str("c");
    }
    CHECK(w.view() == R"(["a","b","c"])");
}

// -----------------------------------------------------------------------------
// Extraction methods
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: take() moves string out", "[json_writer]") {
    JsonWriter w;
    w.str("test");
    auto s = std::move(w).take();
    CHECK(s == "\"test\"");
}

TEST_CASE("JsonWriter: view() provides non-owning view", "[json_writer]") {
    JsonWriter w;
    w.str("hello");
    std::string_view v = w.view();
    CHECK(v == "\"hello\"");
    // Can still use writer after view
    w.str("world");
    CHECK(w.view() == "\"hello\"\"world\"");
}

// -----------------------------------------------------------------------------
// Complex JSON structures (integration tests)
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: complex nested object with all types", "[json_writer]") {
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_str("name", "test");
        w.comma();
        w.kv_num("count", 42);
        w.comma();
        w.kv_bool("enabled", true);
        w.comma();
        w.kv_null("optional");
        w.comma();
        w.kv_float("temperature", 0.75);
        w.comma();
        w.key("tags");
        {
            auto _arr = w.array();
            w.str("json").comma().str("test");
        }
        w.comma();
        w.key("config");
        {
            auto _cfg = w.object();
            w.kv_num("max_tokens", 2048);
            w.comma();
            w.kv_float("top_p", 0.9, 2);
        }
    }

    auto result = std::string(w.view());
    // Verify structure contains expected elements
    CHECK(result.find("\"name\":") != std::string::npos);
    CHECK(result.find("\"count\":42") != std::string::npos);
    CHECK(result.find("\"enabled\":true") != std::string::npos);
    CHECK(result.find("\"optional\":null") != std::string::npos);
    CHECK(result.find("\"temperature\":") != std::string::npos);
    CHECK(result.find("\"tags\":") != std::string::npos);
    CHECK(result.find("\"config\":") != std::string::npos);
}

TEST_CASE("JsonWriter: LLM request-like structure", "[json_writer]") {
    // Simulates a typical OpenAI API request structure
    JsonWriter w;
    {
        auto _obj = w.object();
        w.kv_str("model", "gpt-4");
        w.comma();
        w.key("messages");
        {
            auto _arr = w.array();
            {
                auto _msg = w.object();
                w.kv_str("role", "system");
                w.comma();
                w.kv_str("content", "You are a helpful assistant.");
            }
            w.comma();
            {
                auto _msg = w.object();
                w.kv_str("role", "user");
                w.comma();
                w.kv_str("content", "Hello!");
            }
        }
        w.comma();
        w.kv_float("temperature", 0.7);
        w.comma();
        w.kv_num("max_tokens", 150);
    }

    auto result = std::string(w.view());
    CHECK(result.find("\"model\":\"gpt-4\"") != std::string::npos);
    CHECK(result.find("\"messages\"") != std::string::npos);
    CHECK(result.find("\"role\":\"system\"") != std::string::npos);
    CHECK(result.find("\"temperature\":") != std::string::npos);
}

TEST_CASE("JsonWriter: array of objects", "[json_writer]") {
    JsonWriter w;
    {
        auto _arr = w.array();
        {
            auto _obj = w.object();
            w.kv_num("id", 1);
        }
        w.comma();
        {
            auto _obj = w.object();
            w.kv_num("id", 2);
        }
        w.comma();
        {
            auto _obj = w.object();
            w.kv_num("id", 3);
        }
    }
    CHECK(w.view() == R"([{"id":1},{"id":2},{"id":3}])");
}

// -----------------------------------------------------------------------------
// JsonUtils tests
// -----------------------------------------------------------------------------

TEST_CASE("JsonUtils: escape_json_string handles basic text", "[json_writer]") {
    CHECK(escape_json_string("hello") == "hello");
}

TEST_CASE("JsonUtils: escape_json_string escapes quotes", "[json_writer]") {
    // A single quote gets escaped to \"
    CHECK(escape_json_string("\"") == "\\\"");
}

TEST_CASE("JsonUtils: escape_json_string escapes backslash", "[json_writer]") {
    CHECK(escape_json_string("\\") == "\\\\");
}

TEST_CASE("JsonUtils: escape_json_string escapes newlines", "[json_writer]") {
    auto result = escape_json_string("line1\nline2");
    CHECK(result.find("\\n") != std::string::npos);
}

TEST_CASE("JsonUtils: escape_json_string handles empty string", "[json_writer]") {
    CHECK(escape_json_string("") == "");
}

TEST_CASE("JsonUtils: escape_json_string handles all control chars", "[json_writer]") {
    std::string with_controls;
    for (int i = 0; i < 32; ++i) {
        with_controls += static_cast<char>(i);
    }
    auto result = escape_json_string(with_controls);
    // All control chars should be escaped (result should be much longer)
    CHECK(result.length() > with_controls.length());
}

// -----------------------------------------------------------------------------
// Pre-reservation and buffer behavior
// -----------------------------------------------------------------------------

TEST_CASE("JsonWriter: custom reservation size", "[json_writer]") {
    // Should compile and work with custom reserve
    JsonWriter w(1024);
    w.str("test");
    CHECK(w.view() == "\"test\"");
}

TEST_CASE("JsonWriter: handles repeated operations efficiently", "[json_writer]") {
    JsonWriter w;
    for (int i = 0; i < 1000; ++i) {
        w.number(i);
        w.comma();
    }
    auto result = w.view();
    CHECK(result.length() > 1000);  // Should have grown
}
