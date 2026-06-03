#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <thread>

#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/auth/KimiOAuthFlow.hpp"
#include "core/tools/Tool.hpp"

using namespace core::llm;
using namespace core::llm::protocols;

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct ScopedEnvVar {
    std::string name;
    std::string old_value;
    bool had_value = false;

    ScopedEnvVar(std::string env_name, const std::string& value)
        : name(std::move(env_name)) {
        if (const char* existing = std::getenv(name.c_str())) {
            had_value = true;
            old_value = existing;
        }
        setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (had_value) {
            setenv(name.c_str(), old_value.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

fs::path make_temp_dir(const std::string& label) {
    const auto path = fs::temp_directory_path()
        / (label + "_" + core::auth::KimiOAuthFlow::generateDeviceId());
    fs::create_directories(path);
    return path;
}

fs::path make_temp_video_file(std::string_view filename = "filo-kimi-video.mp4") {
    const auto path = fs::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "fake-video";
    return path;
}

} // namespace

static ChatRequest make_simple_request(std::string model = "moonshot-v1-8k",
                                       std::string user_text = "Hello") {
    ChatRequest req;
    req.model = std::move(model);
    req.stream = true;
    req.messages.push_back(Message{.role = "user", .content = std::move(user_text)});
    return req;
}

// ─────────────────────────────────────────────────────────────────────────────
// KimiSerializer — basic structure
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("KimiSerializer - model is serialized correctly", "[kimi][serializer]") {
    auto payload = Serializer::serialize(make_simple_request("moonshot-v1-8k"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"moonshot-v1-8k")"));
}

TEST_CASE("KimiSerializer - 32k model is serialized correctly", "[kimi][serializer]") {
    auto payload = Serializer::serialize(make_simple_request("moonshot-v1-32k"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"moonshot-v1-32k")"));
}

TEST_CASE("KimiSerializer - 128k model is serialized correctly", "[kimi][serializer]") {
    auto payload = Serializer::serialize(make_simple_request("moonshot-v1-128k"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"moonshot-v1-128k")"));
}

TEST_CASE("KimiSerializer - k2-5 model is serialized correctly", "[kimi][serializer]") {
    auto payload = Serializer::serialize(make_simple_request("kimi-k2-5"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"kimi-k2-5")"));
}

TEST_CASE("KimiSerializer - k2-5-thinking model is serialized correctly", "[kimi][serializer]") {
    auto payload = Serializer::serialize(make_simple_request("kimi-k2-5-thinking"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"kimi-k2-5-thinking")"));
}

TEST_CASE("KimiSerializer - stream true is included", "[kimi][serializer]") {
    auto req = make_simple_request();
    req.stream = true;
    REQUIRE_THAT(Serializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("stream":true)"));
}

TEST_CASE("KimiSerializer - stream false is included", "[kimi][serializer]") {
    auto req = make_simple_request();
    req.stream = false;
    REQUIRE_THAT(Serializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("stream":false)"));
}

TEST_CASE("KimiSerializer - stream_options included for streaming", "[kimi][serializer]") {
    // Kimi uses OpenAIProtocol with stream_usage=true to include stream_options
    OpenAIProtocol protocol(/*stream_usage=*/true);
    auto req = make_simple_request();
    req.stream = true;
    auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("stream_options")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("include_usage":true)"));
}

TEST_CASE("KimiProtocol - stream_options included for streaming by default", "[kimi][serializer]") {
    KimiProtocol protocol;
    auto req = make_simple_request();
    req.stream = true;
    auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("stream_options")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("include_usage":true)"));
}

TEST_CASE("KimiProtocol - effort enables Kimi thinking for kimi-for-coding",
          "[kimi][serializer][effort]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    req.effort = "medium";

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"medium")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"enabled"})"));
}

TEST_CASE("KimiProtocol - effort enables Kimi thinking for K2 models",
          "[kimi][serializer][effort]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-k2.6");
    req.effort = "low";

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"low")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"enabled"})"));
}

TEST_CASE("KimiProtocol - max effort maps to high", "[kimi][serializer][effort]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    req.effort = "max";

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"high")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"enabled"})"));
}

TEST_CASE("KimiProtocol - auto effort omits Kimi thinking fields",
          "[kimi][serializer][effort]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    req.effort = "auto";

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("thinking")"));
}

TEST_CASE("KimiProtocol - effort is omitted on legacy Moonshot models",
          "[kimi][serializer][effort]") {
    KimiProtocol protocol;
    auto req = make_simple_request("moonshot-v1-128k");
    req.effort = "high";

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("thinking")"));
}

TEST_CASE("KimiProtocol - off effort disables Kimi thinking",
          "[kimi][serializer][effort]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    req.effort = "off";

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"disabled"})"));
}

TEST_CASE("KimiSerializer - user message content appears in payload", "[kimi][serializer]") {
    auto payload = Serializer::serialize(make_simple_request("moonshot-v1-8k", "Tell me a joke"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Tell me a joke"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"user")"));
}

TEST_CASE("KimiSerializer - empty messages produce valid JSON", "[kimi][serializer]") {
    ChatRequest req;
    req.model = "moonshot-v1-8k";
    req.stream = true;
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("messages":[])"));
}

TEST_CASE("KimiProtocol uploads local videos as files and serializes ms references",
          "[kimi][video][upload]") {
    const auto dir = make_temp_dir("filo_kimi_video_upload");
    const auto video = dir / "recording.mp4";
    {
        std::ofstream out(video, std::ios::binary | std::ios::trunc);
        out << "fake-video";
    }

    httplib::Server server;
    std::atomic<int> uploads{0};
    std::string authorization;
    std::string content_type;
    server.Post("/v1/files", [&](const httplib::Request& req, httplib::Response& res) {
        ++uploads;
        authorization = req.get_header_value("Authorization");
        content_type = req.get_header_value("Content-Type");
        res.set_content(R"({"id":"file_kimi_video_123"})", "application/json");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    for (int i = 0; i < 50 && !server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(server.is_running());

    core::auth::AuthInfo auth;
    auth.headers["Authorization"] = "Bearer test-token";

    ChatRequest req;
    req.model = "kimi-k2.6";
    req.messages.push_back(Message{
        .role = "user",
        .content = describe_video_attachment(video.string()),
        .content_parts = {
            ContentPart::make_text("Summarize this recording"),
            ContentPart::make_video(video.string(), "video/mp4"),
        },
    });

    KimiProtocol protocol;
    const std::string base_url = std::format("http://127.0.0.1:{}/v1", port);
    REQUIRE_NOTHROW(protocol.prepare_media_uploads(req, base_url, auth));

    REQUIRE(uploads.load() == 1);
    CHECK(authorization == "Bearer test-token");
    CHECK_THAT(content_type, Catch::Matchers::ContainsSubstring("multipart/form-data"));
    REQUIRE(req.messages[0].content_parts[1].url == "ms://file_kimi_video_123");
    REQUIRE(req.messages[0].content_parts[1].media_id == "file_kimi_video_123");

    const auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("video_url":{"url":"ms://file_kimi_video_123","id":"file_kimi_video_123"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("data:video/"));

    ChatRequest cached_req;
    cached_req.model = "kimi-k2.6";
    cached_req.messages.push_back(Message{
        .role = "user",
        .content_parts = {
            ContentPart::make_video(video.string(), "video/mp4"),
        },
    });
    REQUIRE_NOTHROW(protocol.prepare_media_uploads(cached_req, base_url, auth));
    CHECK(uploads.load() == 1);
    CHECK(cached_req.messages[0].content_parts[0].url == "ms://file_kimi_video_123");

    server.stop();
}

TEST_CASE("HttpLLMProvider validates video file size before Kimi upload",
          "[kimi][video][validation]") {
    const auto dir = make_temp_dir("filo_kimi_video_validation");
    const auto video = dir / "too-large.mp4";
    std::ofstream(video, std::ios::binary | std::ios::trunc).close();
    fs::resize_file(video, kMaxLocalVideoBytes + 1);

    ChatRequest req;
    req.model = "kimi-k2.6";
    req.messages.push_back(Message{
        .role = "user",
        .content_parts = {
            ContentPart::make_video(video.string(), "video/mp4"),
        },
    });

    HttpLLMProvider provider(
        "https://api.moonshot.cn/v1",
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        "kimi-k2.6",
        std::make_unique<KimiProtocol>());

    const auto errors = provider.validate_request(req);
    REQUIRE_FALSE(errors.empty());
    CHECK_THAT(errors.front(), Catch::Matchers::ContainsSubstring("exceeds the 100 MB limit"));
}

TEST_CASE("HttpLLMProvider rejects large inline videos when provider has no upload path",
          "[kimi][video][validation]") {
    const std::string model_id = "test-inline-video-model-"
        + core::auth::KimiOAuthFlow::generateDeviceId();

    ModelInfo model;
    model.canonical_id = model_id;
    model.display_name = "Test Inline Video Model";
    model.provider = "test";
    model.context_window = 128000;
    model.max_output_tokens = 4096;
    model.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput)
        | static_cast<uint32_t>(ModelCapability::TextOutput)
        | static_cast<uint32_t>(ModelCapability::Streaming)
        | static_cast<uint32_t>(ModelCapability::VideoInput);
    model.tier = ModelTier::Balanced;
    ModelRegistry::instance().register_model(std::move(model));

    const auto dir = make_temp_dir("filo_inline_video_validation");
    const auto video = dir / "too-large-for-inline.mp4";
    std::ofstream(video, std::ios::binary | std::ios::trunc).close();
    fs::resize_file(video, kMaxInlineVideoBytes + 1);

    ChatRequest req;
    req.model = model_id;
    req.messages.push_back(Message{
        .role = "user",
        .content_parts = {
            ContentPart::make_video(video.string(), "video/mp4"),
        },
    });

    HttpLLMProvider provider(
        "https://example.invalid/v1",
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        model_id,
        std::make_unique<OpenAIProtocol>());

    const auto errors = provider.validate_request(req);
    REQUIRE_FALSE(errors.empty());
    CHECK_THAT(errors.front(), Catch::Matchers::ContainsSubstring(
        "can only inline videos up to 20 MB"));
}

// ─────────────────────────────────────────────────────────────────────────────
// KimiSerializer — optional parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("KimiSerializer - temperature omitted when not set", "[kimi][serializer]") {
    auto req = make_simple_request();
    // req.temperature not set
    REQUIRE_THAT(Serializer::serialize(req), !Catch::Matchers::ContainsSubstring("temperature"));
}

TEST_CASE("KimiSerializer - temperature included when set", "[kimi][serializer]") {
    auto req = make_simple_request();
    req.temperature = 0.7f;
    REQUIRE_THAT(Serializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("temperature")"));
}

TEST_CASE("KimiSerializer - max_tokens omitted when not set", "[kimi][serializer]") {
    REQUIRE_THAT(Serializer::serialize(make_simple_request()), !Catch::Matchers::ContainsSubstring("max_tokens"));
}

TEST_CASE("KimiSerializer - max_tokens included when set", "[kimi][serializer]") {
    auto req = make_simple_request();
    req.max_tokens = 4096;
    REQUIRE_THAT(Serializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("max_tokens":4096)"));
}

TEST_CASE("KimiProtocol - max_tokens is sent as max_completion_tokens",
          "[kimi][serializer]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    req.max_tokens = 4096;

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("max_completion_tokens":4096)"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("max_tokens":4096)"));
}

// ─────────────────────────────────────────────────────────────────────────────
// KimiSerializer — multi-turn conversation roles
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("KimiSerializer - system message serialized with correct role", "[kimi][serializer]") {
    ChatRequest req;
    req.model = "moonshot-v1-8k";
    req.stream = true;
    req.messages = {
        Message{.role = "system", .content = "You are Kimi, made by Moonshot AI."},
        Message{.role = "user",   .content = "Hello"}
    };
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"system")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("You are Kimi, made by Moonshot AI."));
}

TEST_CASE("KimiSerializer - assistant message serialized with correct role", "[kimi][serializer]") {
    ChatRequest req;
    req.model = "moonshot-v1-8k";
    req.stream = true;
    req.messages = {
        Message{.role = "user",      .content = "Hi"},
        Message{.role = "assistant", .content = "Hello there!"},
        Message{.role = "user",      .content = "Bye"}
    };
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"assistant")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Hello there!"));
}

TEST_CASE("KimiSerializer - tool response message with tool_call_id", "[kimi][serializer]") {
    ChatRequest req;
    req.model = "moonshot-v1-8k";
    req.stream = true;
    Message tool_resp;
    tool_resp.role = "tool";
    tool_resp.content = R"({"exit_code": 0, "output": "hello"})";
    tool_resp.tool_call_id = "call_abc123";
    tool_resp.name = "run_terminal_command";
    req.messages.push_back(tool_resp);

    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_call_id":"call_abc123")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"run_terminal_command")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"tool")"));
}

TEST_CASE("KimiSerializer - assistant message with tool_calls uses null content", "[kimi][serializer]") {
    ChatRequest req;
    req.model = "moonshot-v1-8k";
    req.stream = true;
    Message assistant_msg;
    assistant_msg.role = "assistant";
    // No text content — assistant is making a tool call
    ToolCall tc;
    tc.id = "call_xyz";
    tc.type = "function";
    tc.function.name = "read_file";
    tc.function.arguments = R"({"path": "/etc/hosts"})";
    assistant_msg.tool_calls = {tc};
    req.messages.push_back(assistant_msg);

    KimiProtocol protocol;
    auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_calls")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("id":"call_xyz")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"read_file")"));
    // Empty content with tool_calls should emit null, not an empty string
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("content":"")"));
    // No reasoning_content when empty — sending "" causes Kimi 400 on non-thinking models
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_content"));
}

TEST_CASE("KimiSerializer - assistant tool_calls message with reasoning_content includes it", "[kimi][serializer]") {
    // For kimi-k2-5-thinking: the model returns reasoning_content alongside tool_calls.
    // It must be echoed back in the next request so Kimi can correlate the reasoning.
    ChatRequest req;
    req.model = "kimi-k2-5-thinking";
    req.stream = true;
    Message assistant_msg;
    assistant_msg.role = "assistant";
    assistant_msg.reasoning_content = "I should read the file first.";
    ToolCall tc;
    tc.id = "call_think";
    tc.type = "function";
    tc.function.name = "read_file";
    tc.function.arguments = R"({"path": "/etc/hosts"})";
    assistant_msg.tool_calls = {tc};
    req.messages.push_back(assistant_msg);

    KimiProtocol protocol;
    auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_content":"I should read the file first.")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// KimiSerializer — tool definitions
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("KimiSerializer - single tool definition serialized", "[kimi][serializer][tools]") {
    auto req = make_simple_request();
    core::tools::ToolParameter param{
        .name = "command",
        .type = "string",
        .description = "The shell command to execute",
        .required = true
    };
    core::tools::ToolDefinition def{
        .name = "run_terminal_command",
        .description = "Execute a shell command on the local machine",
        .parameters = {param}
    };
    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tools")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("run_terminal_command")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("command")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":["command"])"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"function")"));
}

TEST_CASE("KimiSerializer - optional parameters excluded from required array", "[kimi][serializer][tools]") {
    auto req = make_simple_request();
    core::tools::ToolDefinition def;
    def.name = "grep_search";
    def.description = "Search for a pattern";
    def.parameters = {
        {.name = "pattern",  .type = "string", .description = "Regex", .required = true},
        {.name = "path", .type = "string", .description = "Root", .required = false}
    };
    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":["pattern"])"));
    // Optional param is in properties but NOT in required
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("path"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("required":["pattern","path"])"));
}

TEST_CASE("KimiSerializer - multiple tools serialized without trailing comma", "[kimi][serializer][tools]") {
    auto req = make_simple_request();
    for (const auto& name : {"read_file", "write_file"}) {
        core::tools::ToolDefinition def;
        def.name = name;
        def.description = "A file tool";
        def.parameters.push_back({.name = "path", .type = "string", .description = "Path", .required = true});
        Tool tool;
        tool.type = "function";
        tool.function = def;
        req.tools.push_back(tool);
    }

    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("read_file")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("write_file")"));
    // No trailing comma before closing bracket (rudimentary check)
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
}

TEST_CASE("KimiProtocol - tool input_schema is normalized for Kimi",
          "[kimi][serializer][tools]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    core::tools::ToolDefinition def;
    def.name = "complex_tool";
    def.description = "Complex schema";
    def.input_schema = R"({"type":"object","properties":{"mode":{"enum":["a","b"]}},"required":["mode"]})";

    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("mode":{"enum":["a","b"],"type":"string"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":["mode"])"));
}

TEST_CASE("KimiProtocol - tool input_schema normalization matches Kimi CLI fallbacks",
          "[kimi][serializer][tools]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    core::tools::ToolDefinition def;
    def.name = "schema_fallbacks";
    def.description = "Schema fallbacks";
    def.input_schema =
        R"({"type":"object","properties":{"payload":{},"values":{"items":{}},"config":{"additionalProperties":{}},"choice":{"anyOf":[{"enum":["a"]},{"const":1}]}}})";

    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("payload":{"type":"string"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("values":{"items":{"type":"string"},"type":"array"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("config":{"additionalProperties":{"type":"string"},"type":"object"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("choice":{"anyOf":[{"enum":["a"],"type":"string"},{"const":1,"type":"integer"}]})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(
        R"("choice":{"anyOf":[{"enum":["a"],"type":"string"},{"const":1,"type":"integer"}],"type")"));
}

TEST_CASE("KimiProtocol - root input_schema is treated as a container",
          "[kimi][serializer][tools]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    core::tools::ToolDefinition def;
    def.name = "untyped_root";
    def.description = "Untyped root";
    def.input_schema = R"({})";

    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("parameters":{})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("parameters":{"type":"string"})"));
}

TEST_CASE("KimiProtocol - parameter schema roots are normalized as properties",
          "[kimi][serializer][tools]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    core::tools::ToolDefinition def;
    def.name = "parameter_schema";
    def.description = "Parameter schema";
    def.parameters.push_back({
        .name = "mode",
        .type = "string",
        .description = "Mode",
        .required = true,
        .schema = R"({"enum":["smart","full"]})",
    });

    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("mode":{"enum":["smart","full"],"type":"string"})"));
}

TEST_CASE("KimiProtocol - tool input_schema local refs are preserved",
          "[kimi][serializer][tools]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    core::tools::ToolDefinition def;
    def.name = "choose_mode";
    def.description = "Choose a mode";
    def.input_schema =
        R"({"type":"object","properties":{"mode":{"$ref":"#/definitions/Mode"}},"definitions":{"Mode":{"enum":["fast","safe"]}}})";

    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("mode":{"$ref":"#/definitions/Mode"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("definitions":{"Mode":{"enum":["fast","safe"]}})"));
}

TEST_CASE("KimiProtocol - builtin tools use Kimi builtin_function format",
          "[kimi][serializer][tools]") {
    KimiProtocol protocol;
    auto req = make_simple_request("kimi-for-coding");
    core::tools::ToolDefinition def;
    def.name = "$web_search";

    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"builtin_function")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"$web_search")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("parameters")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// KimiSerializer — JSON escaping
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("KimiSerializer - newline in content is JSON-escaped", "[kimi][serializer][escaping]") {
    auto req = make_simple_request("moonshot-v1-8k", "line1\nline2");
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\n)"));
    // Raw newline must NOT appear inside a JSON string value
    // (the payload as a whole may have newlines for formatting but content values must not)
}

TEST_CASE("KimiSerializer - tab in content is JSON-escaped", "[kimi][serializer][escaping]") {
    auto req = make_simple_request("moonshot-v1-8k", "col1\tcol2");
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\t)"));
}

TEST_CASE("KimiSerializer - double-quote in content is JSON-escaped", "[kimi][serializer][escaping]") {
    auto req = make_simple_request("moonshot-v1-8k", R"(He said "hello")");
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\")"));
}

TEST_CASE("KimiSerializer - backslash in content is JSON-escaped", "[kimi][serializer][escaping]") {
    auto req = make_simple_request("moonshot-v1-8k", R"(C:\Users\foo)");
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\\)"));
}

TEST_CASE("KimiProtocol - user video content becomes video_url parts",
          "[kimi][serializer][video]") {
    const auto video = make_temp_video_file();

    ChatRequest req;
    req.model = "kimi-for-coding";
    req.stream = true;
    req.messages.push_back(Message{
        .role = "user",
        .content = describe_video_attachment(video.string()),
        .content_parts = {
            ContentPart::make_text("Analyze this screen recording."),
            ContentPart::make_video(video.string(), "video/mp4"),
        },
    });

    KimiProtocol protocol;
    const auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"video_url")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("video_url":{"url":"data:video/mp4;base64,)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"text","text":"Analyze this screen recording.")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// KimiSerializer — all known Kimi model names round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("KimiSerializer - all known Kimi model names are preserved verbatim", "[kimi][serializer]") {
    const std::vector<std::string> models = {
        // Moonshot v1 series
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
        // K2.x series
        "kimi-k2.6",
        "kimi-k2-5",
        "kimi-k2-5-thinking",
        // Official Kimi Code model ID
        "kimi-for-coding",
        // Legacy models (if any)
        "moonshot-v1-8k-vision",
        "moonshot-v1-32k-vision",
    };
    for (const auto& model : models) {
        INFO("Testing model: " << model);
        auto payload = Serializer::serialize(make_simple_request(model));
        REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":")" + model + "\""));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_kimi_sse_chunk — text content extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_kimi_sse_chunk - extracts text content from delta", "[kimi][parser]") {
    std::string json = R"({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content == "Hello");
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - handles empty delta with only role", "[kimi][parser]") {
    // First chunk from Kimi typically carries only role, no content
    std::string json = R"({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - finish reason stop produces no content", "[kimi][parser]") {
    std::string json = R"({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - finish reason tool_calls produces no content", "[kimi][parser]") {
    std::string json = R"({
        "id": "chatcmpl-abc",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - multi-word content returned intact", "[kimi][parser]") {
    std::string json = R"({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content == " world");
    REQUIRE(tools.empty());
}

TEST_CASE("KimiProtocol parse_event - extracts reasoning_content from delta", "[kimi][parser]") {
    // Kimi K2.5 models send reasoning_content for thinking tokens
    KimiProtocol protocol;
    std::string event = R"(data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"reasoning_content": "Let me think"}, "finish_reason": null}]})";
    auto result = protocol.parse_event(event);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].reasoning_content == "Let me think");
    REQUIRE(result.chunks[0].content.empty());
    REQUIRE(result.chunks[0].tools.empty());
}

TEST_CASE("KimiProtocol parse_event - extracts both content and reasoning_content", "[kimi][parser]") {
    // If both are present, both should be extracted (reasoning for history, content for display)
    KimiProtocol protocol;
    std::string event = R"(data: {"id": "chatcmpl-123", "choices": [{"index": 0, "delta": {"content": "Hello", "reasoning_content": "Thinking..."}, "finish_reason": null}]})";
    auto result = protocol.parse_event(event);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "Hello");
    REQUIRE(result.chunks[0].reasoning_content == "Thinking...");
}

TEST_CASE("KimiProtocol parse_event - accepts data prefix without a space", "[kimi][parser]") {
    KimiProtocol protocol;
    std::string event = R"(data:{"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"NoSpace"},"finish_reason":null}]})";
    auto result = protocol.parse_event(event);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "NoSpace");
}

TEST_CASE("KimiProtocol parse_event - accepts event envelope with CRLF", "[kimi][parser]") {
    KimiProtocol protocol;
    std::string event =
        "event: response.output_text.delta\r\n"
        "data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"content\":\"CRLF\"},\"finish_reason\":null}]}\r\n";
    auto result = protocol.parse_event(event);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "CRLF");
}

TEST_CASE("KimiProtocol parse_event - joins multiline data payloads", "[kimi][parser]") {
    KimiProtocol protocol;
    std::string event =
        "data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}\n"
        "data: ,\"index\":0,\"finish_reason\":null}]}\n";
    auto result = protocol.parse_event(event);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "Hello");
}

TEST_CASE("KimiProtocol parse_event - extracts top-level usage chunk", "[kimi][parser][usage]") {
    KimiProtocol protocol;
    std::string event = R"(data: {"id":"chatcmpl-usage","choices":[],"usage":{"prompt_tokens":42,"completion_tokens":10,"total_tokens":52}})";
    auto result = protocol.parse_event(event);
    REQUIRE(result.prompt_tokens == 42);
    REQUIRE(result.completion_tokens == 10);
    REQUIRE(result.chunks.empty());
}

TEST_CASE("KimiProtocol parse_event - extracts choice-level usage chunk", "[kimi][parser][usage]") {
    // Kimi may return usage inside choices[0].usage instead of top-level usage.
    KimiProtocol protocol;
    std::string event = R"(data: {"id":"chatcmpl-usage","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":8,"completion_tokens":11,"total_tokens":19}}]})";
    auto result = protocol.parse_event(event);
    REQUIRE(result.prompt_tokens == 8);
    REQUIRE(result.completion_tokens == 11);
    REQUIRE(result.chunks.empty());
}

TEST_CASE("KimiProtocol::build_url avoids duplicate slash for trailing-slash base URL",
          "[kimi][protocol][url]") {
    KimiProtocol protocol;
    const std::string url = protocol.build_url("https://api.kimi.com/coding/v1/", "kimi-for-coding");
    REQUIRE(url == "https://api.kimi.com/coding/v1/chat/completions");
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_kimi_sse_chunk — tool call extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_kimi_sse_chunk - first tool call chunk with id and name", "[kimi][parser][tools]") {
    std::string json = R"({
        "id": "chatcmpl-abc",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_xyz789",
                    "type": "function",
                    "function": {"name": "run_terminal_command", "arguments": ""}
                }]
            },
            "finish_reason": null
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].id == "call_xyz789");
    REQUIRE(tools[0].type == "function");
    REQUIRE(tools[0].function.name == "run_terminal_command");
    REQUIRE(tools[0].function.arguments.empty());
    REQUIRE(tools[0].index == 0);
}

TEST_CASE("parse_kimi_sse_chunk - argument streaming chunk (no id/name)", "[kimi][parser][tools]") {
    // Subsequent chunks carry only partial arguments for an ongoing tool call
    std::string json = R"({
        "id": "chatcmpl-abc",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{\"command\": \"ls -la\"}"}
                }]
            },
            "finish_reason": null
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].index == 0);
    REQUIRE(tools[0].function.arguments == R"({"command": "ls -la"})");
    // id and name are absent in streaming argument chunks — they stay empty
    REQUIRE(tools[0].id.empty());
    REQUIRE(tools[0].function.name.empty());
}

TEST_CASE("parse_kimi_sse_chunk - multiple tool calls in single chunk", "[kimi][parser][tools]") {
    std::string json = R"({
        "id": "chatcmpl-multi",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [
                    {"index": 0, "id": "call_aaa", "type": "function", "function": {"name": "read_file", "arguments": ""}},
                    {"index": 1, "id": "call_bbb", "type": "function", "function": {"name": "grep_search", "arguments": ""}}
                ]
            },
            "finish_reason": null
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.size() == 2);
    REQUIRE(tools[0].index == 0);
    REQUIRE(tools[0].id == "call_aaa");
    REQUIRE(tools[0].function.name == "read_file");
    REQUIRE(tools[1].index == 1);
    REQUIRE(tools[1].id == "call_bbb");
    REQUIRE(tools[1].function.name == "grep_search");
}

TEST_CASE("parse_kimi_sse_chunk - tool call index matches correctly", "[kimi][parser][tools]") {
    std::string json = R"({
        "id": "chatcmpl-idx",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{"index": 2, "id": "call_ccc", "type": "function",
                                "function": {"name": "write_file", "arguments": "{}"}}]
            },
            "finish_reason": null
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].index == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_kimi_sse_chunk — error handling / edge cases
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_kimi_sse_chunk - empty string returns empty results", "[kimi][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk("");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - malformed JSON returns empty results", "[kimi][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk("{not valid json!!!}");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - truncated JSON returns empty results", "[kimi][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id": "chatcmpl-123", "choices": [)");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - empty choices array returns empty results", "[kimi][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id": "x", "choices": []})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - missing choices key returns empty results", "[kimi][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id": "x", "object": "chat.completion.chunk"})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - choice without delta returns empty results", "[kimi][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id": "x", "choices": [{"index": 0, "finish_reason": "stop"}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - null content value returns empty string", "[kimi][parser][errors]") {
    // Kimi may send content: null in some chunk types
    std::string json = R"({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "delta": {"content": null}, "finish_reason": null}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    // null is not a string — parser should handle gracefully and return empty
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_kimi_sse_chunk - pure whitespace JSON returns empty results", "[kimi][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk("   ");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("KimiProtocol on_response - extracts request/token rate-limit headers", "[kimi][rate_limit]") {
    KimiProtocol protocol;
    cpr::Header headers;
    headers["x-ratelimit-limit-requests"] = "200";
    headers["x-ratelimit-remaining-requests"] = "123";
    headers["x-ratelimit-limit-tokens"] = "500000";
    headers["x-ratelimit-remaining-tokens"] = "420000";

    protocol.on_response(HttpResponse{200, "{}", headers});
    const auto info = protocol.last_rate_limit();
    REQUIRE(info.requests_limit == 200);
    REQUIRE(info.requests_remaining == 123);
    REQUIRE(info.tokens_limit == 500000);
    REQUIRE(info.tokens_remaining == 420000);
}

TEST_CASE("KimiProtocol on_response - extracts unified utilization and retry-after", "[kimi][rate_limit]") {
    KimiProtocol protocol;
    cpr::Header headers;
    headers["x-ratelimit-unified-5h-utilization"] = "0.82";
    headers["x-ratelimit-unified-7d-utilization"] = "0.34";
    headers["retry-after"] = "12";

    protocol.on_response(HttpResponse{429, "{}", headers});
    const auto info = protocol.last_rate_limit();
    auto find_window = [&](std::string_view label) -> float {
        for (const auto& w : info.usage_windows) {
            if (w.label == label) return w.utilization;
        }
        return 0.0f;
    };
    REQUIRE(find_window("5h") == Catch::Approx(0.82f).epsilon(0.001f));
    REQUIRE(find_window("7d") == Catch::Approx(0.34f).epsilon(0.001f));
    REQUIRE(info.retry_after == 12);
    REQUIRE(info.is_rate_limited);
}

TEST_CASE("KimiProtocol on_response - keeps zero-utilization unified windows", "[kimi][rate_limit]") {
    KimiProtocol protocol;
    cpr::Header headers;
    headers["x-ratelimit-unified-5h-utilization"] = "0";
    headers["x-ratelimit-unified-7d-utilization"] = "0.00";

    protocol.on_response(HttpResponse{200, "{}", headers});
    const auto info = protocol.last_rate_limit();

    auto find_window = [&](std::string_view label) -> std::optional<float> {
        for (const auto& w : info.usage_windows) {
            if (w.label == label) return w.utilization;
        }
        return std::nullopt;
    };

    const auto five_hour = find_window("5h");
    const auto seven_day = find_window("7d");
    REQUIRE(five_hour.has_value());
    REQUIRE(seven_day.has_value());
    REQUIRE(*five_hour == Catch::Approx(0.0f));
    REQUIRE(*seven_day == Catch::Approx(0.0f));
}

TEST_CASE("KimiProtocol on_response - parses case-insensitive headers and defaults missing remaining", "[kimi][rate_limit]") {
    KimiProtocol protocol;
    cpr::Header headers;
    headers["X-MSH-RATELIMIT-LIMIT-REQUESTS"] = "80";
    headers["X-MSH-RATELIMIT-LIMIT-TOKENS"] = "120000";
    // Remaining headers intentionally omitted by the server in this test case.

    protocol.on_response(HttpResponse{200, "{}", headers});
    const auto info = protocol.last_rate_limit();
    REQUIRE(info.requests_limit == 80);
    REQUIRE(info.requests_remaining == 80);
    REQUIRE(info.tokens_limit == 120000);
    REQUIRE(info.tokens_remaining == 120000);
}

TEST_CASE("KimiProtocol enrich_rate_limit - skips usage endpoint call when headers are already complete",
          "[kimi][rate_limit][enrich]") {
    KimiProtocol protocol;
    cpr::Header headers;
    headers["x-ratelimit-limit-requests"] = "100";
    headers["x-ratelimit-remaining-requests"] = "90";
    headers["x-ratelimit-limit-tokens"] = "300000";
    headers["x-ratelimit-remaining-tokens"] = "250000";
    headers["x-ratelimit-unified-5h-utilization"] = "0.25";
    headers["x-ratelimit-unified-7d-utilization"] = "0.40";

    protocol.on_response(HttpResponse{200, "{}", headers});
    const auto before = protocol.last_rate_limit();
    REQUIRE(before.tokens_limit == 300000);
    REQUIRE(before.tokens_remaining == 250000);
    REQUIRE(before.usage_windows.size() == 2);

    // With complete rate-limit headers, enrich_rate_limit should return early and
    // not mutate the current snapshot.
    protocol.enrich_rate_limit(
        "https://api.kimi.com/coding/v1",
        cpr::Header{{"Authorization", "Bearer test-token"}},
        HttpResponse{200, "{}", {}});

    const auto after = protocol.last_rate_limit();
    REQUIRE(after.tokens_limit == before.tokens_limit);
    REQUIRE(after.tokens_remaining == before.tokens_remaining);
    REQUIRE(after.usage_windows.size() == before.usage_windows.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// HttpLLMProvider with OpenAIProtocol (Kimi uses OpenAI-compatible API)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("KimiProvider - constructs with default model", "[kimi][provider]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("kimi-test-key-abc");
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.moonshot.cn/v1",
        creds,
        "moonshot-v1-8k",
        std::make_unique<KimiProtocol>()));
}

TEST_CASE("KimiProvider - constructs with explicit model", "[kimi][provider]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("kimi-test-key-abc");
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.moonshot.cn/v1",
        creds,
        "moonshot-v1-32k",
        std::make_unique<KimiProtocol>()));
}

TEST_CASE("KimiProvider - constructs with k2-5 model", "[kimi][provider]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("kimi-test-key-abc");
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.moonshot.cn/v1",
        creds,
        "kimi-k2-5",
        std::make_unique<KimiProtocol>()));
}

TEST_CASE("KimiProvider - constructs with empty API key", "[kimi][provider]") {
    // Empty key is allowed at construction; the error surfaces when a request is sent.
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("");
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.moonshot.cn/v1",
        creds,
        "moonshot-v1-8k",
        std::make_unique<KimiProtocol>()));
}

TEST_CASE("KimiProvider - satisfies LLMProvider interface", "[kimi][provider]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("kimi-test-key-abc");
    HttpLLMProvider provider(
        "https://api.moonshot.cn/v1",
        creds,
        "moonshot-v1-8k",
        std::make_unique<KimiProtocol>());
    // Verify static polymorphism: HttpLLMProvider IS-A LLMProvider
    LLMProvider& base = provider;
    (void)base;
    SUCCEED("HttpLLMProvider is substitutable for LLMProvider");
}

TEST_CASE("KimiProvider - can be stored as shared_ptr<LLMProvider>", "[kimi][provider]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("kimi-test-key-abc");
    std::shared_ptr<LLMProvider> prov =
        std::make_shared<HttpLLMProvider>(
            "https://api.moonshot.cn/v1",
            creds,
            "moonshot-v1-8k",
            std::make_unique<KimiProtocol>());
    REQUIRE(prov != nullptr);
}

TEST_CASE("KimiProvider - reports correct capabilities", "[kimi][provider]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("kimi-test-key-abc");
    HttpLLMProvider provider(
        "https://api.moonshot.cn/v1",
        creds,
        "moonshot-v1-8k",
        std::make_unique<KimiProtocol>());
    auto caps = provider.capabilities();
    REQUIRE(caps.supports_tool_calls == true);
    REQUIRE(caps.is_local == false);
}

TEST_CASE("KimiOAuthFlow - device id uses UUID format", "[kimi][oauth]") {
    const std::string id = core::auth::KimiOAuthFlow::generateDeviceId();
    REQUIRE(id.size() == 36);
    REQUIRE(id[8] == '-');
    REQUIRE(id[13] == '-');
    REQUIRE(id[14] == '4');
    REQUIRE(id[18] == '-');
    REQUIRE(id[23] == '-');
    REQUIRE((id[19] == '8' || id[19] == '9' || id[19] == 'a' || id[19] == 'b'));
}

TEST_CASE("KimiOAuthFlow - common headers match managed Kimi Code platform",
          "[kimi][oauth][headers]") {
    const fs::path sandbox = make_temp_dir("filo_kimi_headers");
    const ScopedEnvVar xdg("XDG_CONFIG_HOME", sandbox.string());

    const auto headers = core::auth::KimiOAuthFlow::getCommonHeaders();

    REQUIRE(headers.at("X-Msh-Platform") == "kimi_code_cli");
    REQUIRE(headers.at("X-Msh-Version") == "kimi-code-cli/1.46.0");
    REQUIRE_FALSE(headers.at("X-Msh-Device-Name").empty());
    REQUIRE_FALSE(headers.at("X-Msh-Device-Model").empty());
    REQUIRE_FALSE(headers.at("X-Msh-Os-Version").empty());
    REQUIRE(headers.contains("X-Msh-Device-Id"));
    REQUIRE(headers.at("X-Msh-Device-Id").size() == 36);
    REQUIRE(fs::exists(sandbox / "filo" / "kimi_device_id"));

    fs::remove_all(sandbox);
}

TEST_CASE("KimiOAuthFlow - common headers preserve legacy 32-hex device id",
          "[kimi][oauth][headers]") {
    const fs::path sandbox = make_temp_dir("filo_kimi_legacy_headers");
    const ScopedEnvVar xdg("XDG_CONFIG_HOME", sandbox.string());
    const fs::path config_dir = sandbox / "filo";
    fs::create_directories(config_dir);

    const std::string legacy_id = "0123456789abcdef0123456789abcdef";
    {
        std::ofstream file(config_dir / "kimi_device_id");
        file << legacy_id;
    }

    const auto headers = core::auth::KimiOAuthFlow::getCommonHeaders();

    REQUIRE(headers.at("X-Msh-Device-Id") == legacy_id);

    fs::remove_all(sandbox);
}

TEST_CASE("KimiProtocol - headers identify Kimi CLI and preserve OAuth device id",
          "[kimi][oauth][headers]") {
    KimiProtocol protocol;
    core::auth::AuthInfo auth;
    auth.headers["Authorization"] = "Bearer token";
    auth.properties["device_id"] = "device-from-jwt";

    const auto headers = protocol.build_headers(auth);

    REQUIRE(headers.at("User-Agent") == "kimi-code-cli/1.46.0");
    REQUIRE(headers.at("X-Msh-Platform") == "kimi_code_cli");
    REQUIRE(headers.at("X-Msh-Device-Id") == "device-from-jwt");
    REQUIRE(headers.at("Authorization") == "Bearer token");
}

TEST_CASE("ProviderFactory - kimi oauth disables synthetic cost estimation", "[kimi][factory][oauth]") {
    core::config::ProviderConfig cfg;
    cfg.model = "kimi-for-coding";
    cfg.auth_type = "oauth_kimi";

    auto provider = core::llm::ProviderFactory::create_provider("kimi", cfg);
    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->should_estimate_cost());
}

TEST_CASE("ProviderFactory - kimi api-key mode keeps cost estimation enabled", "[kimi][factory][oauth]") {
    core::config::ProviderConfig cfg;
    cfg.model = "kimi-for-coding";
    cfg.auth_type = "api_key";
    cfg.api_key = "test-key";

    auto provider = core::llm::ProviderFactory::create_provider("kimi", cfg);
    REQUIRE(provider != nullptr);
    REQUIRE(provider->should_estimate_cost());
}

// ─────────────────────────────────────────────────────────────────────────────
// KimiSerializer — payload is valid JSON structure (structural checks)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("KimiSerializer - payload starts with { and ends with }", "[kimi][serializer][structure]") {
    auto payload = Serializer::serialize(make_simple_request());
    REQUIRE(!payload.empty());
    REQUIRE(payload.front() == '{');
    REQUIRE(payload.back() == '}');
}

TEST_CASE("KimiSerializer - messages array is always present", "[kimi][serializer][structure]") {
    ChatRequest req;
    req.model = "moonshot-v1-8k";
    req.stream = true;
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("messages":[])"));
}

TEST_CASE("KimiSerializer - tools array absent when no tools provided", "[kimi][serializer][structure]") {
    auto payload = Serializer::serialize(make_simple_request());
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("tools")"));
}

TEST_CASE("KimiSerializer - no trailing commas in JSON arrays", "[kimi][serializer][structure]") {
    // A trailing comma before ] is invalid JSON — verify it's not present
    auto payload = Serializer::serialize(make_simple_request());
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",}"));
}

TEST_CASE("KimiSerializer - no trailing commas with multiple messages", "[kimi][serializer][structure]") {
    ChatRequest req;
    req.model = "moonshot-v1-8k";
    req.stream = true;
    req.messages = {
        Message{.role = "user", .content = "first"},
        Message{.role = "assistant", .content = "second"},
        Message{.role = "user", .content = "third"}
    };
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",}"));
}
