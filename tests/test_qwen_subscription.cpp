#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>
#include <simdjson.h>

#include "TestSessionContext.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/auth/FileTokenStore.hpp"
#include "core/auth/OAuthErrors.hpp"
#include "core/auth/QwenSubscriptionAuthenticationStrategy.hpp"
#include "core/auth/QwenSubscriptionOAuthFlow.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/QwenSubscription.hpp"
#include "core/llm/ReasoningEffortResolution.hpp"
#include "core/llm/catalog/QwenSubscriptionModelCatalogProvider.hpp"
#include "core/llm/providers/QwenModelCatalogSelector.hpp"
#include "core/llm/protocols/QwenSubscriptionProtocol.hpp"
#include "core/tools/WebBackendAdapters.hpp"

#include <algorithm>
#include <atomic>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

using namespace core::llm;
using Catch::Matchers::ContainsSubstring;
namespace subscription = core::llm::qwen_subscription;

namespace {

constexpr std::int64_t kNow = 1'800'000'000;

class ScopedServer {
public:
    ScopedServer() = default;
    ~ScopedServer() {
        server.stop();
        if (thread_.joinable()) thread_.join();
    }
    [[nodiscard]] int start() {
        port_ = server.bind_to_any_port("127.0.0.1");
        if (port_ > 0) thread_ = std::thread([this] { server.listen_after_bind(); });
        return port_;
    }
    [[nodiscard]] std::string origin() const { return std::format("http://127.0.0.1:{}", port_); }
    httplib::Server server;

private:
    int port_ = 0;
    std::thread thread_;
};

class TempDir {
public:
    TempDir() : path_(std::filesystem::temp_directory_path()
                      / std::format("filo-qwen-sub-{}", std::chrono::steady_clock::now()
                                                            .time_since_epoch().count())) {
        std::filesystem::create_directories(path_);
    }
    ~TempDir() { std::error_code ec; std::filesystem::remove_all(path_, ec); }
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

/// Scripted answers for the sign-in prompts.
class ScriptedUI final : public core::auth::ui::AuthUI {
public:
    explicit ScriptedUI(std::deque<std::optional<std::string>> answers) : answers_(std::move(answers)) {}
    void show_header(const std::string&) override {}
    void show_instructions(const std::string& text) override { shown += text + '\n'; }
    void show_url(const std::string& url, const std::string&) override { urls.push_back(url); }
    std::string prompt_secret(const std::string&) override { return {}; }
    std::optional<std::string> prompt_text(const std::string&) override {
        ++prompts;
        if (answers_.empty()) return std::nullopt;
        auto answer = answers_.front();
        answers_.pop_front();
        return answer;
    }
    void show_success(const std::string&) override {}
    void show_error(const std::string& message) override { errors.push_back(message); }

    std::string shown;
    std::vector<std::string> urls;
    std::vector<std::string> errors;
    int prompts = 0;

private:
    std::deque<std::optional<std::string>> answers_;
};

ChatRequest request_for(std::string model, std::string effort = {}) {
    ChatRequest request;
    request.model = std::move(model);
    request.stream = true;
    request.effort = std::move(effort);
    request.messages.push_back(Message{.role = "user", .content = "Hello"});
    return request;
}

simdjson::dom::element parse(simdjson::dom::parser& parser, const std::string& json) {
    simdjson::dom::element element;
    REQUIRE(parser.parse(json).get(element) == simdjson::SUCCESS);
    return element;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Fixed contract
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Qwen Subscription links carry the CLI attribution Qwen Code uses",
          "[qwen][subscription]") {
    CHECK(subscription::authorization_url("AB C")
          == "https://chat.qwen.ai/authorize?user_code=AB%20C&client=cli"
             "&entry=qwen_code&paywallScene=cli");
    CHECK(subscription::plans_url()
          == "https://chat.qwen.ai/?memberPlan=qwen_code&entry=qwen_code&paywallScene=cli");
}

TEST_CASE("Only Plus and Pro memberships include the gateway", "[qwen][subscription]") {
    CHECK(subscription::membership_qualifies("plus"));
    CHECK(subscription::membership_qualifies(" PRO "));
    CHECK_FALSE(subscription::membership_qualifies("lite"));
    CHECK_FALSE(subscription::membership_qualifies("free"));
    CHECK_FALSE(subscription::membership_qualifies(""));
    CHECK(subscription::display_plan("pLUS") == "Plus");
    CHECK(subscription::display_plan("free") == "Free");
}

TEST_CASE("Qwen Subscription thinking budgets follow Qwen Code", "[qwen][subscription][thinking]") {
    CHECK(subscription::thinking_budget("") == 16384);
    CHECK(subscription::thinking_budget("minimal") == 4096);
    CHECK(subscription::thinking_budget("low") == 4096);
    CHECK(subscription::thinking_budget("medium") == 16384);
    CHECK(subscription::thinking_budget("high") == 32768);
    CHECK(subscription::thinking_budget("xhigh") == 32768);
    CHECK(subscription::thinking_budget("max") == 32768);
    CHECK_FALSE(subscription::thinking_budget("none"));
    CHECK_FALSE(subscription::thinking_budget("off"));
}

TEST_CASE("Only HTTP 402 insufficient_quota reads as exhausted subscription credits",
          "[qwen][subscription][errors]") {
    const auto quota = [](std::string member, std::string reset) {
        return std::format(R"({{"error":{{"code":"insufficient_quota","message":"raw",)"
                           R"("request_id":"req-1","quota":{{"member_type":"{}","reset_at":{}}}}}}})",
                           member, reset);
    };
    SECTION("Plus is told to wait or upgrade, with days and hours rounded up") {
        const auto message = subscription::gateway_error_message(
            402, quota("plus", std::to_string(kNow + 86400 + 3601)), kNow);
        REQUIRE(message);
        CHECK(*message == "You've used up your weekly Qwen Subscription credits. Your credits "
                          "are expected to reset in 1 day and 2 hours. You can wait or run "
                          "/upgrade to get more credits. (request_id: req-1)");
    }
    SECTION("Pro waits for the reset; under a day shows hours only") {
        const auto message = subscription::gateway_error_message(
            402, quota("pro", std::to_string(kNow + 1)), kNow);
        REQUIRE(message);
        CHECK_THAT(*message, ContainsSubstring("reset in 1 hour. Please try again after the reset."));
        CHECK_THAT(*message, !ContainsSubstring("/upgrade"));
    }
    SECTION("An unknown tier and an elapsed reset make no upgrade or time claim") {
        const auto message = subscription::gateway_error_message(
            402, quota("team", std::to_string(kNow - 5)), kNow);
        REQUIRE(message);
        CHECK(*message == "You've used up your Qwen Subscription credits. Please try again "
                          "after the reset. (request_id: req-1)");
    }
    SECTION("Missing quota context still produces quota guidance") {
        const auto message = subscription::gateway_error_message(
            402, R"({"error":{"code":"insufficient_quota"}})", kNow);
        REQUIRE(message);
        CHECK_THAT(*message, ContainsSubstring("used up your Qwen Subscription credits"));
    }
    SECTION("Other errors keep the public message; wording alone is not quota") {
        CHECK(subscription::gateway_error_message(
                  429, R"({"error":{"code":"insufficient_quota","message":"Slow down"}})", kNow)
              == "Slow down");
        CHECK(subscription::gateway_error_message(403, R"({"error_description":"Region blocked"})", kNow)
              == "Region blocked");
        CHECK_FALSE(subscription::gateway_error_message(500, "<html>", kNow));
    }
    SECTION("An unsafe request id is not echoed") {
        const auto message = subscription::gateway_error_message(
            402, R"({"error":{"code":"insufficient_quota","request_id":"a b"}})", kNow);
        REQUIRE(message);
        CHECK_THAT(*message, !ContainsSubstring("request_id"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wire protocol
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Qwen Subscription sends the thinking switch with a budget and no effort tier",
          "[qwen][subscription][thinking][serializer]") {
    simdjson::dom::parser parser;
    SECTION("An unset effort is medium") {
        const auto body = parse(parser, protocols::QwenSubscriptionProtocol{}.serialize(
            request_for("qwen4-plus")));
        CHECK(body["enable_thinking"].get_bool().value());
        CHECK(body["thinking_budget"].get_int64().value() == 16384);
        CHECK(body["stream_options"]["include_usage"].get_bool().value());
        CHECK(body["reasoning_effort"].error() == simdjson::NO_SUCH_FIELD);
        CHECK(body["reasoning"].error() == simdjson::NO_SUCH_FIELD);
    }
    SECTION("The configured default applies when the turn leaves effort unset") {
        const auto body = parse(parser, protocols::QwenSubscriptionProtocol{"low"}.serialize(
            request_for("qwen4-plus")));
        CHECK(body["thinking_budget"].get_int64().value() == 4096);
    }
    SECTION("Off disables thinking without a budget") {
        const auto body = parse(parser, protocols::QwenSubscriptionProtocol{"high"}.serialize(
            request_for("qwen4-plus", "none")));
        CHECK_FALSE(body["enable_thinking"].get_bool().value());
        CHECK(body["thinking_budget"].error() == simdjson::NO_SUCH_FIELD);
    }
    SECTION("xhigh lands on the high budget") {
        const auto body = parse(parser, protocols::QwenSubscriptionProtocol{}.serialize(
            request_for("qwen3.8-max", "xhigh")));
        CHECK(body["thinking_budget"].get_int64().value() == 32768);
    }
}

TEST_CASE("Qwen Subscription offers off, low, medium and high", "[qwen][subscription][thinking]") {
    const auto caps = protocols::QwenSubscriptionProtocol{}.reasoning_capabilities("qwen4-plus");
    CHECK(resolve_reasoning_effort("none", caps).effective == "off");
    CHECK(resolve_reasoning_effort("medium", caps).effective == "medium");
    CHECK(resolve_reasoning_effort("xhigh", caps).effective == "high");
    CHECK(resolve_reasoning_effort("max", caps).effective == "high");
}

TEST_CASE("Qwen Subscription replays Qwen reasoning and keeps empty tool-call content",
          "[qwen][subscription][serializer][tools]") {
    auto request = request_for("qwen4-plus");
    Message assistant{.role = "assistant", .reasoning_content = "Read the file first."};
    assistant.tool_calls.push_back(ToolCall{
        .id = "call_1", .function = {.name = "read", .arguments = R"({"path":"a"})"}});
    request.messages.push_back(std::move(assistant));
    request.messages.push_back(Message{.role = "tool", .content = "contents", .tool_call_id = "call_1"});

    simdjson::dom::parser parser;
    const auto body = parse(parser, protocols::QwenSubscriptionProtocol{}.serialize(request));
    const auto replayed = body["messages"].at(1);
    CHECK(replayed["content"].get_string().value() == "");
    CHECK(replayed["reasoning_content"].get_string().value() == "Read the file first.");
}

TEST_CASE("Qwen Subscription errors name the fix the user needs",
          "[qwen][subscription][errors]") {
    const protocols::QwenSubscriptionProtocol protocol;
    const cpr::Header headers;
    const std::string quota = R"({"error":{"code":"insufficient_quota","quota":{"member_type":"lite"}}})";
    CHECK_THAT(protocol.format_error_message({402, quota, headers}),
               ContainsSubstring("run /upgrade"));
    CHECK(protocol.format_error_message({401, "{}", headers})
          == std::string(subscription::kSignInRequiredMessage));
    CHECK(protocol.format_error_message({400, R"({"error":{"message":"bad model"}})", headers})
          == "Qwen Subscription request failed (HTTP 400): bad model");
    CHECK(protocol.format_error_message({503, "<html>oops</html>", headers})
          == "Qwen Subscription request failed (HTTP 503).");
    CHECK_FALSE(protocol.is_retryable({402, quota, headers}));
    CHECK(protocol.is_retryable({503, "", headers}));
}

// ─────────────────────────────────────────────────────────────────────────────
// Model catalog
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("The gateway catalog keeps its order and its declared metadata",
          "[qwen][subscription][catalog]") {
    const QwenSubscriptionModelCatalogProvider catalog("qwen-subscription");
    const auto result = catalog.parse_models_response(R"({"data":[
        {"id":"qwen4-plus","input_modalities":["text","image"],"context_window_tokens":2000000},
        {"id":"qwen3.8-max"},
        {"id":"qwen3.8-flash","input_modalities":["text"]},
        {"id":"qwen4-plus"},
        {"id":"bad,id"},
        {"id":"qwen3.7-plus","input_modalities":["text","hologram"],"context_window_tokens":-1}
    ]})");
    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 4);
    CHECK(result.models[0].canonical_id == "qwen4-plus");
    CHECK(result.models[1].canonical_id == "qwen3.8-max");
    CHECK(result.models[2].canonical_id == "qwen3.8-flash");
    CHECK(result.models[3].canonical_id == "qwen3.7-plus");

    // Declared fields win; tools and reasoning are always on.
    CHECK(result.models[0].context_window == 2'000'000);
    CHECK(result.models[0].supports(ModelCapability::Vision));
    CHECK_FALSE(result.models[0].supports(ModelCapability::VideoInput));
    CHECK(result.models[0].supports(ModelCapability::FunctionCalling));
    CHECK(result.models[0].supports(ModelCapability::Reasoning));
    // Absent fields fall back to the name rules.
    CHECK(result.models[1].context_window == 1'000'000);
    CHECK(result.models[1].supports(ModelCapability::Vision));
    // An explicit text-only list overrides a vision-capable name.
    CHECK_FALSE(result.models[2].supports(ModelCapability::Vision));
    // Invalid fields are ignored independently.
    CHECK(result.models[3].supports(ModelCapability::Vision));
    CHECK(result.models[3].context_window == 1'000'000);

    CHECK_FALSE(catalog.parse_models_response(R"({"data":[]})").ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider wiring
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("The qwen-subscription provider only ever talks to the fixed gateway",
          "[qwen][subscription][factory]") {
    core::config::ProviderConfig config;
    config.base_url = "https://attacker.example/v1";
    config.model = "qwen4-plus";
    const auto provider = ProviderFactory::create_provider("qwen-subscription", config);
    REQUIRE(provider);
    const auto metadata = provider->metadata();
    REQUIRE(metadata);
    CHECK(metadata->base_url == subscription::kBaseUrl);
    CHECK(metadata->api_type == core::config::ApiType::OpenAI);
    REQUIRE(metadata->credential_source);
    CHECK(metadata->credential_source->billing_kind() == core::auth::BillingKind::Subscription);
}

// ─────────────────────────────────────────────────────────────────────────────
// OAuth device flow
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("filo --auth offers the Qwen account sign-in with its own session store",
          "[qwen][subscription][oauth]") {
    TempDir config;
    const auto manager = core::auth::AuthenticationManager::create_with_defaults(config.path().string());
    const auto providers = manager.available_login_providers();
    CHECK(std::ranges::find(providers, std::string_view{"qwen-subscription"}) != providers.end());
    const auto described = manager.describe_provider("qwen-account");
    REQUIRE(described);
    CHECK(described->login_provider == "qwen-subscription");
    CHECK(described->display_name == "Qwen Subscription");

    core::config::ProviderConfig provider;
    provider.auth_type = "oauth_qwen_subscription";
    const auto credentials = manager.create_credential_source("qwen-subscription", provider);
    REQUIRE(credentials);
    // No sign-in yet: requests must not open a browser behind the user's back.
    CHECK(credentials->availability() == core::auth::CredentialAvailability::Missing);
    CHECK_THROWS_AS(credentials->get_auth(), core::auth::ReauthenticationRequired);
}

TEST_CASE("Qwen Subscription device sign-in polls until the account approves",
          "[qwen][subscription][oauth]") {
    ScopedServer auth;
    std::mutex mutex;
    std::vector<httplib::Params> token_requests;
    httplib::Params device_request;
    std::vector<std::string> user_agents;
    std::atomic<int> polls = 0;
    auth.server.Post("/api/oauth/device/code", [&](const httplib::Request& request, httplib::Response& response) {
        std::lock_guard lock(mutex);
        device_request = request.params;
        user_agents.push_back(request.get_header_value("User-Agent"));
        response.set_content(R"({"device_code":"dev-1","user_code":"QWEN-42",)"
                             R"("verification_uri":"https://chat.qwen.ai/device",)"
                             R"("verification_uri_complete":"https://chat.qwen.ai/device?c=1",)"
                             R"("expires_in":600,"interval":1})", "application/json");
    });
    auth.server.Post("/api/oauth/token", [&](const httplib::Request& request, httplib::Response& response) {
        {
            std::lock_guard lock(mutex);
            token_requests.push_back(request.params);
            user_agents.push_back(request.get_header_value("User-Agent"));
        }
        switch (++polls) {
        case 1:
            response.status = 400;
            response.set_content(R"({"error":"authorization_pending"})", "application/json");
            return;
        case 2:
            response.status = 400;
            response.set_content(R"({"error":"slow_down"})", "application/json");
            return;
        case 3:
            response.status = 503;
            response.set_header("Retry-After", "7");
            return;
        default:
            response.set_content(R"({"access_token":"at-1","refresh_token":"rt-1",)"
                                 R"("token_type":"Bearer","expires_in":3600,"scope":"model:invoke profile:read"})",
                                 "application/json");
        }
    });
    if (auth.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");

    std::vector<std::chrono::milliseconds> waits;
    std::vector<std::string> opened;
    std::ostringstream output;
    core::auth::QwenSubscriptionOAuthFlow flow({
        .auth_origin = auth.origin(),
        .output = &output,
        .open_browser = [&](std::string_view url) { opened.emplace_back(url); },
        .sleep = [&](std::chrono::milliseconds delay) { waits.push_back(delay); },
    });
    const auto token = flow.login();

    CHECK(token.access_token == "at-1");
    CHECK(token.refresh_token == "rt-1");
    CHECK(token.client_id == "qwen-code-cli");
    REQUIRE(opened.size() == 1);
    CHECK(opened.front() == subscription::authorization_url("QWEN-42"));
    CHECK_THAT(output.str(), ContainsSubstring("QWEN-42"));

    std::lock_guard lock(mutex);
    CHECK(device_request.find("client_id")->second == "qwen-code-cli");
    CHECK(device_request.find("scope")->second == "model:invoke profile:read");
    CHECK(device_request.count("code_challenge") == 0);
    // auth.qwen.ai's firewall rejects requests without a client User-Agent.
    for (const auto& agent : user_agents) CHECK(agent.starts_with("filo/"));
    REQUIRE(token_requests.size() == 4);
    CHECK(token_requests.front().find("grant_type")->second
          == "urn:ietf:params:oauth:grant-type:device_code");
    CHECK(token_requests.front().find("device_code")->second == "dev-1");
    // 1s interval, unchanged on pending, +5s on slow_down, then Retry-After wins.
    REQUIRE(waits.size() == 4);
    CHECK(waits[0] == std::chrono::seconds(1));
    CHECK(waits[1] == std::chrono::seconds(1));
    CHECK(waits[2] == std::chrono::seconds(6));
    CHECK(waits[3] == std::chrono::seconds(12));
}

TEST_CASE("Qwen Subscription refresh, membership and revocation use the account session",
          "[qwen][subscription][oauth]") {
    ScopedServer auth;
    std::mutex mutex;
    std::vector<httplib::Params> revocations;
    std::string userinfo_authorization;
    auth.server.Post("/api/oauth/token", [&](const httplib::Request& request, httplib::Response& response) {
        if (request.get_param_value("refresh_token") == "dead") {
            response.status = 400;
            response.set_content(R"({"error":"invalid_grant"})", "application/json");
            return;
        }
        if (request.get_param_value("refresh_token") == "busy") {
            response.status = 503;
            return;
        }
        response.set_content(R"({"access_token":"at-2","token_type":"Bearer","expires_in":3600,"scope":""})",
                             "application/json");
    });
    auth.server.Get("/api/oauth/userinfo", [&](const httplib::Request& request, httplib::Response& response) {
        {
            std::lock_guard lock(mutex);
            userinfo_authorization = request.get_header_value("Authorization");
        }
        if (userinfo_authorization == "Bearer expired") {
            response.status = 401;
            response.set_content(R"({"error":"invalid_token"})", "application/json");
            return;
        }
        response.set_content(R"({"sub":"u-1","name":"Ada","email":"ada@example.com","member_type":"Plus"})",
                             "application/json");
    });
    auth.server.Post("/api/oauth/revoke", [&](const httplib::Request& request, httplib::Response&) {
        std::lock_guard lock(mutex);
        revocations.push_back(request.params);
    });
    if (auth.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");

    core::auth::QwenSubscriptionOAuthFlow flow({.auth_origin = auth.origin()});
    const auto refreshed = flow.refresh("rt-1");
    CHECK(refreshed.access_token == "at-2");
    CHECK(refreshed.refresh_token == "rt-1");
    // Only invalid_grant ends the session; an outage stays recoverable.
    CHECK_THROWS_AS(flow.refresh("dead"), core::auth::OAuthRefreshRejected);
    try {
        (void)flow.refresh("busy");
        FAIL("refresh should fail");
    } catch (const core::auth::OAuthRefreshRejected&) {
        FAIL("a 503 must not end the session");
    } catch (const std::runtime_error& error) {
        CHECK_THAT(error.what(), ContainsSubstring("HTTP 503"));
    }

    const auto info = flow.user_info("at-2");
    CHECK(info.email == "ada@example.com");
    CHECK(info.member_type == "Plus");
    CHECK_THROWS_WITH(flow.user_info("expired"), std::string(subscription::kSignInRequiredMessage));

    flow.revoke(core::auth::OAuthToken{.access_token = "at-2", .refresh_token = "rt-1"});
    std::lock_guard lock(mutex);
    CHECK(userinfo_authorization == "Bearer expired");
    REQUIRE(revocations.size() == 1);
    CHECK(revocations.front().find("token")->second == "rt-1");
    CHECK(revocations.front().find("token_type_hint")->second == "refresh_token");
    CHECK(revocations.front().find("client_id")->second == "qwen-code-cli");
}

// ─────────────────────────────────────────────────────────────────────────────
// Sign-in strategy
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct SignInServer {
    ScopedServer server;
    std::atomic<int> userinfo_calls = 0;
    std::vector<std::string> member_types;

    explicit SignInServer(std::vector<std::string> plans) : member_types(std::move(plans)) {
        server.server.Post("/api/oauth/device/code", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(R"({"device_code":"d","user_code":"U-1","verification_uri":"https://x/",)"
                                 R"("verification_uri_complete":"https://x/?c","expires_in":60,"interval":1})",
                                 "application/json");
        });
        server.server.Post("/api/oauth/token", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(R"({"access_token":"at","refresh_token":"rt","token_type":"Bearer",)"
                                 R"("expires_in":3600,"scope":"model:invoke profile:read"})", "application/json");
        });
        server.server.Get("/api/oauth/userinfo", [this](const httplib::Request&, httplib::Response& response) {
            const auto index = static_cast<std::size_t>(userinfo_calls++);
            const auto& plan = member_types[std::min(index, member_types.size() - 1)];
            response.set_content(std::format(R"({{"sub":"u","name":"Ada","email":"ada@example.com","member_type":"{}"}})", plan),
                                 "application/json");
        });
    }
};

} // namespace

TEST_CASE("Qwen Subscription sign-in confirms the plan and becomes the manual selection",
          "[qwen][subscription][oauth][login]") {
    SignInServer gateway({"free", "Pro"});
    if (gateway.server.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    TempDir config;
    // First check finds Free: the user opens the plans page, then checks again.
    const auto ui = std::make_shared<ScriptedUI>(std::deque<std::optional<std::string>>{"o", ""});
    std::vector<std::string> opened;
    const auto strategy = core::auth::make_qwen_subscription_authentication_strategy({
        .auth_origin = gateway.server.origin(),
        .ui = ui,
        .open_browser = [&](std::string_view url) { opened.emplace_back(url); },
        .sleep = [](std::chrono::milliseconds) {},
    });

    const auto hints = strategy->login(config.path().string());

    CHECK(gateway.userinfo_calls == 2);
    REQUIRE(opened.size() == 2);
    CHECK(opened[0] == subscription::authorization_url("U-1"));
    CHECK(opened[1] == subscription::plans_url());
    REQUIRE_FALSE(ui->errors.empty());
    CHECK_THAT(ui->errors.front(), ContainsSubstring("current plan: Free"));
    CHECK_THAT(hints.front(), ContainsSubstring("ada@example.com with Qwen Code Pro"));
    CHECK_THAT(hints.at(1), ContainsSubstring("first model your subscription lists"));

    core::auth::FileTokenStore store(config.path().string());
    const auto saved = store.load("qwen-subscription");
    REQUIRE(saved);
    CHECK(saved->access_token == "at");
    CHECK(saved->email == "ada@example.com");

    // The model is left to the catalog selector (gateway order), so a stale
    // /model choice cannot outlive the sign-in.
    std::ifstream defaults(config.path() / "model_defaults.json");
    simdjson::dom::parser parser;
    const std::string selection((std::istreambuf_iterator<char>(defaults)), {});
    const auto document = parser.parse(selection);
    std::string_view provider;
    REQUIRE(document["default_provider"].get(provider) == simdjson::SUCCESS);
    CHECK(provider == "qwen-subscription");
    std::string_view mode;
    REQUIRE(document["default_model_selection"].get(mode) == simdjson::SUCCESS);
    CHECK(mode == "manual");
    std::string_view model;
    CHECK(document["providers"]["qwen-subscription"]["model"].get(model) != simdjson::SUCCESS);
}

TEST_CASE("Without a chosen model the subscription takes the gateway's first entry",
          "[qwen][subscription][catalog]") {
    const auto selector = providers::make_qwen_subscription_model_catalog_selector();
    std::vector<ModelInfo> models(2);
    models[0].canonical_id = "qwen4-plus";
    models[1].canonical_id = "qwen3.8-max";

    CHECK(selector->select(models).model == "qwen4-plus");
    const auto none = selector->select({});
    CHECK(none.model.empty());
    CHECK_THAT(none.error, ContainsSubstring("still signed in"));
}

TEST_CASE("Stopping at an unpaid plan keeps the session but reports the requirement",
          "[qwen][subscription][oauth][login]") {
    SignInServer gateway({"lite"});
    if (gateway.server.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    TempDir config;
    const auto ui = std::make_shared<ScriptedUI>(std::deque<std::optional<std::string>>{"q"});
    const auto strategy = core::auth::make_qwen_subscription_authentication_strategy({
        .auth_origin = gateway.server.origin(),
        .ui = ui,
        .open_browser = [](std::string_view) {},
        .sleep = [](std::chrono::milliseconds) {},
    });

    CHECK_THROWS_WITH(strategy->login(config.path().string()),
                      ContainsSubstring("needs a Plus or Pro plan"));
    CHECK(core::auth::FileTokenStore(config.path().string()).load("qwen-subscription"));
    CHECK_FALSE(std::filesystem::exists(config.path() / "model_defaults.json"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Web search
// ─────────────────────────────────────────────────────────────────────────────

namespace {

core::tools::ToolInvocationContext subscription_invocation(std::string token) {
    auto provider = std::make_shared<HttpLLMProvider>(
        std::string(subscription::kBaseUrl),
        core::auth::ApiKeyCredentialSource::as_bearer(std::move(token)),
        "qwen4-plus",
        std::make_unique<protocols::QwenSubscriptionProtocol>(),
        core::config::ApiType::OpenAI,
        "qwen-subscription");
    return {.session_context = test_support::make_workspace_session_context(),
            .provider_name = "qwen-subscription",
            .model_name = "qwen4-plus",
            .provider = std::move(provider)};
}

} // namespace

TEST_CASE("Qwen Subscription web search calls the gateway's MCP web_search tool",
          "[qwen][subscription][web]") {
    ScopedServer mcp;
    std::mutex mutex;
    std::vector<httplib::Request> received;
    mcp.server.Post("/api/v1/mcp/web_search", [&](const httplib::Request& request, httplib::Response& response) {
        {
            std::lock_guard lock(mutex);
            received.push_back(request);
        }
        simdjson::dom::parser parser;
        const auto body = parser.parse(request.body).value();
        const std::string_view method = body["method"].get_string().value();
        if (method == "initialize") {
            response.set_header("Mcp-Session-Id", "session-9");
            response.set_content("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":"
                                 "{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},"
                                 "\"serverInfo\":{\"name\":\"search\",\"version\":\"1\"}}}\n\n",
                                 "text/event-stream");
        } else if (method == "notifications/initialized") {
            response.status = 202;
        } else {
            response.set_content(R"({"jsonrpc":"2.0","id":2,"result":{"content":[)"
                                 R"({"type":"text","text":"Qwen 4 shipped this week.\n\nSources:"},)"
                                 R"({"type":"text","text":"https://example.com/a\nnot a url https://x\nhttps://example.com/a\nhttps://example.com/b"}]}})",
                                 "application/json");
        }
    });
    if (mcp.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");

    const auto backend = core::tools::web::make_qwen_subscription_web_search_backend(
        {.search_url = mcp.origin() + "/api/v1/mcp/web_search"});
    const auto invocation = subscription_invocation("sub-token");
    REQUIRE(backend->supports(invocation));
    const auto result = backend->search({.query = "qwen 4", .limit = 5}, invocation);
    REQUIRE(result);
    CHECK_THAT(result->answer, ContainsSubstring("Qwen 4 shipped this week."));
    REQUIRE(result->results.size() == 2);
    CHECK(result->results[0].url == "https://example.com/a");
    CHECK(result->results[1].url == "https://example.com/b");

    std::lock_guard lock(mutex);
    REQUIRE(received.size() == 3);
    for (const auto& request : received) {
        CHECK(request.get_header_value("Authorization") == "Bearer sub-token");
        CHECK_THAT(request.get_header_value("Accept"), ContainsSubstring("text/event-stream"));
    }
    CHECK_FALSE(received[0].has_header("Mcp-Session-Id"));
    CHECK(received[1].get_header_value("Mcp-Session-Id") == "session-9");
    CHECK(received[2].get_header_value("MCP-Protocol-Version") == "2025-06-18");
    CHECK_THAT(received[2].body, ContainsSubstring(R"("name":"web_search")"));
    CHECK_THAT(received[2].body, ContainsSubstring(R"("query":"qwen 4")"));

    CHECK_FALSE(backend->supports(core::tools::ToolInvocationContext{
        .session_context = test_support::make_workspace_session_context()}));
    CHECK_FALSE(backend->search({.query = std::string(501, 'q')}, invocation));
}

TEST_CASE("Qwen Subscription web search reports exhausted credits", "[qwen][subscription][web]") {
    ScopedServer mcp;
    mcp.server.Post("/mcp", [](const httplib::Request&, httplib::Response& response) {
        response.status = 402;
        response.set_content(R"({"error":{"code":"insufficient_quota","quota":{"member_type":"plus"}}})",
                             "application/json");
    });
    if (mcp.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    const auto backend = core::tools::web::make_qwen_subscription_web_search_backend(
        {.search_url = mcp.origin() + "/mcp"});
    const auto result = backend->search({.query = "q"}, subscription_invocation("t"));
    REQUIRE_FALSE(result);
    CHECK_THAT(result.error(), ContainsSubstring("run /upgrade"));
}
