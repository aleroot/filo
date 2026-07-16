#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/auth/AuthenticationManager.hpp"
#include "core/auth/OAuthCredentialSource.hpp"
#include "core/auth/XaiOAuthCredentialSource.hpp"
#include "core/auth/XaiOAuthFlow.hpp"
#include "core/llm/ModelCatalogProvider.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/protocols/GrokProtocol.hpp"

#include <chrono>
#include <memory>
#include <ranges>

using Catch::Matchers::ContainsSubstring;

namespace {

class StaticTokenStore final : public core::auth::ITokenStore {
public:
    explicit StaticTokenStore(core::auth::OAuthToken token)
        : token_(std::move(token)) {}

    std::optional<core::auth::OAuthToken> load(std::string_view) override {
        return token_;
    }
    void save(std::string_view, const core::auth::OAuthToken& token) override {
        token_ = token;
    }
    void clear(std::string_view) override { token_.reset(); }

private:
    std::optional<core::auth::OAuthToken> token_;
};

class UnusedOAuthFlow final : public core::auth::IOAuthFlow {
public:
    core::auth::OAuthToken login() override {
        throw std::runtime_error("unexpected login");
    }
    core::auth::OAuthToken refresh(std::string_view) override {
        throw std::runtime_error("unexpected refresh");
    }
};

[[nodiscard]] std::int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

} // namespace

TEST_CASE("xAI OAuth uses the pinned Grok Build public client", "[xai][oauth]") {
    CHECK(core::auth::XaiOAuthFlow::kClientId
        == "b1a00492-073a-47ea-816f-4c329264a828");
    const auto scopes = core::auth::XaiOAuthFlow::default_scopes();
    CHECK(std::ranges::find(scopes, "grok-cli:access") != scopes.end());
    CHECK(std::ranges::find(scopes, "offline_access") != scopes.end());
    CHECK(std::ranges::find(scopes, "conversations:write") != scopes.end());
}

TEST_CASE("xAI authorization URL carries PKCE OIDC and Grok referrer fields",
          "[xai][oauth][pkce]") {
    const auto url = core::auth::XaiOAuthFlow::build_authorization_url(
        "https://auth.x.ai/authorize",
        core::auth::XaiOAuthFlow::kClientId,
        "http://127.0.0.1:1234/callback",
        {"openid", "offline_access", "grok-cli:access"},
        "state-value",
        "nonce-value",
        "challenge-value",
        core::auth::XaiOAuthFlow::kReferrer);

    CHECK_THAT(url, ContainsSubstring("response_type=code"));
    CHECK_THAT(url, ContainsSubstring("client_id=b1a00492-073a-47ea-816f-4c329264a828"));
    CHECK_THAT(url, ContainsSubstring("redirect_uri=http%3A%2F%2F127.0.0.1%3A1234%2Fcallback"));
    CHECK_THAT(url, ContainsSubstring("code_challenge_method=S256"));
    CHECK_THAT(url, ContainsSubstring("state=state-value"));
    CHECK_THAT(url, ContainsSubstring("nonce=nonce-value"));
    CHECK_THAT(url, ContainsSubstring("referrer=grok-build"));
}

TEST_CASE("xAI OAuth parses discovery and token responses", "[xai][oauth]") {
    const auto discovery = core::auth::XaiOAuthFlow::parse_discovery_response(R"JSON({
      "authorization_endpoint":"https://auth.x.ai/oauth2/auth",
      "token_endpoint":"https://auth.x.ai/oauth2/token"
    })JSON");
    CHECK(discovery.authorization_endpoint == "https://auth.x.ai/oauth2/auth");
    CHECK(discovery.token_endpoint == "https://auth.x.ai/oauth2/token");
    CHECK(discovery.revocation_endpoint.empty());

    const auto with_revocation = core::auth::XaiOAuthFlow::parse_discovery_response(R"JSON({
      "authorization_endpoint":"https://auth.x.ai/oauth2/auth",
      "token_endpoint":"https://auth.x.ai/oauth2/token",
      "revocation_endpoint":"https://auth.x.ai/oauth2/revoke"
    })JSON");
    CHECK(with_revocation.revocation_endpoint == "https://auth.x.ai/oauth2/revoke");

    const auto token = core::auth::XaiOAuthFlow::parse_token_response(
        R"JSON({"access_token":"access","refresh_token":"refresh","expires_in":3600})JSON",
        1000,
        core::auth::XaiOAuthFlow::kIssuer,
        core::auth::XaiOAuthFlow::kClientId);
    CHECK(token.access_token == "access");
    CHECK(token.refresh_token == "refresh");
    CHECK(token.expires_at == 4600);
    CHECK(token.issuer == core::auth::XaiOAuthFlow::kIssuer);
    CHECK(token.client_id == core::auth::XaiOAuthFlow::kClientId);
}

TEST_CASE("xAI OAuth credentials include Grok session transport markers",
          "[xai][oauth][transport]") {
    core::auth::OAuthToken token;
    token.access_token = "session-access";
    token.refresh_token = "session-refresh";
    token.expires_at = now_unix() + 3600;
    token.issuer = std::string(core::auth::XaiOAuthFlow::kIssuer);
    token.client_id = std::string(core::auth::XaiOAuthFlow::kClientId);

    auto manager = std::make_shared<core::auth::OAuthTokenManager>(
        "grok",
        std::make_shared<UnusedOAuthFlow>(),
        std::make_shared<StaticTokenStore>(token),
        false);
    auto generic = std::make_shared<core::auth::OAuthCredentialSource>(manager);
    const auto generic_auth = generic->get_auth();
    CHECK(generic_auth.headers.find("X-XAI-Token-Auth")
        == generic_auth.headers.end());

    core::auth::XaiOAuthCredentialSource source(generic);
    const auto auth = source.get_auth();

    CHECK(auth.headers.at("Authorization") == "Bearer session-access");
    CHECK(auth.headers.at("X-XAI-Token-Auth") == "xai-grok-cli");
    CHECK(auth.headers.at("x-authenticateresponse") == "authenticate-response");
    CHECK(auth.headers.at("x-grok-client-identifier") == "filo");
    CHECK(auth.properties.at("oauth_issuer") == core::auth::XaiOAuthFlow::kIssuer);
}

TEST_CASE("Grok protocols add request-scoped proxy routing headers",
          "[xai][grok][transport]") {
    core::llm::ChatRequest request;
    request.model = "grok-4.5";
    request.session_id = "conversation-1";
    request.transport_turn_id = "request-1";
    request.auth_properties["oauth"] = "1";
    request.auth_properties["oauth_issuer"] = "https://auth.x.ai";

    cpr::Header chat_headers;
    core::llm::protocols::GrokProtocol chat;
    chat.prepare_headers(
        chat_headers, request, "https://cli-chat-proxy.grok.com/v1");
    CHECK(chat_headers.at("x-grok-conv-id") == "conversation-1");
    CHECK(chat_headers.at("x-grok-req-id") == "request-1");
    CHECK(chat_headers.at("x-grok-model-override") == "grok-4.5");

    cpr::Header response_headers;
    core::llm::protocols::GrokResponsesProtocol responses;
    responses.prepare_headers(
        response_headers, request, "https://cli-chat-proxy.grok.com/v1");
    CHECK(response_headers.at("X-XAI-Token-Auth") == "xai-grok-cli");
    CHECK(response_headers.at("x-grok-client-identifier") == "filo");

    SECTION("identity headers are restricted to the exact HTTPS proxy host") {
        for (const std::string_view untrusted_url : {
                 "https://cli-chat-proxy.grok.com.evil.example/v1",
                 "http://cli-chat-proxy.grok.com/v1",
                 "https://user@cli-chat-proxy.grok.com/v1",
             }) {
            cpr::Header untrusted_headers;
            responses.prepare_headers(untrusted_headers, request, untrusted_url);
            CHECK(untrusted_headers.find("X-XAI-Token-Auth")
                  == untrusted_headers.end());
            CHECK(untrusted_headers.find("x-grok-conv-id")
                  == untrusted_headers.end());
        }
    }
}

TEST_CASE("Grok OAuth catalogs retain session-only models", "[xai][grok][models]") {
    constexpr std::string_view body = R"JSON({"models":[
      {"slug":"grok-4.5","display_name":"Grok 4.5","supported_in_api":false},
      {"slug":"grok-api","display_name":"Grok API","supported_in_api":true}
    ]})JSON";

    core::llm::OpenAICompatibleModelCatalogProvider public_catalog("grok", false);
    const auto public_models = public_catalog.parse_models_response(body);
    REQUIRE(public_models.ok());
    REQUIRE(public_models.models.size() == 1);
    CHECK(public_models.models.front().canonical_id == "grok-api");

    core::llm::OpenAICompatibleModelCatalogProvider session_catalog("grok", true);
    const auto session_models = session_catalog.parse_models_response(body);
    REQUIRE(session_models.ok());
    CHECK(session_models.models.size() == 2);

    const auto build = core::llm::ModelRegistry::instance().get_info("grok-build");
    REQUIRE(build.has_value());
    CHECK(build->context_window == 500000);
    CHECK(build->provider == "grok");
}

TEST_CASE("Grok OAuth provider resolves to the CLI chat proxy", "[xai][grok][factory]") {
    core::config::ProviderConfig config;
    config.auth_type = "oauth_xai";
    config.model = "grok-build";

    const auto provider = core::llm::ProviderFactory::create_provider("grok", config);
    REQUIRE(provider != nullptr);
    const auto metadata = provider->metadata();
    REQUIRE(metadata.has_value());
    CHECK(metadata->base_url == "https://cli-chat-proxy.grok.com/v1");
    CHECK_FALSE(provider->should_estimate_cost());
}
