#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/auth/AuthenticationManager.hpp"
#include "core/auth/FileTokenStore.hpp"
#include "core/auth/ConfigCredentialSource.hpp"
#include "core/auth/OAuthCredentialSource.hpp"
#include "core/config/ConfigSource.hpp"
#include "core/auth/IOAuthFlow.hpp"
#include "core/auth/KimiOAuthFlow.hpp"
#include "core/auth/KimiSessionStore.hpp"
#include "core/auth/OAuthTokenManager.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/mcp/McpOAuth.hpp"
#include "core/mcp/McpConnectionManager.hpp"
#include <httplib.h>
#include "core/portability/ConfigPortability.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/Uuid.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <tuple>
#include <thread>
#include <mutex>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using core::config::ConfigManager;
using core::auth::FileTokenStore;
using core::auth::OAuthToken;
using core::utils::JsonValue;

struct Env {
    std::string name;
    std::optional<std::string> old;
    Env(std::string key, std::optional<std::string> value) : name(std::move(key)) {
        if (const auto* v = std::getenv(name.c_str())) old = v;
        if (value) setenv(name.c_str(), value->c_str(), 1);
        else unsetenv(name.c_str());
    }
    void restore() const {
        if (old) setenv(name.c_str(), old->c_str(), 1);
        else unsetenv(name.c_str());
    }
    ~Env() { restore(); }
};

void write(const fs::path& path, std::string_view text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}
std::string read(const fs::path& path) { return core::utils::read_file(path).value_or(""); }

OAuthToken token(std::string account) {
    OAuthToken t;
    t.access_token = account + "-access";
    t.refresh_token = account + "-refresh";
    t.expires_at = 4102444800LL;
    t.account_id = account;
    return t;
}

struct Fixture {
    fs::path root = fs::canonical(fs::temp_directory_path()) / ("filo_overlay_" + core::utils::random_uuid_v4());
    fs::path base = root / "xdg" / "filo";
    fs::path project = root / "project";
    fs::path custom = root / "config_back";
    fs::path cwd = fs::current_path();
    Env xdg{"XDG_CONFIG_HOME", (root / "xdg").string()};
    Env profile{"FILO_PROFILE", std::nullopt};
    Env state_home{"XDG_STATE_HOME", (root / "state").string()};
    ConfigManager& manager = ConfigManager::get_instance();
    Fixture() {
        manager.set_config_override(std::nullopt);
        fs::create_directories(base);
        fs::create_directories(project);
        fs::create_directories(custom);
    }
    ~Fixture() {
        fs::current_path(cwd);
        manager.set_config_override(std::nullopt);
        xdg.restore();
        profile.restore();
        manager.load();
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
    void select(const fs::path& source) {
        manager.set_config_override(source);
        manager.load(project);
    }
    fs::path bundle(bool secrets = true,
                    core::portability::BundleEncoding encoding = core::portability::BundleEncoding::Json) {
        const auto result = core::portability::export_bundle(
            {.config_dir = custom, .memory_file = custom / "memory.json", .home = root.string()},
            {.include_secrets = secrets});
        const auto path = root / (encoding == core::portability::BundleEncoding::Gzip ? "work.json.gz" : "work.json");
        write(path, core::portability::encode_bundle(result.bundle, encoding));
        return path;
    }
};

class RefreshFlow : public core::auth::IOAuthFlow {
public:
    int calls = 0;
    fs::path log;
    OAuthToken login() override { throw std::runtime_error("unexpected login"); }
    OAuthToken refresh(std::string_view refresh_token) override {
        ++calls;
        if (!log.empty()) std::ofstream(log, std::ios::app) << "refresh\n";
        auto t = token(std::string(refresh_token));
        t.access_token = "rotated";
        return t;
    }
};
std::shared_ptr<core::auth::ConfigCredentialSource> credential(
    std::shared_ptr<const core::config::ConfigSource> source, std::shared_ptr<RefreshFlow> flow) {
    return std::make_shared<core::auth::ConfigCredentialSource>(std::move(source), "claude",
        [flow](const fs::path& directory) {
            auto manager = std::make_shared<core::auth::OAuthTokenManager>("claude", flow,
                std::make_shared<FileTokenStore>(directory.string()), false);
            return std::make_shared<core::auth::OAuthCredentialSource>(std::move(manager));
        });
}

}

TEST_CASE("Config directory overrides only present fields including false zero and empty", "[config][config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"default_provider":"claude","ui_theme":"dark",
        "providers":{"claude":{"model":"personal-model","base_url":"https://personal",
            "api_key":"personal-key","auth_type":"oauth_claude","thinking_budget":8000,"stream_usage":true},
            "custom":{"api_type":"openai","model":"local-model"}},
        "hooks":{"stop":[{"command":"echo local"}]}})");
    write(f.base / "model_defaults.json", R"({"providers":{"claude":{"model":"saved-personal"}}})");
    write(f.project / ".filo" / "settings.json", R"({"ui_footer":"hide"})");
    write(f.custom / "config.json", R"({"providers":{"claude":{"model":"work-model",
        "auth_type":"","thinking_budget":0,"stream_usage":false}},"hooks":{"stop":[]},
        "tool_recovery":false,"strict_tool_schemas":true})");
    const auto before = read(f.base / "config.json");
    f.select(f.custom);
    const auto& config = f.manager.get_config();
    CHECK(config.default_provider == "claude");
    CHECK(config.ui_theme == "dark");
    CHECK(config.ui_footer == "hide");
    CHECK(config.providers.at("claude").model == "work-model");
    CHECK(config.providers.at("claude").api_key == "personal-key");
    CHECK(config.providers.at("claude").base_url == "https://personal");
    CHECK(config.providers.at("claude").auth_type.empty());
    CHECK(config.providers.at("claude").thinking_budget == 0);
    CHECK_FALSE(config.providers.at("claude").stream_usage);
    CHECK(config.providers.at("custom").model == "local-model");
    CHECK(config.hooks.stop.empty());
    CHECK_FALSE(config.tool_recovery);
    CHECK(config.strict_tool_schemas);
    CHECK(read(f.base / "config.json") == before);
}

TEST_CASE("Exported bundles stay sparse and their refreshed logins survive reopen and logout", "[config-overlay][portability]") {
    Fixture f;
    write(f.base / "config.json", R"({"default_provider":"claude","ui_theme":"light",
        "providers":{"claude":{"model":"personal","base_url":"https://local"}}})");
    write(f.custom / "auth_defaults.json", R"({"providers":{"claude":{"auth_type":"oauth_claude"}}})");
    FileTokenStore(f.base.string()).save("claude", token("personal"));
    FileTokenStore(f.custom.string()).save("claude", token("work"));
    const auto path = f.bundle();
    const auto original = read(path);
    const auto personal = read(f.base / "oauth_claude.json");
    f.select(path);
    const fs::path state = f.manager.get_config_dir();
    CHECK(state.parent_path() == f.root / "state/filo/configs");
    CHECK_FALSE(fs::exists(path.string() + ".d"));
    CHECK_FALSE(fs::exists(state / "config.json"));
    CHECK(f.manager.get_config().ui_theme == "light");
    CHECK(f.manager.get_config().providers.at("claude").model == "personal");
    FileTokenStore runtime(state.string());
    REQUIRE(runtime.load("claude"));
    CHECK(runtime.load("claude")->account_id == "work");
    {
        auto lock = runtime.acquire_refresh_lock("claude");
        runtime.save("claude", token("work-rotated"));
    }
    f.manager.set_config_override(std::nullopt);
    f.select(path);
    CHECK(runtime.load("claude")->account_id == "work-rotated");
    const auto auth = core::auth::AuthenticationManager::create_with_defaults(state.string());
    CHECK_NOTHROW(auth.logout("claude", false, f.manager.get_config_source().get()));
    CHECK_FALSE(runtime.load("claude"));
    f.select(path);
    CHECK_FALSE(runtime.load("claude"));
    CHECK(read(path) == original);
    CHECK(read(f.base / "oauth_claude.json") == personal);
    CHECK((fs::status(state).permissions() & fs::perms::group_all) == fs::perms::none);
    CHECK_FALSE(fs::exists(state / "auth_defaults.json"));
    CHECK((fs::status(state / "oauth_claude.json.selected").permissions() & fs::perms::others_all) == fs::perms::none);
}

TEST_CASE("A compressed bundle is read in place and signs in with its Claude login", "[config-overlay][portability][gzip]") {
    Fixture f;
    write(f.base / "config.json", R"({"default_provider":"openai","ui_theme":"light"})");
    write(f.custom / "config.json", R"({"default_provider":"claude",
        "providers":{"claude":{"model":"claude-sonnet-4-6"}}})");
    write(f.custom / "auth_defaults.json", R"({"providers":{"claude":{"auth_type":"oauth_claude"}}})");
    FileTokenStore(f.base.string()).save("claude", token("personal"));
    FileTokenStore(f.custom.string()).save("claude", token("work"));
    const auto path = f.bundle(true, core::portability::BundleEncoding::Gzip);
    const auto original = read(path);
    REQUIRE(core::portability::detect_bundle_encoding(original) == core::portability::BundleEncoding::Gzip);

    f.select(path);
    const auto& config = f.manager.get_config();
    CHECK(config.default_provider == "claude");
    CHECK(config.ui_theme == "light");  // the personal setup still fills the gaps
    CHECK(config.providers.at("claude").model == "claude-sonnet-4-6");
    CHECK(config.providers.at("claude").auth_type == "oauth_claude");

    const fs::path state = f.manager.get_config_dir();
    CHECK(state.parent_path() == f.root / "state/filo/configs");
    auto auth = core::auth::AuthenticationManager::create_with_defaults(state.string());
    core::config::ProviderConfig provider = config.providers.at("claude");
    provider.config_source = f.manager.get_config_source();
    const auto source = auth.create_credential_source("claude", provider);
    REQUIRE(source);
    CHECK(source->get_auth().headers.at("Authorization") == "Bearer work-access");

    // The archive is the read-only source: nothing is unpacked beside it and
    // it is never rewritten; the personal login is untouched.
    CHECK(read(path) == original);
    CHECK_FALSE(fs::exists(path.string() + ".d"));
    CHECK_FALSE(fs::exists(state / "config.json"));
    CHECK(FileTokenStore(f.base.string()).load("claude")->account_id == "personal");

    // Reopening keeps the same state, which is keyed by the archive's path.
    f.manager.set_config_override(std::nullopt);
    f.select(path);
    CHECK(f.manager.get_config_dir() == state.string());
}

TEST_CASE("A damaged compressed bundle is rejected before creating state", "[config-overlay][portability][gzip]") {
    Fixture f;
    FileTokenStore(f.custom.string()).save("claude", token("work"));
    const auto path = f.bundle(true, core::portability::BundleEncoding::Gzip);
    auto bytes = read(path);
    bytes.resize(bytes.size() / 2);
    write(path, bytes);
    try {
        f.select(path);
        FAIL("a truncated archive was accepted");
    } catch (const std::exception& error) {
        CHECK_THAT(error.what(), Catch::Matchers::ContainsSubstring("compressed bundle is damaged"));
    }
    CHECK_FALSE(fs::exists(f.root / "state/filo/configs"));
}

TEST_CASE("Inherited credentials refresh only in isolated state and explicit logout stays local", "[config-overlay][oauth]") {
    Fixture f;
    auto expired = token("personal");
    expired.expires_at = 1;
    FileTokenStore(f.base.string()).save("claude", expired);
    f.select(f.custom);
    auto flow = std::make_shared<RefreshFlow>();
    auto source = credential(f.manager.get_config_source(), flow);
    CHECK(source->get_auth().headers.at("Authorization") == "Bearer rotated");
    CHECK(flow->calls == 1);
    CHECK(FileTokenStore(f.base.string()).load("claude")->access_token == "personal-access");
    CHECK(FileTokenStore(f.manager.get_config_dir()).load("claude")->access_token == "rotated");
    CHECK_FALSE(fs::exists(f.custom / "oauth_claude.json"));
    const auto personal = read(f.base / "oauth_claude.json");
    auto auth = core::auth::AuthenticationManager::create_with_defaults(f.manager.get_config_dir());
    auth.logout("claude", false, f.manager.get_config_source().get());
    CHECK_THROWS(source->get_auth());
    CHECK(read(f.base / "oauth_claude.json") == personal);
    FileTokenStore(f.manager.get_config_dir()).save("claude", token("new-work-login"));
    CHECK(source->get_auth().headers.at("Authorization") == "Bearer new-work-login-access");
}

TEST_CASE("Corrupt or removed state credentials never fall through to the source or personal account", "[config-overlay][oauth]") {
    Fixture f;
    FileTokenStore(f.base.string()).save("claude", token("personal"));
    FileTokenStore(f.custom.string()).save("claude", token("work"));
    f.select(f.custom);
    const fs::path state = f.manager.get_config_dir();
    const auto original = read(f.custom / "oauth_claude.json");
    write(state / "oauth_claude.json", "not json");
    FileTokenStore store(state.string());
    auto source = credential(f.manager.get_config_source(), std::make_shared<RefreshFlow>());
    CHECK_FALSE(store.load("claude"));
    CHECK_FALSE(store.load("claude"));
    CHECK_THROWS(source->get_auth());
    f.manager.load(f.project);
    CHECK_FALSE(store.load("claude"));
    store.save("claude", token("work"));
    fs::remove(state / "oauth_claude.json");
    fs::remove(state / "oauth_claude.json.selected");
    CHECK_THROWS(source->get_auth());
    f.manager.load(f.project);
    auto reopened = credential(f.manager.get_config_source(), std::make_shared<RefreshFlow>());
    CHECK_THROWS(reopened->get_auth());
    CHECK(read(f.custom / "oauth_claude.json") == original);
}

TEST_CASE("Settings model login and MCP persistence retain the chosen config directory", "[config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"default_provider":"grok","ui_theme":"dark"})");
    write(f.custom / "config.json", R"({"providers":{"claude":{"model":"work"}},
        "profiles":{"work":{"ui_banner":"hide"}}})");
    const auto before = read(f.base / "config.json");
    f.select(f.custom);
    std::string error;
    REQUIRE(f.manager.persist_managed_setting(core::config::SettingsScope::User,
        core::config::ManagedSettingKey::UiTheme, "light", f.project, &error));
    REQUIRE(f.manager.persist_model_defaults("claude", "manual", "chosen-work-model", &error));
    REQUIRE(f.manager.persist_login_profile("claude", &error));
    REQUIRE(f.manager.persist_active_profile("work", f.project, &error));
    core::config::McpServerConfig server;
    server.name = "work-tools";
    server.transport = "stdio";
    server.command = "work-mcp";
    REQUIRE(f.manager.persist_mcp_server(server, core::config::SettingsScope::User, f.project, &error));
    f.manager.load(f.project);
    CHECK(fs::path(f.manager.get_config_dir()).parent_path() == f.root / "state/filo/configs");
    CHECK(f.manager.get_config().ui_theme == "light");
    CHECK(f.manager.get_config().ui_banner == "hide");
    CHECK(f.manager.get_config().providers.at("claude").model == "chosen-work-model");
    CHECK(f.manager.get_auth_default_provider() == "claude");
    for (const auto file : {"settings.json", "model_defaults.json", "auth_defaults.json",
                            "profile_defaults.json", "mcp_servers.json"}) {
        CHECK(fs::exists(fs::path(f.manager.get_config_dir()) / file));
        CHECK_FALSE(fs::exists(f.custom / file));
        CHECK_FALSE(fs::exists(f.base / file));
    }
    CHECK(read(f.base / "config.json") == before);
}

TEST_CASE("Named MCP overlays preserve inherited secrets and unrelated servers", "[config-overlay][mcp-oauth]") {
    Fixture f;
    write(f.base / "config.json", R"({"mcp_servers":[{"name":"work","transport":"http",
        "url":"https://personal","headers":{"Authorization":"Bearer local"},"env":["A=1"]},
        {"name":"other","transport":"stdio","command":"other"}]})");
    write(f.custom / "mcp_servers.json", R"({"mcp_servers":[{"name":"work","url":"https://work",
        "headers":{"X-Work":"present"}}]})");
    const auto id = core::mcp::mcp_oauth_provider_id("work");
    FileTokenStore(f.base.string()).save(id, token("personal-mcp"));
    f.select(f.custom);
    const auto& servers = f.manager.get_config().mcp_servers;
    REQUIRE(servers.size() == 2);
    CHECK(servers[0].url == "https://work");
    REQUIRE(servers[0].headers.size() == 2);
    CHECK(servers[0].headers[0].second == "Bearer local");
    CHECK(servers[0].headers[1].second == "present");
    CHECK(servers[0].env == std::vector<std::string>{"A=1"});
    auto oauth_config = servers[0];
    oauth_config.headers.clear();
    oauth_config.auth = "oauth";
    const auto config_source = f.manager.get_config_source();
    core::mcp::McpHttpAuth auth(oauth_config, f.manager.get_config_dir(), [config_source](bool force) {
        const auto directory = config_source->credential_directory(core::mcp::mcp_oauth_provider_id("work")).string();
        return force ? core::mcp::force_refresh_mcp_oauth_token("work", directory)
                     : core::mcp::load_valid_mcp_oauth_token("work", directory);
    });
    CHECK(auth.request_headers().at(0).second == "Bearer personal-mcp-access");
    FileTokenStore(f.manager.get_config_dir()).save(id, token("work-mcp"));
    CHECK(auth.request_headers().at(0).second == "Bearer work-mcp-access");
    config_source->claim_credential(id);
    core::mcp::logout_mcp_oauth("work", f.manager.get_config_dir());
    CHECK(auth.request_headers().empty());
    CHECK(FileTokenStore(f.base.string()).load(id)->account_id == "personal-mcp");
    write(f.custom / "mcp_servers.json", R"({"mcp_servers":[{"name":"work",
        "headers":{"authorization":"Bearer work-static"}}]})");
    f.manager.load(f.project);
    const auto& headers = f.manager.get_config().mcp_servers.at(0).headers;
    REQUIRE(headers.size() == 1);
    CHECK(headers[0].second == "Bearer work-static");
}

TEST_CASE("An overlay does not create a missing default config and resolves paths before chdir", "[config-overlay]") {
    Fixture f;
    fs::remove_all(f.base);
    write(f.custom / "config.json", R"({"ui_theme":"light"})");
    fs::current_path(f.root);
    f.manager.set_config_override("config_back");
    fs::current_path(f.project);
    f.manager.load();
    CHECK(f.manager.get_config().ui_theme == "light");
    CHECK(fs::path(f.manager.get_config_dir()).parent_path() == f.root / "state/filo/configs");
    CHECK_FALSE(fs::exists(f.base));
    f.manager.set_config_override(std::nullopt);
    f.manager.load();
    CHECK(fs::exists(f.base / "config.json"));
}

TEST_CASE("Partial router local model and profile overrides preserve inherited fields", "[config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"active_profile":"daily",
        "profiles":{"daily":{"hooks":{"stop":[{"command":"echo once"}]}}},
        "providers":{"local":{"api_type":"llamacpp","model_path":"/models/base.gguf",
            "llamacpp":{"threads":8,"use_mmap":true}}},
        "router":{"enabled":true,"default_policy":"personal","failover":{"max_wait_seconds":123},
            "guardrails":{"max_session_cost_usd":5},
            "policies":{"personal":{"defaults":[{"provider":"claude"}],"description":"local"}}}})");
    write(f.custom / "config.json", R"({"profiles":{"daily":{"providers":{"local":{
        "llamacpp":{"use_mmap":false}}}}},"router":{"failover":{"on_exhaustion":"wait"},
        "guardrails":{"enforce_on_local":true},"policies":{"personal":{"description":"work"}}}})");
    f.select(f.custom);
    const auto& config = f.manager.get_config();
    REQUIRE(config.hooks.stop.size() == 1);
    CHECK(config.router.enabled);
    CHECK(config.router.default_policy == "personal");
    CHECK(config.router.failover.max_wait_seconds == 123);
    CHECK(config.router.failover.wait_on_exhaustion);
    REQUIRE(config.router.guardrails);
    CHECK(config.router.guardrails->max_session_cost_usd == 5);
    CHECK(config.router.guardrails->enforce_on_local);
    REQUIRE(config.router.policies.at("personal").defaults.size() == 1);
    CHECK(config.router.policies.at("personal").description == "work");
    REQUIRE(config.providers.at("local").local);
    CHECK(config.providers.at("local").local->model_path == "/models/base.gguf");
    CHECK(config.providers.at("local").local->llamacpp->threads == 8);
    CHECK(config.providers.at("local").local->llamacpp->use_mmap == false);
    f.manager.load(f.project);
    CHECK(f.manager.get_config().hooks.stop.size() == 1);
}

TEST_CASE("Config overrides reject malformed inputs without changing local configuration", "[config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"ui_theme":"dark"})");
    const auto before = read(f.base / "config.json");
    SECTION("missing path") { CHECK_THROWS(f.select(f.root / "missing")); }
    SECTION("invalid bundle") {
        write(f.root / "invalid.json", "{}");
        CHECK_THROWS(f.select(f.root / "invalid.json"));
        CHECK_FALSE(fs::exists(f.root / "invalid.json.d"));
    }
    SECTION("invalid directory JSON") {
        write(f.custom / "config.json", "{broken");
        CHECK_THROWS(f.select(f.custom));
    }
    SECTION("invalid credential JSON") {
        write(f.custom / "oauth_claude.json", "{broken");
        CHECK_THROWS(f.select(f.custom));
        CHECK(read(f.custom / "oauth_claude.json") == "{broken");
    }
    SECTION("credential is a named pipe") {
        REQUIRE(::mkfifo((f.custom / "oauth_claude.json").c_str(), 0600) == 0);
        CHECK_THROWS(f.select(f.custom));
    }
    SECTION("dangling credential symlink") {
        fs::create_symlink(f.root / "missing-token", f.custom / "oauth_claude.json");
        CHECK_THROWS(f.select(f.custom));
    }
    SECTION("invalid provider type") {
        write(f.custom / "config.json", R"({"providers":{"claude":{"stream_usage":"false"}}})");
        CHECK_THROWS(f.select(f.custom));
    }
    SECTION("invalid nested local type") {
        write(f.custom / "config.json", R"({"providers":{"local":{"llamacpp":{"use_mmap":"false"}}}})");
        CHECK_THROWS(f.select(f.custom));
    }
    SECTION("invalid MCP headers") {
        write(f.custom / "mcp_servers.json", R"({"mcp_servers":[{"name":"tools","headers":{"Authorization":false}}]})");
        CHECK_THROWS(f.select(f.custom));
    }
    SECTION("invalid profile inheritance") {
        write(f.custom / "config.json", R"({"profiles":{"daily":{"extends_from":[false]}}})");
        CHECK_THROWS(f.select(f.custom));
    }
    SECTION("unknown bundle file") {
        core::portability::Bundle bundle;
        bundle.items.push_back({.kind = core::portability::ItemKind::Provider, .name = "claude",
            .data = *JsonValue::parse(R"({"../../escape":{}})")});
        const auto path = f.root / "invalid.json";
        write(path, core::portability::serialize_bundle(bundle));
        CHECK_THROWS(f.select(path));
        CHECK_FALSE(fs::exists(path.string() + ".d"));
    }
    SECTION("duplicate bundle item") {
        core::portability::Bundle bundle;
        bundle.items.push_back({.kind = core::portability::ItemKind::Provider, .name = "claude",
            .data = *JsonValue::parse(R"({"config.json":{}})")});
        bundle.items.push_back(bundle.items.front());
        CHECK_THROWS(core::portability::config_overlay_files(bundle));
    }
    CHECK(read(f.base / "config.json") == before);
}

TEST_CASE("Config paths retain their state when supplied configuration or credentials change", "[config-overlay]") {
    Fixture f;
    write(f.custom / "config.json", R"({"ui_banner":"hide","providers":{"claude":{"model":"original"}}})");
    FileTokenStore(f.custom.string()).save("claude", token("work"));
    const auto path = f.bundle();
    f.select(path);
    const fs::path state = f.manager.get_config_dir();
    FileTokenStore(state.string()).save("claude", token("rotated"));
    SECTION("reopening never resets existing refreshed credentials or logout") {
        f.select(path);
        CHECK(FileTokenStore(state.string()).load("claude")->account_id == "rotated");
        auto auth = core::auth::AuthenticationManager::create_with_defaults(state.string());
        auth.logout("claude", false, f.manager.get_config_source().get());
        f.select(path);
        CHECK_FALSE(FileTokenStore(state.string()).load("claude"));
    }
    SECTION("whitespace edits keep refreshed tokens") {
        write(path, read(path) + "\n");
        f.select(path);
        CHECK(f.manager.get_config_dir() == state.string());
        CHECK(FileTokenStore(state.string()).load("claude")->account_id == "rotated");
    }
    SECTION("configuration edits are visible without resetting runtime state") {
        write(f.custom / "config.json", R"({"ui_banner":"show","providers":{"claude":{"model":"changed"}}})");
        f.bundle();
        f.manager.load(f.project);
        CHECK(f.manager.get_config_dir() == state.string());
        CHECK(f.manager.get_config().ui_banner == "show");
        CHECK(f.manager.get_config().providers.at("claude").model == "changed");
        CHECK(FileTokenStore(state.string()).load("claude")->account_id == "rotated");
    }
    SECTION("replacing credentials retains user and workspace settings and other logins") {
        std::string error;
        REQUIRE(f.manager.persist_managed_setting(core::config::SettingsScope::User,
            core::config::ManagedSettingKey::UiFooter, "hide", f.project, &error));
        REQUIRE(f.manager.persist_managed_setting(core::config::SettingsScope::Workspace,
            core::config::ManagedSettingKey::UiTheme, "light", f.project, &error));
        REQUIRE(f.manager.persist_model_defaults("claude", "manual", "saved-model", &error));
        FileTokenStore(state.string()).save("openai", token("unrelated-login"));
        const auto user_settings = read(state / "settings.json");
        const auto workspace_settings = read(f.manager.get_config_source()->workspace_state_directory(f.project) / "settings.json");
        const auto model_defaults = read(state / "model_defaults.json");
        FileTokenStore(f.custom.string()).save("claude", token("other-account"));
        f.bundle();
        f.manager.load(f.project);
        CHECK(f.manager.get_config_dir() == state.string());
        CHECK(FileTokenStore(f.manager.get_config_dir()).load("claude")->account_id == "other-account");
        CHECK(FileTokenStore(state.string()).load("openai")->account_id == "unrelated-login");
        CHECK(f.manager.get_config().ui_footer == "hide");
        CHECK(f.manager.get_config().ui_theme == "light");
        CHECK(f.manager.get_config().providers.at("claude").model == "saved-model");
        CHECK(read(state / "settings.json") == user_settings);
        CHECK(read(f.manager.get_config_source()->workspace_state_directory(f.project) / "settings.json") == workspace_settings);
        CHECK(read(state / "model_defaults.json") == model_defaults);
        CHECK(std::distance(fs::directory_iterator(state.parent_path()), fs::directory_iterator{}) == 1);
        FileTokenStore(state.string()).save("claude", token("other-rotated"));
        f.select(path);
        CHECK(FileTokenStore(state.string()).load("claude")->account_id == "other-rotated");
    }
    SECTION("removing supplied credentials resumes inheritance in the same state") {
        FileTokenStore(f.base.string()).save("claude", token("personal"));
        fs::remove(f.custom / "oauth_claude.json");
        f.bundle();
        f.manager.load(f.project);
        CHECK(f.manager.get_config_dir() == state.string());
        auto source = credential(f.manager.get_config_source(), std::make_shared<RefreshFlow>());
        CHECK(source->get_auth().headers.at("Authorization") == "Bearer personal-access");
        CHECK(FileTokenStore(f.base.string()).load("claude")->account_id == "personal");
    }
    SECTION("replacing credentials after logout selects the new supplied login") {
        auto auth = core::auth::AuthenticationManager::create_with_defaults(state.string());
        auth.logout("claude", false, f.manager.get_config_source().get());
        FileTokenStore(f.custom.string()).save("claude", token("new-work"));
        f.bundle();
        f.select(path);
        CHECK(f.manager.get_config_dir() == state.string());
        CHECK(FileTokenStore(state.string()).load("claude")->account_id == "new-work");
    }
    SECTION("symlinked state cannot write into the default config") {
        f.manager.set_config_override(std::nullopt);
        fs::remove_all(state);
        fs::create_directory_symlink(f.base, state);
        CHECK_THROWS(f.select(path));
        CHECK_FALSE(fs::exists(f.base / "oauth_claude.json"));
    }
    CHECK_FALSE(fs::exists(path.string() + ".d"));
}

TEST_CASE("Kimi uses the selected directory for device identity and region", "[config-overlay][oauth]") {
    Fixture f;
    Env host{"KIMI_OAUTH_HOST", std::nullopt};
    Env region{"KIMI_CODE_REGION", std::nullopt};
    Env api{"KIMI_CODE_BASE_URL", std::nullopt};
    FileTokenStore(f.custom.string()).save("kimi", token("work"));
    write(f.custom / "kimi_device_id", "12345678-1234-4234-8234-123456789abc");
    write(f.custom / "kimi_oauth_host", "https://auth.kimi.com");
    f.select(f.custom);
    auto session = std::make_shared<core::auth::KimiSessionStore>(f.manager.get_config_dir());
    const core::auth::KimiOAuthFlow flow(session);
    CHECK(read(fs::path(f.manager.get_config_dir()) / "kimi_device_id") == "12345678-1234-4234-8234-123456789abc");
    CHECK(read(f.custom / "kimi_device_id") == "12345678-1234-4234-8234-123456789abc");
    CHECK(session->persisted_oauth_host() == "https://auth.kimi.com");
    CHECK_FALSE(fs::exists(f.base / "kimi_device_id"));
}

TEST_CASE("Kimi refresh and request identity follow credential replacement in the same config state", "[config-overlay][kimi][oauth]") {
    Fixture f;
    Env host{"KIMI_OAUTH_HOST", std::nullopt};
    Env code_host{"KIMI_CODE_OAUTH_HOST", std::nullopt};
    Env region{"KIMI_CODE_REGION", std::nullopt};
    Env api{"KIMI_CODE_BASE_URL", std::nullopt};
    httplib::Server server;
    std::mutex mutex;
    std::string requested_path, requested_device, requested_token;
    server.Post(R"(/(old|new)/api/oauth/token)", [&](const auto& request, auto& response) {
        std::lock_guard lock(mutex);
        requested_path = request.path;
        requested_device = request.get_header_value("X-Msh-Device-Id");
        requested_token = request.get_param_value("refresh_token");
        response.set_content(R"({"access_token":"refreshed-access","refresh_token":"refreshed-token","expires_in":3600})", "application/json");
    });
    const auto port = server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::jthread listener([&] { server.listen_after_bind(); });
    struct Stop {
        httplib::Server& server;
        ~Stop() { server.stop(); }
    } stop{server};
    const auto origin = "http://127.0.0.1:" + std::to_string(port);
    auto old_token = token("old");
    old_token.device_id = "11111111-1111-4111-8111-111111111111";
    FileTokenStore(f.custom.string()).save("kimi", old_token);
    write(f.custom / "kimi_device_id", old_token.device_id);
    write(f.custom / "kimi_oauth_host", origin + "/old");
    f.select(f.custom);
    const fs::path state = f.manager.get_config_dir();
    std::string error;
    REQUIRE(f.manager.persist_managed_setting(core::config::SettingsScope::User,
        core::config::ManagedSettingKey::UiFooter, "hide", f.project, &error));
    const auto settings = read(state / "settings.json");
    core::config::ProviderConfig config;
    config.auth_type = "oauth_kimi";
    config.config_source = f.manager.get_config_source();
    const auto manager = core::auth::AuthenticationManager::create_with_defaults(state.string());
    auto source = manager.create_credential_source("kimi", config);
    REQUIRE(source);
    auto replacement = token("new");
    replacement.device_id = "22222222-2222-4222-8222-222222222222";
    replacement.expires_at = 1;
    FileTokenStore(f.custom.string()).save("kimi", replacement);
    write(f.custom / "kimi_device_id", replacement.device_id);
    write(f.custom / "kimi_oauth_host", origin + "/new");
    f.manager.load(f.project);
    CHECK(f.manager.get_config_dir() == state.string());
    CHECK(f.manager.get_config().ui_footer == "hide");
    const auto auth = source->get_auth();
    CHECK(auth.headers.at("Authorization") == "Bearer refreshed-access");
    CHECK(auth.properties.at("device_id") == replacement.device_id);
    {
        std::lock_guard lock(mutex);
        CHECK(requested_path == "/new/api/oauth/token");
        CHECK(requested_device == replacement.device_id);
        CHECK(requested_token == "new-refresh");
    }
    const core::llm::protocols::KimiProtocol protocol;
    CHECK(protocol.build_headers(auth).at("X-Msh-Device-Id") == replacement.device_id);
    CHECK_FALSE(fs::exists(f.base / "kimi_device_id"));
    CHECK_FALSE(fs::exists(f.base / "kimi_oauth_host"));
    CHECK(read(f.custom / "oauth_kimi.json").find("refreshed-access") == std::string::npos);
    f.select(f.custom);
    CHECK(f.manager.get_config_dir() == state.string());
    CHECK(read(state / "settings.json") == settings);
    CHECK(FileTokenStore(state.string()).load("kimi")->access_token == "refreshed-access");
}

TEST_CASE("Kimi OAuth uses injected session state without accessing configuration files", "[kimi][oauth]") {
    Fixture f;
    Env host{"KIMI_OAUTH_HOST", std::nullopt};
    Env code_host{"KIMI_CODE_OAUTH_HOST", std::nullopt};
    Env region{"KIMI_CODE_REGION", std::nullopt};
    Env api{"KIMI_CODE_BASE_URL", std::nullopt};
    CHECK_THROWS_AS(core::auth::KimiOAuthFlow(nullptr), std::invalid_argument);
    httplib::Server server;
    std::mutex mutex;
    std::string requested_path, requested_device;
    server.Post(R"(/(memory|replacement|alias|explicit)/api/oauth/token)", [&](const auto& request, auto& response) {
        std::lock_guard lock(mutex);
        requested_path = request.path;
        requested_device = request.get_header_value("X-Msh-Device-Id");
        response.set_content(R"({"access_token":"memory-access","expires_in":3600})", "application/json");
    });
    const auto port = server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::jthread listener([&] { server.listen_after_bind(); });
    struct Stop {
        httplib::Server& server;
        ~Stop() { server.stop(); }
    } stop{server};
    const auto origin = "http://127.0.0.1:" + std::to_string(port);
    class MemoryState final : public core::auth::IKimiSessionState {
    public:
        core::auth::KimiSessionIdentity identity;
        core::auth::KimiSessionIdentity load_identity() override { return identity; }
    };
    auto memory = std::make_shared<MemoryState>();
    memory->identity = {"first-device", origin + "/memory"};
    core::auth::KimiOAuthFlow flow(memory);
    CHECK(flow.refresh("fake-refresh").issuer == origin + "/memory");
    memory->identity = {"replacement-device", origin + "/replacement"};
    CHECK(flow.refresh("fake-refresh").issuer == origin + "/replacement");
    {
        std::lock_guard lock(mutex);
        CHECK(requested_path == "/replacement/api/oauth/token");
        CHECK(requested_device == "replacement-device");
    }
    {
        Env alias{"KIMI_OAUTH_HOST", origin + "/alias"};
        {
            Env primary{"KIMI_CODE_OAUTH_HOST", origin + "/explicit/"};
            CHECK(flow.refresh("fake-refresh").issuer == origin + "/explicit");
        }
        CHECK(flow.refresh("fake-refresh").issuer == origin + "/alias");
    }
    CHECK(fs::is_empty(f.base));
    CHECK(fs::is_empty(f.custom));
    CHECK_FALSE(fs::exists(f.root / "state"));
}

TEST_CASE("Concurrent Kimi pre-login sessions retain the same persistent identity", "[kimi][oauth][KimiSessionStore]") {
    Fixture f;
    const auto directory = (f.root / "prelogin-state").string();
    auto first = std::async(std::launch::async, [&] { return core::auth::KimiSessionStore(directory).device_id(); });
    auto second = std::async(std::launch::async, [&] { return core::auth::KimiSessionStore(directory).device_id(); });
    const auto id = first.get();
    CHECK(id == second.get());
    CHECK(core::auth::KimiSessionStore(directory).device_id() == id);
    CHECK(fs::is_empty(f.base));
}

TEST_CASE("Credential changes replace their companions without disturbing other accounts", "[config-overlay][oauth]") {
    Fixture f;
    FileTokenStore(f.custom.string()).save("kimi", token("old-kimi"));
    FileTokenStore(f.custom.string()).save("claude", token("work-claude"));
    write(f.custom / "kimi_device_id", "old-device");
    write(f.custom / "kimi_oauth_host", "https://auth.kimi.com");
    f.select(f.custom);
    const fs::path state = f.manager.get_config_dir();
    FileTokenStore(state.string()).save("claude", token("refreshed-claude"));
    FileTokenStore(state.string()).save("kimi", token("refreshed-kimi"));
    write(f.custom / "kimi_oauth_host", "https://auth.kimi.ai");
    fs::remove(f.custom / "kimi_device_id");
    f.select(f.custom);
    CHECK(f.manager.get_config_dir() == state.string());
    CHECK(FileTokenStore(state.string()).load("kimi")->account_id == "old-kimi");
    CHECK_FALSE(fs::exists(state / "kimi_device_id"));
    CHECK(read(state / "kimi_oauth_host") == "https://auth.kimi.ai");
    CHECK(FileTokenStore(state.string()).load("claude")->account_id == "refreshed-claude");
    FileTokenStore(f.custom.string()).save("openai", token("new-openai"));
    f.select(f.custom);
    CHECK(f.manager.get_config_dir() == state.string());
    CHECK(FileTokenStore(state.string()).load("claude")->account_id == "refreshed-claude");
    CHECK(FileTokenStore(state.string()).load("openai")->account_id == "new-openai");
}

TEST_CASE("Replacing exported credentials waits for an active refresh before selecting the new login", "[config-overlay][oauth]") {
    Fixture f;
    auto expired = token("old-work");
    expired.expires_at = 1;
    FileTokenStore(f.custom.string()).save("claude", expired);
    const auto path = f.bundle();
    f.select(path);
    const fs::path state = f.manager.get_config_dir();
    class BlockingRefresh : public RefreshFlow {
    public:
        std::promise<void> started, resume;
        OAuthToken refresh(std::string_view refresh_token) override {
            started.set_value();
            resume.get_future().wait();
            return RefreshFlow::refresh(refresh_token);
        }
    };
    auto flow = std::make_shared<BlockingRefresh>();
    auto source = credential(f.manager.get_config_source(), flow);
    FileTokenStore(f.custom.string()).save("claude", token("replacement"));
    f.bundle();
    auto refreshing = std::async(std::launch::async, [&] { return source->get_auth(); });
    flow->started.get_future().wait();
    auto availability = std::async(std::launch::async, [&] { return source->availability(); });
    const auto availability_status = availability.wait_for(std::chrono::milliseconds(100));
    CHECK(availability_status == std::future_status::ready);
    auto replacing = std::async(std::launch::async, [&] {
        return core::config::ConfigSource::open(path, f.base);
    });
    CHECK(replacing.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
    flow->resume.set_value();
    CHECK(availability.get() == core::auth::CredentialAvailability::Unknown);
    CHECK(refreshing.get().headers.at("Authorization") == "Bearer rotated");
    CHECK(replacing.get()->state_directory() == state);
    REQUIRE(FileTokenStore(state.string()).load("claude"));
    CHECK(FileTokenStore(state.string()).load("claude")->account_id == "replacement");
    CHECK(source->get_auth().headers.at("Authorization") == "Bearer replacement-access");
}

TEST_CASE("Failed credential replacement preserves settings and cannot mix accounts", "[config-overlay][oauth]") {
    Fixture f;
    FileTokenStore(f.custom.string()).save("kimi", token("old"));
    write(f.custom / "kimi_device_id", "11111111-1111-4111-8111-111111111111");
    f.select(f.custom);
    const fs::path state = f.manager.get_config_dir();
    write(state / "settings.json", R"({"ui_footer":"hide"})");
    const auto settings = read(state / "settings.json");
    fs::remove(state / "kimi_device_id");
    fs::create_symlink(f.custom / "kimi_device_id", state / "kimi_device_id");
    FileTokenStore(f.custom.string()).save("kimi", token("replacement"));
    const auto supplied_token = read(f.custom / "oauth_kimi.json");
    CHECK_THROWS(f.select(f.custom));
    CHECK_FALSE(FileTokenStore(state.string()).load("kimi"));
    CHECK(read(state / "settings.json") == settings);
    CHECK(read(f.custom / "oauth_kimi.json") == supplied_token);
    fs::remove(state / "kimi_device_id");
    f.select(f.custom);
    CHECK(f.manager.get_config_dir() == state.string());
    REQUIRE(FileTokenStore(state.string()).load("kimi"));
    CHECK(FileTokenStore(state.string()).load("kimi")->account_id == "replacement");
    CHECK(f.manager.get_config().ui_footer == "hide");
}

TEST_CASE("Runtime authentication uses the supplied account and redacted exports inherit API keys", "[config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"providers":{"claude":{"api_key":"personal-key","model":"personal"}}})");
    write(f.custom / "config.json", R"({"providers":{"claude":{"api_key":"withheld-work-key","model":"work"}}})");
    const auto path = f.bundle(false);
    f.select(path);
    CHECK(f.manager.get_config().providers.at("claude").api_key == "personal-key");
    CHECK(f.manager.get_config().providers.at("claude").model == "work");
    FileTokenStore(f.base.string()).save("claude", token("personal"));
    FileTokenStore(f.manager.get_config_dir()).save("claude", token("work"));
    auto auth = core::auth::AuthenticationManager::create_with_defaults(f.manager.get_config_dir());
    core::config::ProviderConfig config;
    config.auth_type = "oauth_claude";
    config.config_source = f.manager.get_config_source();
    const auto source = auth.create_credential_source("claude", config);
    REQUIRE(source);
    CHECK(source->get_auth().headers.at("Authorization") == "Bearer work-access");
    auth.logout("claude", false, f.manager.get_config_source().get());
    CHECK_THROWS(source->get_auth());
    CHECK(FileTokenStore(f.base.string()).load("claude")->account_id == "personal");
}

TEST_CASE("A disappearing config directory fails instead of using another account", "[config-overlay]") {
    Fixture f;
    FileTokenStore(f.base.string()).save("claude", token("personal"));
    f.select(f.custom);
    auto source = credential(f.manager.get_config_source(), std::make_shared<RefreshFlow>());
    fs::remove_all(f.custom);
    CHECK_THROWS(f.manager.load(f.project));
    CHECK_THROWS(source->get_auth());
    CHECK_FALSE(fs::exists(f.custom));
}

TEST_CASE("Two processes initialize the same bundle without resetting each other's state", "[config-overlay]") {
    Fixture f;
    FileTokenStore(f.custom.string()).save("claude", token("work"));
    const auto path = f.bundle();
    const auto child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        try { f.select(path); ::_exit(0); }
        catch (...) { ::_exit(1); }
    }
    f.select(path);
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
    CHECK(FileTokenStore(f.manager.get_config_dir()).load("claude")->account_id == "work");
}

TEST_CASE("Injected configuration sources do not depend on the global manager", "[config-overlay][oauth]") {
    Fixture f;
    FileTokenStore(f.base.string()).save("claude", token("personal"));
    auto config = core::config::ConfigSource::open(f.custom, f.base);
    auto source = credential(config, std::make_shared<RefreshFlow>());
    CHECK(source->get_auth().headers.at("Authorization") == "Bearer personal-access");
    FileTokenStore(config->state_directory().string()).save("claude", token("work"));
    CHECK(source->get_auth().headers.at("Authorization") == "Bearer work-access");
    CHECK(f.manager.get_config_dir() == f.base.string());
    CHECK(FileTokenStore(f.base.string()).load("claude")->account_id == "personal");
}

TEST_CASE("Profile changes retain inherited hooks without applying the same profile twice", "[config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"active_profile":"Daily",
        "hooks":{"stop":[{"command":"echo user"}]},
        "profiles":{"Daily":{"hooks":{"stop":[{"command":"echo profile"}]}}}})");
    write(f.project / ".filo" / "config.json", R"({"hooks":{"stop":[{"command":"echo workspace"}]}})");
    write(f.custom / "config.json", R"({"profiles":{"daily":{"ui_banner":"hide"}}})");
    f.select(f.custom);
    CHECK(f.manager.get_config().hooks.stop.size() == 3);
    CHECK(f.manager.get_config().ui_banner == "hide");
    f.manager.load(f.project);
    CHECK(f.manager.get_config().hooks.stop.size() == 3);
    write(f.custom / "config.json", R"({"profiles":{"daily":{"ui_banner":"hide"}},"hooks":{"stop":[]}})");
    f.manager.load(f.project);
    CHECK(f.manager.get_config().hooks.stop.empty());
}

TEST_CASE("Concurrent instances share refresh state only for the same config", "[config-overlay][oauth]") {
    Fixture f;
    bool inherited = false;
    SECTION("supplied credential") {}
    SECTION("inherited credential across different config folders") { inherited = true; }
    auto expired = token("expired");
    expired.expires_at = 1;
    FileTokenStore((inherited ? f.base : f.custom).string()).save("claude", expired);
    f.select(f.custom);
    const auto log = f.root / "refreshes";
    const auto run = [&] {
        auto flow = std::make_shared<RefreshFlow>();
        flow->log = log;
        auto source = credential(f.manager.get_config_source(), flow);
        return source->get_auth().headers.at("Authorization") == "Bearer rotated";
    };
    const auto child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        try {
            if (inherited) {
                const auto second = f.root / "second";
                fs::create_directories(second);
                f.select(second);
            }
            ::_exit(run() ? 0 : 1);
        } catch (...) { ::_exit(1); }
    }
    CHECK(run());
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
    CHECK(read(log) == (inherited ? "refresh\nrefresh\n" : "refresh\n"));
    if (inherited) CHECK(FileTokenStore(f.base.string()).load("claude")->access_token == "expired-access");
}

TEST_CASE("An empty override preserves the normal configuration resolution", "[config][config-overlay]") {
    Fixture f;
    SECTION("sparse user config") {
        write(f.base / "config.json", R"({"default_provider":"claude","ui_theme":"dark",
            "providers":{"claude":{"model":"personal","stream_usage":true,"thinking_budget":8000}},
            "hooks":{"stop":["user-hook"]},"active_profile":"daily",
            "profiles":{"daily":{"ui_banner":"hide","hooks":{"stop":["profile-hook"]}}}})");
        write(f.base / "auth_defaults.json", R"({"providers":{"claude":{"auth_type":"oauth_claude"}}})");
        write(f.base / "settings.json", R"({"ui_footer":"hide","auto_compact_threshold":"35000"})");
        write(f.base / "mcp_servers.json", R"({"mcp_servers":[{"name":"tools","transport":"http",
            "url":"https://user","headers":{"Authorization":"personal"}}]})");
        write(f.project / ".filo/config.json", R"({"providers":{"claude":{"stream_usage":false,
            "thinking_budget":0}},"hooks":{"stop":["project-hook"]}})");
        write(f.project / ".filo/mcp_servers.json", R"({"mcp_servers":[{"name":"tools",
            "transport":"http","url":"https://workspace"}]})");
        write(f.project / ".filo/settings.json", R"({"ui_theme":"light"})");
        write(f.base / "model_defaults.json", R"({"providers":{"claude":{"model":"saved-model"}}})");
    }
    SECTION("fresh install") { fs::remove_all(f.base); }
    SECTION("malformed user file retains legacy fallback") { write(f.base / "config.json", "{broken"); }
    SECTION("whitespace environment profile is ignored") {
        write(f.base / "config.json", R"({"active_profile":"daily","profiles":{"daily":{"ui_banner":"hide"}}})");
        setenv("FILO_PROFILE", " \t ", 1);
    }
    SECTION("unresolved profile retains legacy fallback") {
        write(f.base / "config.json", R"({"active_profile":"missing"})");
    }
    const bool fresh = !fs::exists(f.base / "config.json");
    f.manager.load(f.project);
    const auto normal = f.manager.get_config();
    const auto active = f.manager.get_active_profile();
    const auto profile_count = f.manager.get_profiles().size();
    const auto settings = f.manager.get_settings_overlay(core::config::SettingsScope::User);
    std::map<std::string, std::string> files;
    for (const auto& file : fs::directory_iterator(f.base)) files[file.path().filename()] = read(file.path());
    if (fresh) fs::remove_all(f.base);
    f.select(f.custom);
    const auto& overlay = f.manager.get_config();
    CHECK(std::tie(overlay.default_provider, overlay.default_model_selection, overlay.default_mode,
        overlay.default_approval_mode, overlay.ui_theme, overlay.ui_banner, overlay.ui_footer,
        overlay.steering_mode, overlay.auto_compact_threshold, overlay.auto_compact_threshold_explicit,
        overlay.tool_output_token_limit, overlay.tool_recovery, overlay.tool_recovery_explicit,
        overlay.strict_tool_schemas, overlay.strict_tool_schemas_explicit, overlay.context_compression)
        == std::tie(normal.default_provider, normal.default_model_selection, normal.default_mode,
        normal.default_approval_mode, normal.ui_theme, normal.ui_banner, normal.ui_footer,
        normal.steering_mode, normal.auto_compact_threshold, normal.auto_compact_threshold_explicit,
        normal.tool_output_token_limit, normal.tool_recovery, normal.tool_recovery_explicit,
        normal.strict_tool_schemas, normal.strict_tool_schemas_explicit, normal.context_compression));
    REQUIRE(overlay.providers.size() == normal.providers.size());
    for (const auto& [name, provider] : normal.providers) {
        const auto& actual = overlay.providers.at(name);
        CHECK(std::tie(actual.api_type, actual.api_key, actual.model, actual.base_url, actual.auth_type,
            actual.reasoning_effort, actual.thinking_budget, actual.stream_usage, actual.wire_api, actual.service_tier)
            == std::tie(provider.api_type, provider.api_key, provider.model, provider.base_url, provider.auth_type,
            provider.reasoning_effort, provider.thinking_budget, provider.stream_usage, provider.wire_api, provider.service_tier));
    }
    REQUIRE(overlay.hooks.stop.size() == normal.hooks.stop.size());
    for (std::size_t i = 0; i < normal.hooks.stop.size(); ++i) CHECK(overlay.hooks.stop[i].command == normal.hooks.stop[i].command);
    REQUIRE(overlay.mcp_servers.size() == normal.mcp_servers.size());
    for (std::size_t i = 0; i < normal.mcp_servers.size(); ++i) {
        CHECK(overlay.mcp_servers[i].url == normal.mcp_servers[i].url);
        CHECK(overlay.mcp_servers[i].headers == normal.mcp_servers[i].headers);
    }
    CHECK(overlay.router.enabled == normal.router.enabled);
    CHECK(overlay.router.default_policy == normal.router.default_policy);
    CHECK(overlay.router.policies.size() == normal.router.policies.size());
    CHECK(overlay.has_router_section == normal.has_router_section);
    CHECK(f.manager.get_active_profile() == active);
    CHECK(f.manager.get_profiles().size() == profile_count);
    CHECK(f.manager.get_settings_overlay(core::config::SettingsScope::User).ui_footer == settings.ui_footer);
    if (fresh) {
        CHECK_FALSE(fs::exists(f.base));
    } else {
        for (const auto& file : fs::directory_iterator(f.base)) {
            REQUIRE(files.contains(file.path().filename()));
            CHECK(read(file.path()) == files.at(file.path().filename()));
        }
    }
    CHECK(fs::is_empty(f.custom));
}

TEST_CASE("Normal token storage keeps its original filesystem behavior", "[config-overlay][FileTokenStore]") {
    Fixture f;
    const auto missing = f.root / "missing";
    FileTokenStore store(missing.string());
    CHECK_NOTHROW(store.clear("claude"));
    CHECK_FALSE(fs::exists(missing));
    store.save("claude", token("personal"));
    REQUIRE(fs::exists(missing / "oauth_claude.json"));
    CHECK_FALSE(fs::exists(missing / "oauth_claude.json.selected"));
    store.clear("claude");
    CHECK(fs::is_empty(missing));
    write(missing / "oauth_claude.json", "{broken");
    CHECK_FALSE(store.load("claude"));
    CHECK_FALSE(fs::exists(missing / "oauth_claude.json.selected"));
}

TEST_CASE("Explicitly selecting the default directory isolates writes without duplicating its layers", "[config][config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"active_profile":"daily","profiles":{"daily":{"ui_theme":"light"}},
        "hooks":{"stop":["once"]}})");
    const auto original = read(f.base / "config.json");
    f.select(f.base);
    REQUIRE(f.manager.get_config_source());
    CHECK(f.manager.get_config().hooks.stop.size() == 1);
    CHECK(f.manager.get_config().ui_theme == "light");
    CHECK(f.manager.get_config_dir() != f.base.string());
    std::string error;
    REQUIRE(f.manager.persist_model_defaults("claude", "manual", "saved", &error));
    CHECK(fs::exists(fs::path(f.manager.get_config_dir()) / "model_defaults.json"));
    CHECK_FALSE(fs::exists(f.base / "model_defaults.json"));
    CHECK(read(f.base / "config.json") == original);
    CHECK(fs::is_empty(f.custom));
}

TEST_CASE("Supplied files apply aliases in layer order and can clear explicit values", "[config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"active_profile":"personal","profiles":{"personal":{"ui_banner":"hide"}},
        "providers":{"local":{"api_type":"llamacpp","model_path":"base.gguf","llamacpp":{"use_mmap":true}}}})");
    write(f.base / "auth_defaults.json", R"({"default_provider":"claude"})");
    write(f.custom / "config.json", R"({"active_profile":"", "providers":{"local":{"llamacpp":{"threads":8}}},
        "hooks":{"stop":["old-hook"]},"mcp_servers":[{"name":"tools","transport":"http","url":"https://tools",
        "oauth_scopes":["read"],"headers":{"Authorization":"old"}}]})");
    write(f.custom / "auth_defaults.json", R"({"default_provider":"","providers":{"local":{"threads":4}}})");
    write(f.custom / "mcp_servers.json", R"({"hooks":{"before_complete":["new-hook"]},
        "mcp_servers":[{"name":"tools","oauth_scopes":"","headers":[]}]})");
    f.select(f.custom);
    const auto& config = f.manager.get_config();
    CHECK(config.default_provider.empty());
    CHECK(f.manager.get_auth_default_provider().empty());
    CHECK(f.manager.get_active_profile().empty());
    CHECK(config.providers.at("local").local->model_path == "base.gguf");
    CHECK(config.providers.at("local").local->llamacpp->threads == 4);
    CHECK(config.providers.at("local").local->llamacpp->use_mmap == true);
    REQUIRE(config.hooks.stop.size() == 1);
    CHECK(config.hooks.stop[0].command == "new-hook");
    REQUIRE(config.mcp_servers.size() == 1);
    CHECK(config.mcp_servers[0].oauth_scopes.empty());
    CHECK(config.mcp_servers[0].headers.empty());
    write(f.custom / "mcp_servers.json", R"({"mcp_servers":[]})");
    f.manager.load(f.project);
    CHECK(f.manager.get_config().mcp_servers.empty());
}

TEST_CASE("Default loading retains legacy collection and alias semantics", "[config][config-overlay]") {
    Fixture f;
    write(f.base / "config.json", R"({"tools":{"write":{"allowed_commands":["user"]}},
        "tool_policies":{"write":{"trusted_urls":["https://example.test"]}},
        "mcp_servers":[{"name":"duplicate","transport":"stdio","command":"first"},
                       {"name":"duplicate","transport":"stdio","command":"second"}]})");
    f.manager.load(f.project);
    const auto& config = f.manager.get_config();
    CHECK_FALSE(config.tool_policies.at("write").allowed_commands.has_value());
    REQUIRE(config.tool_policies.at("write").trusted_urls.has_value());
    CHECK(config.tool_policies.at("write").trusted_urls->at(0) == "https://example.test");
    REQUIRE(config.mcp_servers.size() == 1);
    CHECK(config.mcp_servers[0].command == "second");
}

TEST_CASE("MCP connection composition selects the configured credential store", "[config-overlay][mcp][oauth]") {
    Fixture f;
    httplib::Server server;
    std::mutex mutex;
    std::string authorization;
    server.Post("/mcp", [&](const auto& request, auto& response) {
        {
            std::lock_guard lock(mutex);
            authorization = request.get_header_value("Authorization");
        }
        const auto body = JsonValue::parse(request.body);
        if (!body) { response.status = 400; return; }
        const auto* id = body->find("id");
        if (!id) { response.status = 202; return; }
        auto result = *JsonValue::parse(R"({"jsonrpc":"2.0","result":{"protocolVersion":"2025-11-25",
            "capabilities":{"tools":{}},"serverInfo":{"name":"test","version":"1"},"tools":[]}})");
        result.set("id", *id);
        response.set_content(result.dump(), "application/json");
    });
    const auto port = server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::jthread listener([&] { server.listen_after_bind(); });
    struct Stop {
        httplib::Server& server;
        ~Stop() { core::mcp::McpConnectionManager::get_instance().shutdown_all(); server.stop(); }
    } stop{server};
    write(f.base / "mcp_servers.json", "{\"mcp_servers\":[{\"name\":\"work\",\"transport\":\"http\","
        "\"auth\":\"oauth\",\"url\":\"http://127.0.0.1:" + std::to_string(port) + "/mcp\"}]}");
    const auto id = core::mcp::mcp_oauth_provider_id("work");
    FileTokenStore(f.base.string()).save(id, token("personal"));
    f.select(f.custom);
    auto& tools = core::tools::ToolManager::get_instance();
    auto& connections = core::mcp::McpConnectionManager::get_instance();
    connections.connect_all(f.manager.get_config(), tools);
    REQUIRE(connections.connected_count() == 1);
    {
        std::lock_guard lock(mutex);
        CHECK(authorization == "Bearer personal-access");
    }
    FileTokenStore(f.manager.get_config_dir()).save(id, token("work"));
    connections.connect_all(f.manager.get_config(), tools);
    REQUIRE(connections.connected_count() == 1);
    {
        std::lock_guard lock(mutex);
        CHECK(authorization == "Bearer work-access");
    }
    CHECK(FileTokenStore(f.base.string()).load(id)->account_id == "personal");
}

TEST_CASE("MCP auth accepts a token loader without knowing configuration directories", "[config-overlay][mcp][oauth]") {
    core::config::McpServerConfig config;
    config.name = "test";
    config.transport = "http";
    config.url = "https://example.test/mcp";
    config.auth = "oauth";
    auto current = std::optional{token("initial")};
    int refreshes = 0;
    auto auth = core::mcp::make_mcp_http_auth(config, {}, [&](bool force_refresh) {
        if (force_refresh) { ++refreshes; current = token("rotated"); }
        return current;
    });
    REQUIRE(auth);
    CHECK(auth->request_headers().at(0).second == "Bearer initial-access");
    CHECK(auth->can_refresh());
    CHECK(auth->refresh_after_unauthorized());
    CHECK(refreshes == 1);
    CHECK(auth->request_headers().at(0).second == "Bearer rotated-access");
    current.reset();
    CHECK(auth->request_headers().empty());
}

TEST_CASE("Overlay profiles track what was actually applied before model defaults", "[config-overlay][profiles]") {
    Fixture f;
    write(f.base / "config.json", R"({"active_profile":"daily","profiles":{
        "daily":{"ui_banner":"hide"},"parent":{"hooks":{"stop":["parent-hook"]}}}})");
    write(f.base / "model_defaults.json", R"({"profiles":{"daily":{"extends":"parent"}}})");
    write(f.custom / "config.json", R"({"profiles":{"daily":{"ui_theme":"light"}}})");
    f.select(f.custom);
    const auto& config = f.manager.get_config();
    CHECK(config.ui_banner == "hide");
    CHECK(config.ui_theme == "light");
    REQUIRE(config.hooks.stop.size() == 1);
    CHECK(config.hooks.stop[0].command == "parent-hook");
}

TEST_CASE("Identical exports at different paths keep independent tokens and settings", "[config-overlay][portability]") {
    Fixture f;
    write(f.custom / "config.json", R"({"ui_theme":"dark"})");
    FileTokenStore(f.custom.string()).save("claude", token("work"));
    const auto first = f.bundle();
    const auto second = f.root / "another/work.json";
    write(second, read(first));
    f.select(first);
    const auto first_source = f.manager.get_config_source();
    const auto first_state = f.manager.get_config_dir();
    FileTokenStore(first_state).save("claude", token("rotated"));
    std::string error;
    REQUIRE(f.manager.persist_managed_setting(core::config::SettingsScope::User,
        core::config::ManagedSettingKey::UiTheme, "light", f.project, &error));
    f.select(second);
    CHECK(f.manager.get_config_dir() != first_state);
    CHECK(f.manager.get_config().ui_theme == "dark");
    CHECK(FileTokenStore(f.manager.get_config_dir()).load("claude")->account_id == "work");
    auto auth = core::auth::AuthenticationManager::create_with_defaults(first_state);
    auth.logout("claude", false, first_source.get());
    CHECK(FileTokenStore(f.manager.get_config_dir()).load("claude")->account_id == "work");
    f.select(first);
    CHECK(f.manager.get_config().ui_theme == "light");
    CHECK_FALSE(FileTokenStore(first_state).load("claude"));
    CHECK(read(first) == read(second));
    CHECK_FALSE(fs::exists(first.string() + ".d"));
    CHECK_FALSE(fs::exists(second.string() + ".d"));
}

TEST_CASE("Canonical input identity keeps shared state across aliases and JSON formatting", "[config-overlay]") {
    Fixture f;
    FileTokenStore(f.custom.string()).save("claude", token("work"));
    f.select(f.custom);
    const auto state = f.manager.get_config_dir();
    FileTokenStore(state).save("claude", token("rotated"));
    const auto alias = f.root / "alias";
    fs::create_directory_symlink(f.custom, alias);
    f.select(alias);
    CHECK(f.manager.get_config_dir() == state);
    CHECK(FileTokenStore(state).load("claude")->account_id == "rotated");
    auto original = JsonValue::load(f.custom / "oauth_claude.json");
    REQUIRE(original);
    std::ranges::reverse(original->members());
    write(f.custom / "oauth_claude.json", original->dump());
    f.select(f.custom);
    CHECK(f.manager.get_config_dir() == state);
    CHECK(FileTokenStore(state).load("claude")->account_id == "rotated");
}

TEST_CASE("Workspace changes stay in per-config state and retain workspace isolation", "[config-overlay]") {
    Fixture f;
    write(f.custom / "settings.json", R"({"ui_banner":"hide"})");
    write(f.project / ".filo/settings.json", R"({"ui_theme":"dark"})");
    const auto project_settings = read(f.project / ".filo/settings.json");
    f.select(f.custom);
    std::string error;
    REQUIRE(f.manager.persist_managed_setting(core::config::SettingsScope::User,
        core::config::ManagedSettingKey::UiFooter, "hide", f.project, &error));
    REQUIRE(f.manager.persist_managed_setting(core::config::SettingsScope::Workspace,
        core::config::ManagedSettingKey::UiTheme, "light", f.project, &error));
    core::config::McpServerConfig server;
    server.name = "workspace-tools";
    server.transport = "stdio";
    server.command = "tools";
    REQUIRE(f.manager.persist_mcp_server(server, core::config::SettingsScope::Workspace, f.project, &error));
    CHECK(f.manager.get_config().ui_theme == "light");
    CHECK(f.manager.get_config().ui_banner == "hide");
    CHECK(f.manager.get_settings_overlay(core::config::SettingsScope::User).ui_banner == "hide");
    CHECK(read(f.project / ".filo/settings.json") == project_settings);
    CHECK_FALSE(fs::exists(f.project / ".filo/mcp_servers.json"));
    CHECK(fs::exists(f.manager.get_mcp_overlay_path(core::config::SettingsScope::Workspace, f.project)));
    const auto other = f.root / "other-project";
    fs::create_directories(other);
    f.manager.load(other);
    CHECK(f.manager.get_config().ui_theme != "light");
    CHECK(f.manager.get_config().mcp_servers.empty());
    f.manager.load(f.project);
    CHECK(f.manager.get_config().ui_theme == "light");
    CHECK(f.manager.get_config().mcp_servers.size() == 1);
    REQUIRE(f.manager.remove_mcp_server("workspace-tools", core::config::SettingsScope::Workspace, f.project, &error));
    CHECK(f.manager.get_config().mcp_servers.empty());
}

TEST_CASE("Read-only supplied folders remain unchanged after refresh and logout", "[config-overlay][oauth]") {
    Fixture f;
    write(f.custom / "config.json", R"({"providers":{"claude":{"model":"work"}}})");
    auto expired = token("work");
    expired.expires_at = 1;
    FileTokenStore(f.custom.string()).save("claude", expired);
    std::map<std::string, std::string> original;
    for (const auto& file : fs::directory_iterator(f.custom)) original[file.path().filename()] = read(file.path());
    fs::permissions(f.custom, fs::perms::owner_read | fs::perms::owner_exec);
    struct Restore {
        fs::path path;
        ~Restore() { fs::permissions(path, fs::perms::owner_all); }
    } restore{f.custom};
    f.select(f.custom);
    auto flow = std::make_shared<RefreshFlow>();
    auto source = credential(f.manager.get_config_source(), flow);
    CHECK(source->get_auth().headers.at("Authorization") == "Bearer rotated");
    auto auth = core::auth::AuthenticationManager::create_with_defaults(f.manager.get_config_dir());
    auth.logout("claude", false, f.manager.get_config_source().get());
    f.manager.load(f.project);
    CHECK_THROWS(source->get_auth());
    std::size_t count = 0;
    for (const auto& file : fs::directory_iterator(f.custom)) {
        ++count;
        REQUIRE(original.contains(file.path().filename()));
        CHECK(read(file.path()) == original.at(file.path().filename()));
    }
    CHECK(count == original.size());
}

TEST_CASE("Config state respects XDG state home and its home-directory fallback", "[config-overlay]") {
    Fixture f;
    Env home{"HOME", (f.root / "home").string()};
    std::optional<std::string> state_override;
    SECTION("unset state home") {}
    SECTION("empty state home") { state_override = ""; }
    SECTION("relative state home is ignored") { state_override = "relative"; }
    Env state{"XDG_STATE_HOME", state_override};
    f.select(f.custom);
    CHECK(fs::path(f.manager.get_config_dir()).parent_path() == f.root / "home/.local/state/filo/configs");
    CHECK_FALSE(fs::exists(f.root / "relative"));
}

TEST_CASE("Config paths preserve literal whitespace and expand an explicit home prefix", "[config-overlay]") {
    Fixture f;
    write(f.custom / "config.json", R"({"ui_theme":"light"})");
    const auto normal = f.bundle();
    const auto spaced = f.root / " work.json ";
    write(f.custom / "config.json", R"({"ui_theme":"dark"})");
    const auto changed = core::portability::export_bundle(
        {.config_dir = f.custom, .memory_file = f.custom / "memory.json", .home = f.root.string()}, {});
    write(spaced, core::portability::serialize_bundle(changed.bundle));
    f.select(spaced);
    CHECK(f.manager.get_config().ui_theme == "dark");
    CHECK(f.manager.get_config_source()->input_path() == spaced);
    Env home{"HOME", f.root.string()};
    f.select("~/work.json");
    CHECK(f.manager.get_config().ui_theme == "light");
    CHECK(f.manager.get_config_source()->input_path() == normal);
}
