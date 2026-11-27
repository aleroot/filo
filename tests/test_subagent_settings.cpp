#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestEnvironment.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/config/SubagentOverlay.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/Uuid.hpp"
#include "tui/SubagentSettings.hpp"
#include "tui/SubagentSettingsPage.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using core::config::ConfigManager;
using core::config::SettingsScope;
using core::config::SubagentModelChoice;
using core::config::SubagentModelChoices;
using Catch::Matchers::ContainsSubstring;

namespace {

void write(const fs::path& path, std::string_view text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}

std::string read(const fs::path& path) {
    return core::utils::read_file(path).value_or("");
}

/// Unsets a variable for the enclosing scope and restores it afterwards.
class ScopedUnsetEnv {
public:
    explicit ScopedUnsetEnv(std::string name) : name_(std::move(name)) {
        if (const char* value = std::getenv(name_.c_str())) previous_ = value;
        ::unsetenv(name_.c_str());
    }
    ~ScopedUnsetEnv() {
        if (previous_) ::setenv(name_.c_str(), previous_->c_str(), 1);
    }
    ScopedUnsetEnv(const ScopedUnsetEnv&) = delete;
    ScopedUnsetEnv& operator=(const ScopedUnsetEnv&) = delete;

private:
    std::string name_;
    std::optional<std::string> previous_;
};

/// Destroyed last (declared first), after the environment is restored, so
/// the singleton is left loaded from the real environment again.
struct ReloadConfigOnExit {
    fs::path cwd = fs::current_path();
    ~ReloadConfigOnExit() {
        auto& manager = ConfigManager::get_instance();
        manager.set_config_override(std::nullopt);
        manager.load(cwd);
    }
};

/// A config home, state home and project of its own, so nothing here touches
/// the user's configuration.
struct ConfigFixture {
    ReloadConfigOnExit reload_on_exit;
    fs::path root = fs::canonical(fs::temp_directory_path())
        / ("filo_subagents_" + core::utils::random_uuid_v4());
    fs::path base = root / "xdg" / "filo";
    fs::path project = root / "project";
    test_support::ScopedEnvVar xdg{"XDG_CONFIG_HOME", (root / "xdg").string()};
    test_support::ScopedEnvVar state{"XDG_STATE_HOME", (root / "state").string()};
    ScopedUnsetEnv profile{"FILO_PROFILE"};
    ConfigManager& manager = ConfigManager::get_instance();

    ConfigFixture() {
        manager.set_config_override(std::nullopt);
        fs::create_directories(base);
        fs::create_directories(project);
        manager.load(project);
    }
    ~ConfigFixture() {
        manager.set_config_override(std::nullopt);
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
    [[nodiscard]] const core::config::SubagentConfig& review() const {
        return manager.get_config().subagents.at("code-review");
    }
};

class FakeStore final : public tui::SubagentModelStore {
public:
    core::config::AppConfig app;
    SubagentModelChoices user;
    int saves = 0;

    [[nodiscard]] core::config::AppConfig config() const override { return app; }
    [[nodiscard]] std::expected<SubagentModelChoices, std::string>
    scoped(SettingsScope scope) const override {
        return scope == SettingsScope::User ? user : SubagentModelChoices{};
    }
    bool save(SettingsScope, std::string_view name,
              const std::optional<SubagentModelChoice>& choice, std::string*) override {
        ++saves;
        if (choice) {
            user[std::string(name)] = *choice;
            app.subagents[std::string(name)].provider = choice->provider;
            app.subagents[std::string(name)].model = choice->model;
        } else {
            user.erase(std::string(name));
            app.subagents[std::string(name)].provider.clear();
            app.subagents[std::string(name)].model.clear();
        }
        return true;
    }
};

} // namespace

TEST_CASE("subagents.json sets one subagent's model and keeps everything else",
          "[config][subagents]") {
    const fs::path root = fs::temp_directory_path() / ("filo_overlay_" + core::utils::random_uuid_v4());
    const fs::path file = root / "subagents.json";
    write(file, R"({"note":"kept","subagents":{"code-review":{"max_steps":4},"explore":{"model":"m"}}})");

    REQUIRE(core::config::write_subagent_model(file, "code-review",
                                               SubagentModelChoice{"openai", "gpt-6.1-terra"}));
    auto read_back = core::config::read_subagent_overlay(file);
    REQUIRE(read_back.has_value());
    CHECK(read_back->at("code-review") == SubagentModelChoice{"openai", "gpt-6.1-terra"});
    CHECK(read_back->at("explore") == SubagentModelChoice{"", "m"});
    CHECK_THAT(read(file), ContainsSubstring(R"("max_steps": 4)"));
    CHECK_THAT(read(file), ContainsSubstring(R"("note": "kept")"));

    SECTION("clearing keeps the entry's other fields") {
        REQUIRE(core::config::write_subagent_model(file, "code-review", std::nullopt));
        CHECK_FALSE(core::config::read_subagent_overlay(file)->contains("code-review"));
        CHECK_THAT(read(file), ContainsSubstring(R"("max_steps": 4)"));
    }
    SECTION("a file left empty is removed") {
        write(file, R"({"subagents":{"code-review":{"provider":"a","model":"b"}}})");
        REQUIRE(core::config::write_subagent_model(file, "code-review", std::nullopt));
        CHECK_FALSE(fs::exists(file));
        CHECK(core::config::read_subagent_overlay(file)->empty());
    }
    SECTION("a malformed file is reported and left alone") {
        write(file, "{not json");
        std::string error;
        CHECK_FALSE(core::config::write_subagent_model(file, "code-review",
                                                       SubagentModelChoice{"a", "b"}, &error));
        CHECK_FALSE(error.empty());
        CHECK(read(file) == "{not json");
        CHECK_FALSE(core::config::read_subagent_overlay(file).has_value());
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
}

TEST_CASE("A saved subagent model applies at once, and the workspace wins over the user",
          "[config][subagents]") {
    ConfigFixture f;
    const auto builtin_steps = f.review().max_steps;

    REQUIRE(f.manager.persist_subagent_model(SettingsScope::User, "code-review",
                                             SubagentModelChoice{"openai", "gpt-6.1-terra"},
                                             f.project));
    CHECK(fs::exists(f.base / "subagents.json"));
    CHECK(f.review().provider == "openai");
    CHECK(f.review().model == "gpt-6.1-terra");
    CHECK(f.review().max_steps == builtin_steps);

    REQUIRE(f.manager.persist_subagent_model(SettingsScope::Workspace, "code-review",
                                             SubagentModelChoice{"claude", "claude-opus-5-5"},
                                             f.project));
    CHECK(fs::exists(f.project / ".filo" / "subagents.json"));
    CHECK(f.review().provider == "claude");

    REQUIRE(f.manager.persist_subagent_model(SettingsScope::Workspace, "code-review",
                                             std::nullopt, f.project));
    CHECK(f.review().provider == "openai");
    CHECK(f.review().model == "gpt-6.1-terra");
}

TEST_CASE("Under --config, subagent models are saved in that config's own state",
          "[config][subagents][config-overlay]") {
    ConfigFixture f;
    const fs::path source = f.root / "work-config";
    write(source / "config.json", R"({"default_provider":"claude"})");
    f.manager.set_config_override(source);
    f.manager.load(f.project);

    REQUIRE(f.manager.persist_subagent_model(SettingsScope::User, "code-review",
                                             SubagentModelChoice{"openai", "gpt-6.1-terra"},
                                             f.project));
    const fs::path saved = f.manager.get_subagent_overlay_path(SettingsScope::User, f.project);
    CHECK(saved.string().starts_with((f.root / "state").string()));
    CHECK(fs::exists(saved));
    CHECK_FALSE(fs::exists(f.base / "subagents.json"));
    CHECK_FALSE(fs::exists(source / "subagents.json"));
    CHECK(f.review().model == "gpt-6.1-terra");

    REQUIRE(f.manager.persist_subagent_model(SettingsScope::Workspace, "code-review",
                                             SubagentModelChoice{"claude", "claude-opus-5-5"},
                                             f.project));
    const fs::path workspace_saved =
        f.manager.get_subagent_overlay_path(SettingsScope::Workspace, f.project);
    CHECK(workspace_saved.string().starts_with((f.root / "state").string()));
    CHECK_FALSE(fs::exists(f.project / ".filo" / "subagents.json"));
    CHECK(f.review().model == "claude-opus-5-5");

    f.manager.set_config_override(std::nullopt);
    f.manager.load(f.project);
    CHECK(f.review().model.empty());
}

TEST_CASE("The Agents page lists built-ins first and marks what the scope sets",
          "[tui][settings][subagents]") {
    core::config::AppConfig config;
    config.subagents["general"].description = "General.";
    config.subagents["explore"].model = "gpt-6-luna";
    config.subagents["explore"].provider = "openai";
    config.subagents["code-review"].model = "gpt-6.1-terra";
    config.subagents["zeta"].description = "Custom.";
    config.subagents["alpha"];
    const SubagentModelChoices scoped{{"explore", {"openai", "gpt-6-luna"}}};

    const auto rows = tui::subagent_settings_rows(config, scoped);
    REQUIRE(rows.size() == 6);
    CHECK(rows[0].name == "general");
    CHECK(rows[1].name == "explore");
    CHECK(rows[2].name == "code-review");
    CHECK(rows[3].name == "reader");
    CHECK(rows[4].name == "alpha");
    CHECK(rows[5].name == "zeta");

    CHECK(rows[0].value == "Conversation's model");
    CHECK(rows[0].inherited);
    CHECK(rows[1].value == "openai · gpt-6-luna");
    CHECK_FALSE(rows[1].inherited);
    CHECK(rows[2].value == "gpt-6.1-terra");
    CHECK_THAT(rows[2].description, ContainsSubstring("/review"));
    CHECK_THAT(rows[3].description, ContainsSubstring("read tool"));
    CHECK(rows[3].value == "Off");
    CHECK(tui::describe_subagent_choice({"openai", ""}) == "openai · its default model");
}

TEST_CASE("The Agents page turns keys into picks, clears and notices",
          "[tui][settings][subagents]") {
    FakeStore store;
    store.app.subagents["code-review"].model = "gpt-6.2";
    int reloads = 0;
    tui::SubagentSettingsPage page(store, [&reloads] { ++reloads; });
    page.open();
    using Kind = tui::SubagentSettingsPage::Action::Kind;
    const auto scope = SettingsScope::User;

    CHECK_FALSE(page.handle(ftxui::Event::Tab, scope).has_value());
    REQUIRE(page.handle(ftxui::Event::Character('3'), scope)->kind == Kind::Notice);
    CHECK(page.rows(scope)[static_cast<std::size_t>(page.selected())].name == "code-review");

    const auto choose = page.handle(ftxui::Event::Return, scope);
    REQUIRE(choose.has_value());
    CHECK(choose->kind == Kind::Choose);
    CHECK(choose->subagent == "code-review");
    CHECK(choose->current == SubagentModelChoice{"", "gpt-6.2"});

    const auto inherited = page.handle(ftxui::Event::Backspace, scope);
    REQUIRE(inherited.has_value());
    CHECK(inherited->kind == Kind::Notice);
    CHECK_THAT(inherited->message, ContainsSubstring("already inheriting"));
    CHECK(store.saves == 0);

    const auto status = page.save(scope, "code-review", SubagentModelChoice{"openai", "gpt-6.1-terra"});
    CHECK_THAT(status, ContainsSubstring("openai · gpt-6.1-terra"));
    CHECK(reloads == 1);
    CHECK(page.rows(scope)[2].value == "openai · gpt-6.1-terra");
    CHECK_FALSE(page.rows(scope)[2].inherited);

    const auto clear = page.handle(ftxui::Event::Backspace, scope);
    REQUIRE(clear.has_value());
    CHECK(clear->kind == Kind::Clear);
    CHECK(clear->subagent == "code-review");

    CHECK(page.handle(ftxui::Event::Escape, scope)->kind == Kind::Back);
}

TEST_CASE("The Agents page keeps each scope's read error to itself", "[tui][settings][subagents]") {
    class BrokenUserStore final : public tui::SubagentModelStore {
    public:
        [[nodiscard]] core::config::AppConfig config() const override { return {}; }
        [[nodiscard]] std::expected<SubagentModelChoices, std::string>
        scoped(SettingsScope scope) const override {
            if (scope == SettingsScope::User) return std::unexpected("bad json");
            return SubagentModelChoices{};
        }
        bool save(SettingsScope, std::string_view, const std::optional<SubagentModelChoice>&,
                  std::string*) override { return true; }
    } store;
    tui::SubagentSettingsPage page(store, {});
    page.open();
    CHECK_THAT(page.note(SettingsScope::User), ContainsSubstring("bad json"));
    CHECK(page.note(SettingsScope::Workspace).empty());
    CHECK_THAT(page.note(SettingsScope::User), ContainsSubstring("bad json"));
}
