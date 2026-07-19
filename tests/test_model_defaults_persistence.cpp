#include <catch2/catch_test_macros.hpp>

#include "core/config/ModelDefaultsPersistence.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace {

class ScopedEnvironmentVariable {
public:
    ScopedEnvironmentVariable(const char* name, const std::string& value)
        : name_(name) {
        if (const char* previous = std::getenv(name); previous != nullptr) {
            previous_ = previous;
        }
        ::setenv(name, value.c_str(), 1);
    }

    ~ScopedEnvironmentVariable() {
        if (previous_) {
            ::setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    std::optional<std::string> previous_;
};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

} // namespace

TEST_CASE("Only the earliest live instance persists model defaults",
          "[config][model][concurrency]") {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto sandbox = std::filesystem::temp_directory_path()
        / ("filo_model_owner_" + std::to_string(nonce));
    std::filesystem::create_directories(sandbox);
    ScopedEnvironmentVariable xdg_config_home("XDG_CONFIG_HOME", sandbox.string());

    auto& manager = core::config::ConfigManager::get_instance();
    std::optional<core::config::ModelDefaultsPersistence> follower;
    {
        core::config::ModelDefaultsPersistence owner(manager);
        follower.emplace(manager);

        CHECK(owner.persist("openai", "manual", "gpt-owner").status
              == core::config::ModelPersistenceStatus::Saved);
        CHECK(follower->persist("claude", "manual", "claude-follower").status
              == core::config::ModelPersistenceStatus::SessionOnly);

        const auto saved = read_file(sandbox / "filo" / "model_defaults.json");
        CHECK(saved.contains("gpt-owner"));
        CHECK_FALSE(saved.contains("claude-follower"));
    }

    CHECK(follower->persist("claude", "manual", "claude-successor").status
          == core::config::ModelPersistenceStatus::Saved);
    CHECK(read_file(sandbox / "filo" / "model_defaults.json")
              .contains("claude-successor"));

    follower.reset();
    std::error_code ignored;
    std::filesystem::remove_all(sandbox, ignored);
}
