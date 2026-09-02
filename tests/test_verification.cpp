#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestSessionContext.hpp"
#include "core/tools/VerificationTool.hpp"
#include "core/verification/Verification.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <ranges>
#include <string>

namespace {

class TempProject {
public:
  TempProject()
      : root_(
            std::filesystem::temp_directory_path() /
            std::format(
                "filo_verification_{}",
                std::chrono::steady_clock::now().time_since_epoch().count())) {
    std::filesystem::create_directories(root_);
  }

  ~TempProject() {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }

  void write(std::string_view relative, std::string_view content) const {
    const auto path = root_ / relative;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    stream << content;
  }

  [[nodiscard]] const std::filesystem::path &root() const noexcept {
    return root_;
  }

private:
  std::filesystem::path root_;
};

[[nodiscard]] const core::verification::Recipe *
find_recipe(const core::verification::DiscoveryResult &discovery,
            std::string_view id) {
  const auto found =
      std::ranges::find(discovery.recipes, id, &core::verification::Recipe::id);
  return found == discovery.recipes.end() ? nullptr : &*found;
}

} // namespace

TEST_CASE("verification catalog selects a minimal fallback quality gate",
          "[verification][catalog]") {
  using core::verification::Catalog;
  using core::verification::Kind;
  using core::verification::Recipe;

  SECTION("required recipes are authoritative") {
    const std::vector<Recipe> recipes{
        Recipe{.id = "build", .kind = Kind::Build},
        Recipe{.id = "policy", .kind = Kind::Lint, .required = true},
        Recipe{.id = "test", .kind = Kind::Test},
    };
    REQUIRE(Catalog::default_quality_gate(recipes) ==
            std::vector<std::string>{"policy"});
  }
  SECTION("workflow replaces separate build and test") {
    const std::vector<Recipe> recipes{
        Recipe{.id = "build", .kind = Kind::Build},
        Recipe{.id = "workflow", .kind = Kind::Workflow},
        Recipe{.id = "test", .kind = Kind::Test},
    };
    REQUIRE(Catalog::default_quality_gate(recipes) ==
            std::vector<std::string>{"workflow"});
  }
  SECTION("build and test are both selected when available") {
    const std::vector<Recipe> recipes{
        Recipe{.id = "build", .kind = Kind::Build},
        Recipe{.id = "lint", .kind = Kind::Lint},
        Recipe{.id = "test", .kind = Kind::Test},
    };
    REQUIRE(Catalog::default_quality_gate(recipes) ==
            std::vector<std::string>{"build", "test"});
  }
}

TEST_CASE("verification catalog discovers explicit and native recipes",
          "[verification][catalog]") {
  TempProject project;
  project.write(
      ".filo/verification.json",
      R"({"version":1,"recipes":[{"id":"quality","name":"Project quality gate","kind":"test","executable":"./scripts/quality","arguments":["--ci"],"working_directory":".","timeout_seconds":900,"required":true}]})");
  project.write(
      "CMakePresets.json",
      R"({"version":6,"configurePresets":[{"name":"debug"}],"buildPresets":[{"name":"debug","configurePreset":"debug"}],"testPresets":[{"name":"debug","configurePreset":"debug"}],"workflowPresets":[{"name":"quality","steps":[{"type":"configure","name":"debug"},{"type":"build","name":"debug"},{"type":"test","name":"debug"}]}]})");
  project.write(
      "package.json",
      R"({"scripts":{"test":"vitest","lint":"eslint .","start":"vite"}})");
  project.write("Cargo.toml",
                "[package]\nname = \"sample\"\nversion = \"0.1.0\"\n");

  const core::verification::Catalog catalog;
  const auto discovery = catalog.discover(project.root());

  const auto *explicit_recipe = find_recipe(discovery, "quality");
  REQUIRE(explicit_recipe != nullptr);
  CHECK(explicit_recipe->required);
  CHECK(explicit_recipe->source == core::verification::Source::ProjectConfig);
  CHECK(explicit_recipe->command.arguments == std::vector<std::string>{"--ci"});

  const auto *cmake_build = find_recipe(discovery, "cmake:build:debug");
  REQUIRE(cmake_build != nullptr);
  CHECK(cmake_build->command.arguments ==
        (std::vector<std::string>{"--build", "--preset", "debug"}));
  REQUIRE(find_recipe(discovery, "cmake:test:debug") != nullptr);
  const auto *workflow = find_recipe(discovery, "cmake:workflow:quality");
  REQUIRE(workflow != nullptr);
  CHECK(workflow->kind == core::verification::Kind::Workflow);
  REQUIRE(find_recipe(discovery, "package:test") != nullptr);
  REQUIRE(find_recipe(discovery, "package:lint") != nullptr);
  CHECK(find_recipe(discovery, "package:start") == nullptr);
  REQUIRE(find_recipe(discovery, "cargo:test") != nullptr);

  const std::string rendered =
      core::verification::Catalog::render_for_prompt(discovery.recipes);
  CHECK_THAT(rendered, Catch::Matchers::ContainsSubstring(
                           "quality | test | project-config"));
  CHECK_THAT(rendered, Catch::Matchers::ContainsSubstring("run_verification"));
}

TEST_CASE(
    "verification receipt parsing rejects forged or inconsistent evidence",
    "[verification][receipt][security]") {
  CHECK_FALSE(core::verification::parse_receipt(
                  R"({"output":"ctest passed","exit_code":0})")
                  .has_value());

  CHECK_FALSE(
      core::verification::parse_receipt(
          R"({"output":"failed","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"r1","recipe_id":"test","kind":"test","source":"agent-proposed","command":"test","working_directory":"/tmp","exit_code":1,"duration_ms":1}})")
          .has_value());

  const auto parsed = core::verification::parse_receipt(
      R"({"output":"ok","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"r2","recipe_id":"quality","kind":"test","source":"project-config","command":"./quality","working_directory":"/workspace","exit_code":0,"duration_ms":12}})");
  REQUIRE(parsed.has_value());
  CHECK(parsed->passed());
  CHECK(parsed->source == core::verification::Source::ProjectConfig);
}

TEST_CASE("invalid authoritative verification configuration fails visibly",
          "[verification][catalog][configuration]") {
  TempProject project;
  project.write(
      ".filo/verification.json",
      R"({"version":1,"recipes":[{"id":"bad id","kind":"test","executable":"ctest"}]})");

  const core::verification::Catalog catalog;
  const auto discovery = catalog.discover(project.root());
  CHECK(discovery.project_config_present);
  CHECK_FALSE(discovery.project_config_valid);
  REQUIRE_FALSE(discovery.warnings.empty());
  CHECK_THAT(core::verification::Catalog::render_warnings(discovery.warnings),
             Catch::Matchers::ContainsSubstring("configuration errors"));
}

TEST_CASE("authoritative verification configuration rejects malformed fields",
          "[verification][catalog][configuration]") {
  TempProject project;
  project.write(
      ".filo/verification.json",
      R"({"version":1,"recipes":[{"id":"quality","kind":"test","executable":"ctest","arguments":"--output-on-failure","required":"yes"}]})");

  const core::verification::Catalog catalog;
  const auto discovery = catalog.discover(project.root());
  CHECK(discovery.project_config_present);
  CHECK_FALSE(discovery.project_config_valid);
  CHECK(discovery.recipes.empty());
  REQUIRE_FALSE(discovery.warnings.empty());
  CHECK_THAT(core::verification::Catalog::render_warnings(discovery.warnings),
             Catch::Matchers::ContainsSubstring(
                 "arguments must be an array of strings"));
}

TEST_CASE("run_verification preserves argv boundaries and emits typed receipts",
          "[verification][tool][security]") {
  TempProject project;
  const auto context =
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = project.root(),
          .enforce = true,
          .version = 1,
      });
  core::tools::VerificationTool tool;

  const std::string failed = tool.execute(
      R"({"kind":"test","executable":"test","arguments":["-e","missing || true"]})",
      context);
  const auto failed_receipt = core::verification::parse_receipt(failed);
  REQUIRE(failed_receipt.has_value());
  CHECK_FALSE(failed_receipt->passed());
  CHECK(failed_receipt->command == "test -e 'missing || true'");

  const std::string passed = tool.execute(
      R"({"kind":"test","executable":"test","arguments":["-d","."]})", context);
  const auto passed_receipt = core::verification::parse_receipt(passed);
  REQUIRE(passed_receipt.has_value());
  CHECK(passed_receipt->passed());
  CHECK(passed_receipt->kind == core::verification::Kind::Test);
  CHECK(passed_receipt->source == core::verification::Source::AgentProposed);

  const std::string rejected = tool.execute(
      "{\"kind\":\"test\",\"executable\":\"test\\ntrue\",\"arguments\":[]}",
      context);
  CHECK_THAT(rejected,
             Catch::Matchers::ContainsSubstring("invalid characters"));

  const std::string masked = tool.execute(
      R"({"kind":"test","executable":"sh","arguments":["-c","false || true"]})",
      context);
  CHECK_THAT(masked, Catch::Matchers::ContainsSubstring(
                         "cannot invoke a shell interpreter"));
}

TEST_CASE("run_verification resolves repository recipes by immutable id",
          "[verification][tool][recipe]") {
  TempProject project;
  project.write(
      ".filo/verification.json",
      R"({"version":1,"recipes":[{"id":"directory-exists","kind":"test","executable":"test","arguments":["-d","."]}]})");
  const auto context =
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = project.root(),
          .enforce = true,
          .version = 1,
      });
  core::tools::VerificationTool tool;

  const std::string result = tool.execute(
      R"({"recipe_id":"directory-exists","executable":"false"})", context);
  const auto receipt = core::verification::parse_receipt(result);
  REQUIRE(receipt.has_value());
  CHECK(receipt->passed());
  CHECK(receipt->recipe_id == "directory-exists");
  CHECK(receipt->source == core::verification::Source::ProjectConfig);
  CHECK(receipt->command == "test -d .");
}
