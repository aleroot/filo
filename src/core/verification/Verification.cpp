#include "Verification.hpp"

#include "../utils/JsonUtils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <simdjson.h>
#include <unordered_map>
#include <unordered_set>

namespace core::verification {

namespace {

constexpr std::size_t kMaxManifestBytes = 1024 * 1024;
constexpr std::size_t kMaxRecipesPerProvider = 128;
constexpr std::size_t kMaxArgumentsPerRecipe = 128;
constexpr std::size_t kMaxArgumentBytes = 16 * 1024;

[[nodiscard]] std::string ascii_lower(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (const unsigned char ch : value) {
    out.push_back(static_cast<char>(std::tolower(ch)));
  }
  return out;
}

[[nodiscard]] bool valid_identifier(std::string_view value) noexcept {
  if (value.empty() || value.size() > 128)
    return false;
  return std::ranges::all_of(value, [](const unsigned char ch) {
    return std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == ':';
  });
}

[[nodiscard]] std::expected<std::string, std::string>
read_bounded_file(const std::filesystem::path &path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    return std::unexpected(
        std::format("cannot stat {}: {}", path.string(), ec.message()));
  }
  if (size > kMaxManifestBytes) {
    return std::unexpected(std::format("{} exceeds the {} byte manifest limit",
                                       path.string(), kMaxManifestBytes));
  }

  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return std::unexpected(std::format("cannot read {}", path.string()));
  }
  std::string content;
  content.resize(static_cast<std::size_t>(size));
  stream.read(content.data(), static_cast<std::streamsize>(content.size()));
  if (!stream && !stream.eof()) {
    return std::unexpected(
        std::format("failed while reading {}", path.string()));
  }
  return content;
}

[[nodiscard]] std::string host_system_name() {
#if defined(_WIN32)
  return "Windows";
#elif defined(__APPLE__)
  return "Darwin";
#elif defined(__linux__)
  return "Linux";
#else
  return "Unknown";
#endif
}

[[nodiscard]] std::vector<std::string>
string_array(const simdjson::dom::object &object, std::string_view key) {
  std::vector<std::string> values;
  simdjson::dom::array array;
  if (object[key].get(array) != simdjson::SUCCESS)
    return values;
  for (const auto item : array) {
    std::string_view value;
    if (item.get(value) == simdjson::SUCCESS)
      values.emplace_back(value);
  }
  return values;
}

[[nodiscard]] std::expected<std::string, std::string>
optional_string(const simdjson::dom::object &object, std::string_view key) {
  std::string_view value;
  const auto status = object[key].get(value);
  if (status == simdjson::NO_SUCH_FIELD)
    return std::string{};
  if (status != simdjson::SUCCESS)
    return std::unexpected(std::format("{} must be a string", key));
  return std::string(value);
}

[[nodiscard]] std::expected<std::vector<std::string>, std::string>
optional_string_array(const simdjson::dom::object &object,
                      std::string_view key) {
  simdjson::dom::array array;
  const auto status = object[key].get(array);
  if (status == simdjson::NO_SUCH_FIELD)
    return std::vector<std::string>{};
  if (status != simdjson::SUCCESS) {
    return std::unexpected(std::format("{} must be an array of strings", key));
  }

  std::vector<std::string> values;
  for (const auto item : array) {
    std::string_view value;
    if (item.get(value) != simdjson::SUCCESS) {
      return std::unexpected(std::format("{} must contain only strings", key));
    }
    values.emplace_back(value);
  }
  return values;
}

[[nodiscard]] std::expected<int, std::string>
optional_int(const simdjson::dom::object &object, std::string_view key,
             int fallback) {
  int64_t value = 0;
  const auto status = object[key].get(value);
  if (status == simdjson::NO_SUCH_FIELD)
    return fallback;
  if (status != simdjson::SUCCESS || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    return std::unexpected(std::format("{} must be an integer", key));
  }
  return static_cast<int>(value);
}

[[nodiscard]] std::expected<bool, std::string>
optional_bool(const simdjson::dom::object &object, std::string_view key,
              bool fallback) {
  bool value = false;
  const auto status = object[key].get(value);
  if (status == simdjson::NO_SUCH_FIELD)
    return fallback;
  if (status != simdjson::SUCCESS)
    return std::unexpected(std::format("{} must be a boolean", key));
  return value;
}

[[nodiscard]] std::vector<std::string>
string_or_array(const simdjson::dom::object &object, std::string_view key) {
  std::string_view single;
  if (object[key].get(single) == simdjson::SUCCESS) {
    return {std::string(single)};
  }
  return string_array(object, key);
}

[[nodiscard]] bool recipe_is_valid(const Recipe &recipe, std::string &reason) {
  if (!valid_identifier(recipe.id)) {
    reason =
        "recipe id must contain only letters, digits, '.', '_', '-', or ':'";
    return false;
  }
  if (recipe.command.executable.empty()) {
    reason = "recipe executable is empty";
    return false;
  }
  if (recipe.command.arguments.size() > kMaxArgumentsPerRecipe) {
    reason = "recipe has too many arguments";
    return false;
  }
  std::size_t argument_bytes = recipe.command.executable.size();
  for (const auto &argument : recipe.command.arguments)
    argument_bytes += argument.size();
  if (argument_bytes > kMaxArgumentBytes) {
    reason = "recipe command exceeds the argument size limit";
    return false;
  }
  if (recipe.command.timeout_seconds <= 0 ||
      recipe.command.timeout_seconds > 3600) {
    reason = "recipe timeout must be between 1 and 3600 seconds";
    return false;
  }
  return true;
}

class ProjectConfigProvider final : public IRecipeProvider {
public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "project config";
  }

  [[nodiscard]] DiscoveryResult
  discover(const DiscoveryContext &context) const override {
    DiscoveryResult result;
    const auto path = context.project_root / ".filo" / "verification.json";
    if (!std::filesystem::is_regular_file(path))
      return result;
    result.project_config_present = true;

    const auto content = read_bounded_file(path);
    if (!content.has_value()) {
      result.project_config_valid = false;
      result.warnings.push_back(content.error());
      return result;
    }

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(*content).get(document) != simdjson::SUCCESS) {
      result.project_config_valid = false;
      result.warnings.push_back(".filo/verification.json is not valid JSON");
      return result;
    }

    int64_t version = 0;
    core::utils::json::ignore_error(document["version"].get(version));
    if (version != 1) {
      result.project_config_valid = false;
      result.warnings.push_back(
          ".filo/verification.json must declare schema version 1");
      return result;
    }

    simdjson::dom::array recipes;
    if (document["recipes"].get(recipes) != simdjson::SUCCESS) {
      result.project_config_valid = false;
      result.warnings.push_back(
          ".filo/verification.json must contain a recipes array");
      return result;
    }

    std::unordered_set<std::string> recipe_ids;

    for (const auto item : recipes) {
      if (result.recipes.size() >= kMaxRecipesPerProvider) {
        result.warnings.push_back(
            "project verification recipes were truncated at 128");
        break;
      }
      simdjson::dom::object object;
      if (item.get(object) != simdjson::SUCCESS) {
        result.project_config_valid = false;
        result.warnings.push_back("project recipe must be a JSON object");
        continue;
      }

      const auto id = optional_string(object, "id");
      const auto name = optional_string(object, "name");
      const auto kind = optional_string(object, "kind");
      const auto executable = optional_string(object, "executable");
      const auto arguments = optional_string_array(object, "arguments");
      const auto working_directory =
          optional_string(object, "working_directory");
      const auto timeout = optional_int(object, "timeout_seconds", 600);
      const auto required = optional_bool(object, "required", false);
      const std::array field_results{
          id.has_value(),        name.has_value(),
          kind.has_value(),      executable.has_value(),
          arguments.has_value(), working_directory.has_value(),
          timeout.has_value(),   required.has_value(),
      };
      if (!std::ranges::all_of(field_results, std::identity{})) {
        result.project_config_valid = false;
        const auto first_error = [&]() -> std::string {
          if (!id)
            return id.error();
          if (!name)
            return name.error();
          if (!kind)
            return kind.error();
          if (!executable)
            return executable.error();
          if (!arguments)
            return arguments.error();
          if (!working_directory)
            return working_directory.error();
          if (!timeout)
            return timeout.error();
          return required.error();
        }();
        result.warnings.push_back(
            std::format("ignored project recipe: {}", first_error));
        continue;
      }

      const auto parsed_kind = kind_from_string(*kind);
      if (!parsed_kind.has_value()) {
        result.project_config_valid = false;
        result.warnings.push_back(
            "project recipe has an unknown verification kind");
        continue;
      }

      Recipe recipe{
          .id = *id,
          .display_name = *name,
          .kind = *parsed_kind,
          .source = Source::ProjectConfig,
          .command =
              {
                  .executable = *executable,
                  .arguments = *arguments,
                  .working_directory = *working_directory,
                  .timeout_seconds = *timeout,
              },
          .required = *required,
      };
      if (recipe.display_name.empty())
        recipe.display_name = recipe.id;

      std::string reason;
      if (!recipe_is_valid(recipe, reason)) {
        result.project_config_valid = false;
        result.warnings.push_back(
            std::format("ignored project recipe '{}': {}", recipe.id, reason));
        continue;
      }
      if (!recipe_ids.insert(recipe.id).second) {
        result.project_config_valid = false;
        result.warnings.push_back(
            std::format("ignored duplicate project recipe id '{}'", recipe.id));
        continue;
      }
      result.recipes.push_back(std::move(recipe));
    }
    return result;
  }
};

struct ConfigurePreset {
  std::vector<std::string> inherits;
  std::optional<bool> host_condition;
};

[[nodiscard]] std::optional<bool>
evaluate_host_condition(simdjson::dom::element condition,
                        std::string_view host_system) {
  bool boolean = false;
  if (condition.get(boolean) == simdjson::SUCCESS)
    return boolean;

  simdjson::dom::object object;
  if (condition.get(object) != simdjson::SUCCESS)
    return std::nullopt;
  const std::string type = core::utils::json::string_field(object, "type");
  if (type == "const") {
    return core::utils::json::bool_field(object, "value", false);
  }
  if (type != "equals" && type != "notEquals")
    return std::nullopt;

  const std::string lhs = core::utils::json::string_field(object, "lhs");
  const std::string rhs = core::utils::json::string_field(object, "rhs");
  bool equal = false;
  if (lhs == "${hostSystemName}")
    equal = rhs == host_system;
  else if (rhs == "${hostSystemName}")
    equal = lhs == host_system;
  else
    equal = lhs == rhs;
  return type == "equals" ? equal : !equal;
}

[[nodiscard]] bool configure_is_available(
    std::string_view name,
    const std::unordered_map<std::string, ConfigurePreset> &presets,
    std::unordered_set<std::string> &visiting) {
  const auto it = presets.find(std::string(name));
  if (it == presets.end())
    return true;
  if (!visiting.insert(it->first).second)
    return false;

  const auto erase_guard = [&] { visiting.erase(it->first); };
  if (it->second.host_condition.has_value() && !*it->second.host_condition) {
    erase_guard();
    return false;
  }
  for (const auto &parent : it->second.inherits) {
    if (!configure_is_available(parent, presets, visiting)) {
      erase_guard();
      return false;
    }
  }
  erase_guard();
  return true;
}

class CMakePresetProvider final : public IRecipeProvider {
public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "CMake presets";
  }

  [[nodiscard]] DiscoveryResult
  discover(const DiscoveryContext &context) const override {
    DiscoveryResult result;
    const auto path = context.project_root / "CMakePresets.json";
    if (!std::filesystem::is_regular_file(path))
      return result;

    const auto content = read_bounded_file(path);
    if (!content.has_value()) {
      result.warnings.push_back(content.error());
      return result;
    }
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(*content).get(document) != simdjson::SUCCESS) {
      result.warnings.push_back("CMakePresets.json is not valid JSON");
      return result;
    }

    std::unordered_map<std::string, ConfigurePreset> configure_presets;
    simdjson::dom::array configure_array;
    if (document["configurePresets"].get(configure_array) ==
        simdjson::SUCCESS) {
      for (const auto item : configure_array) {
        simdjson::dom::object object;
        if (item.get(object) != simdjson::SUCCESS)
          continue;
        const std::string name =
            core::utils::json::string_field(object, "name");
        if (name.empty())
          continue;
        ConfigurePreset preset;
        preset.inherits = string_or_array(object, "inherits");
        simdjson::dom::element condition;
        if (object["condition"].get(condition) == simdjson::SUCCESS) {
          preset.host_condition =
              evaluate_host_condition(condition, context.host_system);
        }
        configure_presets.emplace(name, std::move(preset));
      }
    }

    const auto append_presets = [&](std::string_view array_name, Kind kind,
                                    std::string_view id_prefix,
                                    std::string_view executable,
                                    std::vector<std::string> command_prefix) {
      simdjson::dom::array presets;
      if (document[array_name].get(presets) != simdjson::SUCCESS)
        return;
      for (const auto item : presets) {
        if (result.recipes.size() >= kMaxRecipesPerProvider)
          break;
        simdjson::dom::object object;
        if (item.get(object) != simdjson::SUCCESS)
          continue;
        if (core::utils::json::bool_field(object, "hidden", false))
          continue;

        const std::string name =
            core::utils::json::string_field(object, "name");
        const std::string configure =
            core::utils::json::string_field(object, "configurePreset");
        if (name.empty())
          continue;

        std::unordered_set<std::string> visiting;
        if (!configure.empty() &&
            !configure_is_available(configure, configure_presets, visiting)) {
          continue;
        }
        simdjson::dom::element condition;
        if (object["condition"].get(condition) == simdjson::SUCCESS) {
          const auto available =
              evaluate_host_condition(condition, context.host_system);
          if (available.has_value() && !*available)
            continue;
        }

        auto arguments = command_prefix;
        arguments.push_back(name);
        result.recipes.push_back(Recipe{
            .id = std::format("cmake:{}:{}", id_prefix, name),
            .display_name = std::format("CMake {} preset {}", id_prefix, name),
            .kind = kind,
            .source = Source::CMakePreset,
            .command =
                {
                    .executable = std::string(executable),
                    .arguments = std::move(arguments),
                },
        });
      }
    };

    append_presets("buildPresets", Kind::Build, "build", "cmake",
                   {"--build", "--preset"});
    append_presets("testPresets", Kind::Test, "test", "ctest", {"--preset"});

    simdjson::dom::array workflows;
    if (document["workflowPresets"].get(workflows) == simdjson::SUCCESS) {
      for (const auto item : workflows) {
        if (result.recipes.size() >= kMaxRecipesPerProvider)
          break;
        simdjson::dom::object object;
        if (item.get(object) != simdjson::SUCCESS ||
            core::utils::json::bool_field(object, "hidden", false)) {
          continue;
        }
        const std::string name =
            core::utils::json::string_field(object, "name");
        if (name.empty())
          continue;

        std::string configure;
        simdjson::dom::array steps;
        if (object["steps"].get(steps) == simdjson::SUCCESS) {
          for (const auto step_element : steps) {
            simdjson::dom::object step;
            if (step_element.get(step) != simdjson::SUCCESS)
              continue;
            if (core::utils::json::string_field(step, "type") == "configure") {
              configure = core::utils::json::string_field(step, "name");
              break;
            }
          }
        }
        std::unordered_set<std::string> visiting;
        if (!configure.empty() &&
            !configure_is_available(configure, configure_presets, visiting)) {
          continue;
        }
        result.recipes.push_back(Recipe{
            .id = std::format("cmake:workflow:{}", name),
            .display_name = std::format("CMake workflow preset {}", name),
            .kind = Kind::Workflow,
            .source = Source::CMakePreset,
            .command =
                {
                    .executable = "cmake",
                    .arguments = {"--workflow", "--preset", name},
                },
        });
      }
    }
    return result;
  }
};

[[nodiscard]] std::optional<Kind> package_script_kind(std::string_view name) {
  const std::string lower = ascii_lower(name);
  const auto family = [&](std::string_view prefix) {
    return lower == prefix || lower.starts_with(std::string(prefix) + ":");
  };
  if (family("test") || family("verify"))
    return Kind::Test;
  if (family("lint"))
    return Kind::Lint;
  if (family("typecheck") || family("type-check") || family("check-types")) {
    return Kind::TypeCheck;
  }
  if (family("build") || family("compile"))
    return Kind::Build;
  if (family("format:check") || family("format-check"))
    return Kind::Format;
  return std::nullopt;
}

class PackageScriptProvider final : public IRecipeProvider {
public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "package scripts";
  }

  [[nodiscard]] DiscoveryResult
  discover(const DiscoveryContext &context) const override {
    DiscoveryResult result;
    const auto path = context.project_root / "package.json";
    if (!std::filesystem::is_regular_file(path))
      return result;
    const auto content = read_bounded_file(path);
    if (!content.has_value()) {
      result.warnings.push_back(content.error());
      return result;
    }

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(*content).get(document) != simdjson::SUCCESS) {
      result.warnings.push_back("package.json is not valid JSON");
      return result;
    }
    simdjson::dom::object scripts;
    if (document["scripts"].get(scripts) != simdjson::SUCCESS)
      return result;

    std::string manager = "npm";
    if (std::filesystem::exists(context.project_root / "pnpm-lock.yaml"))
      manager = "pnpm";
    else if (std::filesystem::exists(context.project_root / "yarn.lock"))
      manager = "yarn";
    else if (std::filesystem::exists(context.project_root / "bun.lock") ||
             std::filesystem::exists(context.project_root / "bun.lockb"))
      manager = "bun";

    for (const auto field : scripts) {
      if (result.recipes.size() >= kMaxRecipesPerProvider)
        break;
      const std::string name(field.key);
      const auto kind = package_script_kind(name);
      if (!kind.has_value() || !valid_identifier(name))
        continue;
      result.recipes.push_back(Recipe{
          .id = std::format("package:{}", name),
          .display_name = std::format("package script {}", name),
          .kind = *kind,
          .source = Source::PackageScript,
          .command =
              {
                  .executable = manager,
                  .arguments = {"run", name},
              },
      });
    }
    return result;
  }
};

class EcosystemProvider final : public IRecipeProvider {
public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "ecosystem defaults";
  }

  [[nodiscard]] DiscoveryResult
  discover(const DiscoveryContext &context) const override {
    DiscoveryResult result;
    const auto add = [&](Recipe recipe) {
      result.recipes.push_back(std::move(recipe));
    };
    const auto add_default = [&](std::string id, std::string display_name,
                                 Kind kind, std::string executable,
                                 std::vector<std::string> arguments) {
      add(Recipe{
          .id = std::move(id),
          .display_name = std::move(display_name),
          .kind = kind,
          .source = Source::EcosystemDefault,
          .command =
              {
                  .executable = std::move(executable),
                  .arguments = std::move(arguments),
                  .working_directory = {},
                  .timeout_seconds = 600,
              },
          .required = false,
      });
    };
    const auto exists = [&](std::string_view name) {
      return std::filesystem::is_regular_file(context.project_root / name);
    };

    if (exists("Cargo.toml")) {
      add_default("cargo:test", "Cargo tests", Kind::Test, "cargo", {"test"});
      add_default("cargo:check", "Cargo check", Kind::TypeCheck, "cargo",
                  {"check"});
    }
    if (exists("go.mod")) {
      add_default("go:test", "Go tests", Kind::Test, "go", {"test", "./..."});
      add_default("go:vet", "Go vet", Kind::Lint, "go", {"vet", "./..."});
    }
    if (exists("Package.swift")) {
      add_default("swift:test", "Swift package tests", Kind::Test, "swift",
                  {"test"});
    }
    if (exists("pom.xml")) {
      add_default("maven:test", "Maven tests", Kind::Test, "mvn", {"test"});
      add_default("maven:verify", "Maven verification lifecycle", Kind::Test,
                  "mvn", {"verify"});
    }
    if (exists("gradlew")) {
      add_default("gradle:test", "Gradle tests", Kind::Test, "./gradlew",
                  {"test"});
      add_default("gradle:check", "Gradle checks", Kind::Lint, "./gradlew",
                  {"check"});
    }
    return result;
  }
};

[[nodiscard]] std::string prompt_escape(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (const char ch : value) {
    if (ch == '\n' || ch == '\r')
      out.push_back(' ');
    else
      out.push_back(ch);
  }
  return out;
}

} // namespace

std::string_view to_string(Kind kind) noexcept {
  switch (kind) {
  case Kind::Build:
    return "build";
  case Kind::Test:
    return "test";
  case Kind::Lint:
    return "lint";
  case Kind::TypeCheck:
    return "typecheck";
  case Kind::Format:
    return "format";
  case Kind::Workflow:
    return "workflow";
  case Kind::Custom:
    return "custom";
  }
  return "custom";
}

std::string_view to_string(Source source) noexcept {
  switch (source) {
  case Source::ProjectConfig:
    return "project-config";
  case Source::CMakePreset:
    return "cmake-preset";
  case Source::PackageScript:
    return "package-script";
  case Source::EcosystemDefault:
    return "ecosystem-default";
  case Source::AgentProposed:
    return "agent-proposed";
  }
  return "agent-proposed";
}

std::optional<Kind> kind_from_string(std::string_view value) noexcept {
  const std::string lower = ascii_lower(value);
  if (lower == "build")
    return Kind::Build;
  if (lower == "test")
    return Kind::Test;
  if (lower == "lint")
    return Kind::Lint;
  if (lower == "typecheck" || lower == "type-check")
    return Kind::TypeCheck;
  if (lower == "format")
    return Kind::Format;
  if (lower == "workflow")
    return Kind::Workflow;
  if (lower == "custom")
    return Kind::Custom;
  return std::nullopt;
}

std::optional<Source> source_from_string(std::string_view value) noexcept {
  const std::string lower = ascii_lower(value);
  if (lower == "project-config")
    return Source::ProjectConfig;
  if (lower == "cmake-preset")
    return Source::CMakePreset;
  if (lower == "package-script")
    return Source::PackageScript;
  if (lower == "ecosystem-default")
    return Source::EcosystemDefault;
  if (lower == "agent-proposed")
    return Source::AgentProposed;
  return std::nullopt;
}

Catalog::Catalog()
    : Catalog({
          std::make_shared<ProjectConfigProvider>(),
          std::make_shared<CMakePresetProvider>(),
          std::make_shared<PackageScriptProvider>(),
          std::make_shared<EcosystemProvider>(),
      }) {}

Catalog::Catalog(std::vector<std::shared_ptr<const IRecipeProvider>> providers)
    : providers_(std::move(providers)) {}

DiscoveryResult
Catalog::discover(const std::filesystem::path &project_root) const noexcept {
  DiscoveryResult combined;
  if (project_root.empty())
    return combined;

  const DiscoveryContext context{
      .project_root = project_root,
      .host_system = host_system_name(),
  };
  std::unordered_set<std::string> recipe_ids;
  for (const auto &provider : providers_) {
    if (!provider)
      continue;
    try {
      auto discovered = provider->discover(context);
      if (discovered.project_config_present) {
        combined.project_config_present = true;
        combined.project_config_valid =
            combined.project_config_valid && discovered.project_config_valid;
      }
      for (auto &warning : discovered.warnings) {
        combined.warnings.push_back(
            std::format("{}: {}", provider->name(), warning));
      }
      for (auto &recipe : discovered.recipes) {
        if (combined.recipes.size() >= 256) {
          combined.warnings.push_back(
              "verification catalog was truncated at 256 recipes");
          break;
        }
        if (!recipe_ids.insert(recipe.id).second) {
          combined.warnings.push_back(std::format(
              "duplicate verification recipe id '{}' was ignored", recipe.id));
          continue;
        }
        combined.recipes.push_back(std::move(recipe));
      }
    } catch (const std::exception &error) {
      combined.warnings.push_back(std::format("{} discovery failed: {}",
                                              provider->name(), error.what()));
    } catch (...) {
      combined.warnings.push_back(
          std::format("{} discovery failed", provider->name()));
    }
  }
  std::ranges::sort(combined.recipes, {}, &Recipe::id);
  return combined;
}

std::string Catalog::render_for_prompt(std::span<const Recipe> recipes) {
  std::string rendered =
      "\n[Repository verification recipes]\n"
      "Use `run_verification` with a recipe_id whenever one covers the change. "
      "These commands come from repository-owned configuration or native "
      "project metadata. "
      "Unless the project marks explicit required recipes, AUTO requires one "
      "successful build "
      "and one successful test receipt when both kinds are available.\n";
  if (recipes.empty()) {
    rendered +=
        "No native recipe was discovered. Use run_verification with a typed "
        "custom "
        "executable and arguments; never hide failures with shell operators.\n";
  } else {
    constexpr std::size_t kPromptRecipeLimit = 32;
    const std::size_t count = std::min(recipes.size(), kPromptRecipeLimit);
    for (std::size_t i = 0; i < count; ++i) {
      const auto &recipe = recipes[i];
      rendered += std::format("- {} | {} | {} | {}{}\n", recipe.id,
                              to_string(recipe.kind), to_string(recipe.source),
                              prompt_escape(display_command(recipe.command)),
                              recipe.required ? " | required" : "");
    }
    if (recipes.size() > count) {
      rendered += std::format("- ... {} more recipes omitted\n",
                              recipes.size() - count);
    }
  }
  rendered += "[/Repository verification recipes]";
  return rendered;
}

std::string Catalog::render_warnings(std::span<const std::string> warnings) {
  if (warnings.empty())
    return {};
  std::string rendered = "\n[Verification discovery warnings]\n"
                         "Treat these as configuration errors; do not silently "
                         "replace an invalid project gate.\n";
  constexpr std::size_t kWarningLimit = 16;
  const std::size_t count = std::min(warnings.size(), kWarningLimit);
  for (std::size_t i = 0; i < count; ++i) {
    rendered += "- ";
    rendered += prompt_escape(warnings[i]);
    rendered += "\n";
  }
  if (warnings.size() > count) {
    rendered += std::format("- ... {} more warnings omitted\n",
                            warnings.size() - count);
  }
  rendered += "[/Verification discovery warnings]";
  return rendered;
}

std::vector<std::string>
Catalog::default_quality_gate(std::span<const Recipe> recipes) {
  std::vector<std::string> selected;
  for (const Recipe &recipe : recipes) {
    if (recipe.required) {
      selected.push_back(recipe.id);
    }
  }
  if (!selected.empty()) {
    return selected;
  }

  const auto first_of_kind = [&](Kind kind) -> const Recipe * {
    const auto found = std::ranges::find(recipes, kind, &Recipe::kind);
    return found == recipes.end() ? nullptr : &*found;
  };
  if (const Recipe *workflow = first_of_kind(Kind::Workflow)) {
    return {workflow->id};
  }
  if (const Recipe *build = first_of_kind(Kind::Build)) {
    selected.push_back(build->id);
  }
  if (const Recipe *test = first_of_kind(Kind::Test)) {
    selected.push_back(test->id);
  }
  if (selected.empty() && !recipes.empty()) {
    selected.push_back(recipes.front().id);
  }
  return selected;
}

std::string display_command(const CommandSpec &command) {
  const auto quote = [](std::string_view token) {
    if (!token.empty() &&
        std::ranges::all_of(token, [](const unsigned char ch) {
          return std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.' ||
                 ch == '/' || ch == ':' || ch == '=';
        })) {
      return std::string(token);
    }
    std::string out = "'";
    for (const char ch : token)
      out += ch == '\'' ? "'\\''" : std::string(1, ch);
    out += "'";
    return out;
  };

  std::string displayed = quote(command.executable);
  for (const auto &argument : command.arguments) {
    displayed += " ";
    displayed += quote(argument);
  }
  return displayed;
}

std::expected<Receipt, std::string>
parse_receipt(std::string_view tool_result) noexcept {
  try {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(tool_result.data(), tool_result.size()).get(document) !=
        simdjson::SUCCESS) {
      return std::unexpected("verification result is not valid JSON");
    }
    simdjson::dom::object root;
    if (document.get(root) != simdjson::SUCCESS) {
      return std::unexpected("verification result must be an object");
    }
    simdjson::dom::object receipt;
    if (root["verification_receipt"].get(receipt) != simdjson::SUCCESS) {
      return std::unexpected("verification receipt is missing");
    }
    const int version =
        core::utils::json::int_field(receipt, "schema_version", 0);
    if (version != kReceiptSchemaVersion) {
      return std::unexpected("unsupported verification receipt schema");
    }
    const auto kind =
        kind_from_string(core::utils::json::string_field(receipt, "kind"));
    const auto source =
        source_from_string(core::utils::json::string_field(receipt, "source"));
    if (!kind.has_value() || !source.has_value()) {
      return std::unexpected("verification receipt has invalid provenance");
    }
    const auto nested_exit =
        core::utils::json::first_int64_field(receipt, {"exit_code"});
    const auto root_exit =
        core::utils::json::first_int64_field(root, {"exit_code"});
    if (!nested_exit.has_value() || !root_exit.has_value() ||
        *nested_exit != *root_exit) {
      return std::unexpected("verification receipt exit code is inconsistent");
    }
    if (*nested_exit < std::numeric_limits<int>::min() ||
        *nested_exit > std::numeric_limits<int>::max()) {
      return std::unexpected("verification receipt exit code is out of range");
    }

    Receipt parsed{
        .receipt_id = core::utils::json::string_field(receipt, "receipt_id"),
        .recipe_id = core::utils::json::string_field(receipt, "recipe_id"),
        .kind = *kind,
        .source = *source,
        .command = core::utils::json::string_field(receipt, "command"),
        .working_directory =
            core::utils::json::string_field(receipt, "working_directory"),
        .exit_code = static_cast<int>(*nested_exit),
        .duration_ms =
            core::utils::json::first_int64_field(receipt, {"duration_ms"})
                .value_or(0),
        .evidence = core::utils::json::string_field(root, "output"),
    };
    if (parsed.receipt_id.empty() || parsed.recipe_id.empty() ||
        parsed.command.empty()) {
      return std::unexpected("verification receipt is incomplete");
    }
    return parsed;
  } catch (const std::exception &error) {
    return std::unexpected(error.what());
  } catch (...) {
    return std::unexpected("verification receipt parsing failed");
  }
}

} // namespace core::verification
