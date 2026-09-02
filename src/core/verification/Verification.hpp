#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::verification {

inline constexpr int kReceiptSchemaVersion = 1;

enum class Kind {
  Build,
  Test,
  Lint,
  TypeCheck,
  Format,
  Workflow,
  Custom,
};

enum class Source {
  ProjectConfig,
  CMakePreset,
  PackageScript,
  EcosystemDefault,
  AgentProposed,
};

[[nodiscard]] std::string_view to_string(Kind kind) noexcept;
[[nodiscard]] std::string_view to_string(Source source) noexcept;
[[nodiscard]] std::optional<Kind>
kind_from_string(std::string_view value) noexcept;
[[nodiscard]] std::optional<Source>
source_from_string(std::string_view value) noexcept;

struct CommandSpec {
  std::string executable;
  std::vector<std::string> arguments;
  std::filesystem::path working_directory;
  int timeout_seconds = 600;

  bool operator==(const CommandSpec &) const = default;
};

struct Recipe {
  std::string id;
  std::string display_name;
  Kind kind = Kind::Custom;
  Source source = Source::AgentProposed;
  CommandSpec command;
  bool required = false;

  bool operator==(const Recipe &) const = default;
};

struct Receipt {
  std::string receipt_id;
  std::string recipe_id;
  Kind kind = Kind::Custom;
  Source source = Source::AgentProposed;
  std::string command;
  std::filesystem::path working_directory;
  int exit_code = -1;
  long long duration_ms = 0;
  std::string evidence;

  [[nodiscard]] bool passed() const noexcept { return exit_code == 0; }
};

struct DiscoveryContext {
  std::filesystem::path project_root;
  std::string host_system;
};

struct DiscoveryResult {
  std::vector<Recipe> recipes;
  std::vector<std::string> warnings;
  bool project_config_present = false;
  bool project_config_valid = true;
};

class IRecipeProvider {
public:
  virtual ~IRecipeProvider() = default;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual DiscoveryResult
  discover(const DiscoveryContext &context) const = 0;
};

// Repository-aware catalog. Providers are injected to keep discovery open for
// new ecosystems without changing AUTO or the verification runner.
class Catalog {
public:
  Catalog();
  explicit Catalog(
      std::vector<std::shared_ptr<const IRecipeProvider>> providers);

  [[nodiscard]] DiscoveryResult
  discover(const std::filesystem::path &project_root) const noexcept;

  [[nodiscard]] static std::string
  render_for_prompt(std::span<const Recipe> recipes);
  [[nodiscard]] static std::string
  render_warnings(std::span<const std::string> warnings);

  /// Select the smallest deterministic quality gate for fallback execution:
  /// all explicitly required recipes; otherwise one workflow, or one build
  /// and one test, with a single-recipe fallback for other ecosystems.
  [[nodiscard]] static std::vector<std::string>
  default_quality_gate(std::span<const Recipe> recipes);

private:
  std::vector<std::shared_ptr<const IRecipeProvider>> providers_;
};

[[nodiscard]] std::string display_command(const CommandSpec &command);

// Strict parser for results emitted by the built-in run_verification tool.
// Ordinary terminal output can never be interpreted as a receipt.
[[nodiscard]] std::expected<Receipt, std::string>
parse_receipt(std::string_view tool_result) noexcept;

} // namespace core::verification
