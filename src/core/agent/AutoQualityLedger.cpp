#include "AutoQualityLedger.hpp"

#include "SafetyPolicy.hpp"
#include "../tools/ToolNames.hpp"
#include "../utils/JsonUtils.hpp"

#include <cctype>
#include <ranges>

namespace core::agent {

namespace {

[[nodiscard]] std::string ascii_lower(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (const unsigned char ch : value) {
    out.push_back(static_cast<char>(std::tolower(ch)));
  }
  return out;
}

} // namespace

void AutoQualityLedger::observe_tool(
    std::string_view tool_name, std::string_view arguments,
    std::string_view result, bool succeeded, bool mutation_hint,
    bool trusted_verification_receipts) noexcept {
  const bool verification_tool =
      core::tools::names::is_verification_tool(tool_name);
  bool mutation = succeeded && !verification_tool &&
                  (mutation_hint ||
                   core::tools::names::is_file_modification_tool(tool_name));
  if (succeeded && core::tools::names::is_terminal_tool(tool_name)) {
    const std::string command =
        CommandSafetyPolicy::extract_shell_command(arguments);
    if (CommandSafetyPolicy::classify(command) != CommandSafetyClass::Safe) {
      mutation = true;
    }
  }
  if (succeeded && core::tools::names::is_subagent_tool(tool_name)) {
    const std::string worker =
        ascii_lower(core::utils::json::first_string_field(
                        arguments, {"subagent_type", "worker"})
                        .value_or(std::string{}));
    if (worker != "explore") {
      mutation = true;
    }
  }
  if (mutation) {
    mutation_observed_.store(true, std::memory_order_release);
    mutation_generation_.fetch_add(1, std::memory_order_acq_rel);
  }
  if (verification_tool && trusted_verification_receipts) {
    verification_attempted_.store(true, std::memory_order_release);
    const auto receipt = core::verification::parse_receipt(result);
    if (succeeded && receipt.has_value() && receipt->passed()) {
      observe_verification_receipt(*receipt);
    }
  }
}

void AutoQualityLedger::begin_tool_batch() noexcept {
  batch_generation_.store(mutation_generation_.load(std::memory_order_acquire),
                          std::memory_order_release);
}

void AutoQualityLedger::end_tool_batch() noexcept {
  batch_generation_.store(kNoActiveBatch, std::memory_order_release);
}

std::uint64_t AutoQualityLedger::evidence_generation() const noexcept {
  const auto batch = batch_generation_.load(std::memory_order_acquire);
  return batch == kNoActiveBatch
             ? mutation_generation_.load(std::memory_order_acquire)
             : batch;
}

void AutoQualityLedger::record_passed_recipe(
    const core::verification::Receipt &receipt,
    std::uint64_t generation) noexcept {
  std::lock_guard lock(receipts_mutex_);
  passed_recipes_.insert_or_assign(receipt.recipe_id,
                                   PassedRecipeEvidence{
                                       .source = receipt.source,
                                       .command = receipt.command,
                                       .mutation_generation = generation,
                                   });
}

void AutoQualityLedger::observe_verification_receipt(
    const core::verification::Receipt &receipt) noexcept {
  verification_attempted_.store(true, std::memory_order_release);
  if (!receipt.passed())
    return;
  record_passed_recipe(receipt, evidence_generation());
}

bool AutoQualityLedger::requirements_met(
    std::span<const core::verification::Recipe> recipes) const noexcept {
  std::vector<std::string_view> required_ids;
  for (const auto &recipe : recipes) {
    if (recipe.required)
      required_ids.push_back(recipe.id);
  }
  std::lock_guard lock(receipts_mutex_);
  const auto generation = mutation_generation_.load(std::memory_order_acquire);
  const auto recipe_passed = [&](const core::verification::Recipe &recipe) {
    const auto passed = passed_recipes_.find(recipe.id);
    return passed != passed_recipes_.end() &&
           passed->second.source == recipe.source &&
           passed->second.mutation_generation == generation &&
           passed->second.command ==
               core::verification::display_command(recipe.command);
  };
  if (required_ids.empty()) {
    if (recipes.empty()) {
      return std::ranges::any_of(passed_recipes_, [&](const auto &entry) {
        return entry.second.mutation_generation == generation;
      });
    }
    const auto catalog_has_kind = [&](core::verification::Kind kind) {
      return std::ranges::any_of(
          recipes, [&](const auto &recipe) { return recipe.kind == kind; });
    };
    const auto passed_kind = [&](core::verification::Kind kind) {
      return std::ranges::any_of(recipes, [&](const auto &recipe) {
        return recipe.kind == kind && recipe_passed(recipe);
      });
    };

    const bool has_build = catalog_has_kind(core::verification::Kind::Build);
    const bool has_test = catalog_has_kind(core::verification::Kind::Test);
    if (passed_kind(core::verification::Kind::Workflow))
      return true;
    if (has_build || has_test) {
      return (!has_build || passed_kind(core::verification::Kind::Build)) &&
             (!has_test || passed_kind(core::verification::Kind::Test));
    }
    return std::ranges::any_of(
        recipes, [&](const auto &recipe) { return recipe_passed(recipe); });
  }
  return std::ranges::all_of(required_ids, [&](std::string_view id) {
    const auto recipe =
        std::ranges::find(recipes, id, &core::verification::Recipe::id);
    return recipe != recipes.end() && recipe_passed(*recipe);
  });
}

bool AutoQualityLedger::has_explicit_exception(
    std::string_view response) noexcept {
  const std::string lower = ascii_lower(response);
  return lower.contains("verification exception:") ||
         lower.contains("verification not applicable:");
}

std::string AutoQualityLedger::verification_follow_up() {
  return "AUTO quality gate: workspace changes were made, but Filo has no "
         "successful fresh quality-hook evidence for the latest mutation. "
         "Fix the reported failure and let the completion hooks run again. "
         "You may use `run_verification` for targeted checks, but it is not "
         "the only quality mechanism. If no check can genuinely run, explain "
         "the concrete blocker in the final answer using `Verification "
         "exception: <specific reason>`.";
}

} // namespace core::agent
