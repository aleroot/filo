#pragma once

#include "../verification/Verification.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::agent {

// Thread-safe evidence store for one AUTO turn. Successful receipts are bound
// to the latest mutation generation so a later edit invalidates older proof.
class AutoQualityLedger final {
public:
  void observe_tool(std::string_view tool_name, std::string_view arguments,
                    std::string_view result, bool succeeded,
                    bool mutation_hint = false,
                    bool trusted_verification_receipts = false) noexcept;
  void observe_verification_receipt(
      const core::verification::Receipt &receipt) noexcept;

  /// Snapshot the mutation counter before dispatching a batch of tools that
  /// may run concurrently. Receipts minted by that batch bind to this
  /// generation, so a mutation racing a verification command inside the same
  /// batch still invalidates the evidence. Sampling at record time instead
  /// would let a receipt silently claim a mutation it never observed.
  void begin_tool_batch() noexcept;
  /// Leave batch scope; later receipts bind to the live generation again.
  void end_tool_batch() noexcept;

  [[nodiscard]] bool mutation_observed() const noexcept {
    return mutation_observed_.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool verification_attempted() const noexcept {
    return verification_attempted_.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool verification_passed() const noexcept {
    std::lock_guard lock(receipts_mutex_);
    const auto generation = mutation_generation_.load(std::memory_order_acquire);
    return std::ranges::any_of(passed_recipes_, [&](const auto &entry) {
      return entry.second.mutation_generation == generation;
    });
  }
  [[nodiscard]] bool needs_verification() const noexcept {
    return mutation_observed() && !verification_passed();
  }
  [[nodiscard]] bool requirements_met(
      std::span<const core::verification::Recipe> recipes) const noexcept;
  [[nodiscard]] bool needs_verification(
      std::span<const core::verification::Recipe> recipes) const noexcept {
    return mutation_observed() && !requirements_met(recipes);
  }

  [[nodiscard]] std::string verification_summary() const;

  [[nodiscard]] static bool
  has_explicit_exception(std::string_view response) noexcept;
  [[nodiscard]] static std::string verification_follow_up();

private:
  struct PassedRecipeEvidence {
    core::verification::Source source =
        core::verification::Source::AgentProposed;
    std::string command;
    std::uint64_t mutation_generation = 0;
  };

  static constexpr std::uint64_t kNoActiveBatch =
      std::numeric_limits<std::uint64_t>::max();

  /// Generation a receipt observed when its command started: the batch
  /// snapshot while a batch is open, otherwise the live counter.
  [[nodiscard]] std::uint64_t evidence_generation() const noexcept;
  void record_passed_recipe(const core::verification::Receipt &receipt,
                            std::uint64_t generation) noexcept;

  std::atomic_bool mutation_observed_{false};
  std::atomic_uint64_t mutation_generation_{0};
  std::atomic_uint64_t batch_generation_{kNoActiveBatch};
  std::atomic_bool verification_attempted_{false};
  mutable std::mutex receipts_mutex_;
  std::unordered_map<std::string, PassedRecipeEvidence> passed_recipes_;
};

} // namespace core::agent
