#pragma once

#include <array>
#include <cctype>
#include <string>
#include <string_view>

namespace core::agent {

// AgentMode is the single source of truth for user-visible execution modes.
// Keep presentation (colours, key bindings) outside this type; the core owns
// only stable names and behavioural capabilities.
enum class AgentMode {
  Auto,
  Build,
  Debug,
  Research,
  Execute,
};

struct AgentModeDescriptor {
  AgentMode mode;
  std::string_view name;
  std::string_view description;
};

inline constexpr std::array<AgentModeDescriptor, 5> kAgentModes{{
    {AgentMode::Auto, "AUTO",
     "Adapt between a direct turn and an orchestrated, evidence-gated "
     "workflow."},
    {AgentMode::Build, "BUILD",
     "Build software methodically with normal agent autonomy."},
    {AgentMode::Debug, "DEBUG",
     "Reproduce, diagnose, fix, and verify a failure."},
    {AgentMode::Research, "RESEARCH",
     "Inspect and plan without modifying the workspace."},
    {AgentMode::Execute, "EXECUTE", "Execute the requested changes directly."},
}};

[[nodiscard]] constexpr std::string_view to_string(AgentMode mode) noexcept {
  for (const auto &descriptor : kAgentModes) {
    if (descriptor.mode == mode)
      return descriptor.name;
  }
  return "BUILD";
}

[[nodiscard]] inline AgentMode
agent_mode_from_string(std::string_view value) noexcept {
  std::string normalized;
  normalized.reserve(value.size());
  for (const unsigned char ch : value) {
    if (std::isalpha(ch)) {
      normalized.push_back(static_cast<char>(std::toupper(ch)));
    }
  }

  // PLAN remains a compatibility alias for the historical read-only mode.
  if (normalized == "AUTO")
    return AgentMode::Auto;
  if (normalized == "DEBUG")
    return AgentMode::Debug;
  if (normalized == "PLAN" || normalized == "RESEARCH")
    return AgentMode::Research;
  if (normalized == "EXECUTE")
    return AgentMode::Execute;
  return AgentMode::Build;
}

[[nodiscard]] constexpr bool is_read_only_mode(AgentMode mode) noexcept {
  return mode == AgentMode::Research;
}

[[nodiscard]] constexpr bool is_auto_mode(AgentMode mode) noexcept {
  return mode == AgentMode::Auto;
}

// AUTO is a parent-transaction mode. Delegated workers must never inherit it:
// they would re-enter graph planning and try to take the same repository lease.
// Explore stays read-only; mutating workers use BUILD so the parent remains
// the sole quality-gate and exclusive-write owner.
[[nodiscard]] constexpr AgentMode
delegated_execution_mode(AgentMode parent, bool read_only) noexcept {
  if (parent != AgentMode::Auto)
    return parent;
  return read_only ? AgentMode::Research : AgentMode::Build;
}

[[nodiscard]] inline std::string
delegated_mode_name(std::string_view parent_mode, bool read_only) {
  const AgentMode parsed = agent_mode_from_string(parent_mode);
  if (parsed != AgentMode::Auto) {
    std::string preserved;
    preserved.reserve(parent_mode.size());
    for (const unsigned char ch : parent_mode) {
      if (std::isalpha(ch)) {
        preserved.push_back(static_cast<char>(std::toupper(ch)));
      }
    }
    return preserved.empty() ? std::string("BUILD") : preserved;
  }
  return std::string(to_string(delegated_execution_mode(parsed, read_only)));
}

} // namespace core::agent
