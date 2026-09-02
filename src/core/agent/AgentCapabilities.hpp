#pragma once

#include "../context/SessionContext.hpp"
#include "../verification/Verification.hpp"

#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace core::llm {
class LLMProvider;
}

namespace core::memory {
class MemorySystem;
}

namespace core::tools {
class ToolManager;
}

namespace core::agent {
class SubagentOrchestrator;
struct SubagentEvent;
}

namespace core::agent::capabilities {

using PermissionCheck =
    std::function<bool(const std::string &, const std::string &)>;

struct VerificationRequest {
  std::string recipe_id;
  core::context::SessionContext session_context;
  std::string tool_call_id;
  std::string provider_name;
  std::string model_name;
  std::shared_ptr<core::llm::LLMProvider> provider;
  PermissionCheck permission_check;
};

[[nodiscard]] std::expected<core::verification::Receipt, std::string>
execute_verification(core::tools::ToolManager &tools,
                     VerificationRequest request);

struct ExplorationRequest {
  std::string description;
  std::string prompt;
  std::shared_ptr<core::llm::LLMProvider> provider;
  std::string provider_name;
  std::string model_name;
  std::string parent_mode;
  core::context::SessionContext session_context;
  std::string tool_call_id;
  PermissionCheck permission_check;
  std::function<void(const SubagentEvent &)> on_subagent_event;
  std::function<bool()> cancellation_requested;
  std::shared_ptr<core::memory::MemorySystem> memory_system;
};

[[nodiscard]] std::expected<std::string, std::string>
execute_exploration(SubagentOrchestrator &orchestrator,
                    ExplorationRequest request);

} // namespace core::agent::capabilities
