#pragma once

#include "../verification/Verification.hpp"
#include "ShellTool.hpp"
#include "Tool.hpp"

#include <atomic>
#include <cstdint>
#include <memory>

namespace core::tools {

// Executes one verification recipe as an argv vector and emits a typed receipt.
// It deliberately does not accept a shell command string: control operators,
// substitutions, and redirections cannot be used to mask a failing check.
class VerificationTool final : public Tool {
public:
  VerificationTool();
  explicit VerificationTool(std::unique_ptr<shell::IShellExecutor> executor);

  ToolDefinition get_definition() const override;
  std::string execute(const std::string &json_args,
                      const core::context::SessionContext &context) override;
  std::string execute(const std::string &json_args,
                      const ToolInvocationContext &invocation) override;
  void clear_session_state(std::string_view session_id) override;

private:
  std::string execute_impl(const std::string &json_args,
                           const ToolInvocationContext &invocation);
  [[nodiscard]] std::string
  make_receipt_id(std::string_view tool_call_id);

  core::verification::Catalog catalog_;
  ShellTool shell_;
  std::atomic_uint64_t receipt_sequence_{0};
};

} // namespace core::tools
