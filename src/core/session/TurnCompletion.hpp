#pragma once

#include <string>

namespace core::session {

// Transport-independent outcome of a policy that decides whether an agent
// turn may finish, must continue with feedback, or has irrecoverably failed.
enum class TurnCompletionAction { Complete, Continue, Fail };

struct TurnCompletionResult {
  TurnCompletionAction action = TurnCompletionAction::Complete;
  bool quality_gate_satisfied = false;
  std::string status;
  std::string message;
};

} // namespace core::session
