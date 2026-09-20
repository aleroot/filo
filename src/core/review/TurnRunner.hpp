#pragma once

#include "Types.hpp"

#include "../agent/Agent.hpp"
#include "../llm/LLMProvider.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace core::review {

class TurnRunner {
public:
    struct Request {
        std::string prompt;
        int max_tokens = kReviewMaxOutputTokens;
        int max_steps = kReviewMaxStepsPerTurn;
        std::string effort = "off";
        bool json_object = true;
        /// Empty means no tools. A non-empty list is the exact allow-set
        /// (read-only tools for large groups). Never leave this empty while
        /// intending "all tools" — review must not be able to edit the tree.
        std::vector<std::string> allowed_tools;
    };

    struct Response {
        std::string text;
        bool interrupted = false;
        std::string error;
    };

    virtual ~TurnRunner() = default;
    [[nodiscard]] virtual Response run(const Request& request) = 0;

    /// Return an independent runner when the underlying provider supports
    /// concurrent requests. A null result deliberately preserves serial
    /// execution for stateful or internally serialized providers.
    [[nodiscard]] virtual std::unique_ptr<TurnRunner>
    fork_for_parallel_request() const {
        return {};
    }
};

// Occupies the session Agent (turn_in_progress, Esc/stop, auth) and restores
// conversation history after each group so later files do not inherit the
// previous group's patch or JSON.
class AgentTurnRunner final : public TurnRunner {
public:
    explicit AgentTurnRunner(
        std::shared_ptr<core::agent::Agent> agent,
        std::function<bool()> cancellation_requested = {});

    [[nodiscard]] Response run(const Request& request) override;
    [[nodiscard]] std::unique_ptr<TurnRunner>
    fork_for_parallel_request() const override;

private:
    std::shared_ptr<core::agent::Agent> agent_;
    std::function<bool()> cancellation_requested_;
};

// History-free completion used by unit tests and any caller that already owns
// an isolated provider. Forks when the provider allows parallel requests.
class ProviderTurnRunner final : public TurnRunner {
public:
    ProviderTurnRunner(std::shared_ptr<core::llm::LLMProvider> provider,
                       std::string model,
                       std::function<bool()> cancellation_requested = {});

    [[nodiscard]] Response run(const Request& request) override;
    [[nodiscard]] std::unique_ptr<TurnRunner>
    fork_for_parallel_request() const override;

private:
    std::shared_ptr<core::llm::LLMProvider> provider_;
    std::string model_;
    std::function<bool()> cancellation_requested_;
};

} // namespace core::review
