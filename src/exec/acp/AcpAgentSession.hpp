#pragma once

#include "AcpSession.hpp"
#include "core/agent/Agent.hpp"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace exec::acp {

/// Filo agent bound to one editor session.
class AgentSession final : public Session {
public:
    struct Options {
        bool yolo = false;
        std::vector<std::string> trusted_tools;
        /// A resumed conversation keeps its start time: the session file is
        /// named from it, and a new time would write a second file.
        std::string created_at;
    };

    AgentSession(std::shared_ptr<core::agent::Agent> agent,
                 std::string id,
                 std::filesystem::path cwd,
                 std::string provider_name,
                 Options options,
                 std::vector<core::llm::Message> replay = {},
                 std::filesystem::path sessions_dir = {});

    ~AgentSession() override;
    AgentSession(const AgentSession&) = delete;
    AgentSession& operator=(const AgentSession&) = delete;

    [[nodiscard]] std::string id() const override;
    [[nodiscard]] core::utils::JsonValue opened_result() const override;
    void announce(ClientChannel& client) override;
    void replay(ClientChannel& client) override;
    [[nodiscard]] std::string prompt(const core::utils::JsonValue& blocks,
                                     ClientChannel& client) override;
    void cancel() override;
    void discard() override;
    [[nodiscard]] std::optional<std::string> set_mode(std::string_view mode_id,
                                                      ClientChannel& client) override;
    [[nodiscard]] std::optional<std::string> set_config(
        std::string_view config_id,
        const core::utils::JsonValue& value,
        ClientChannel& client) override;
    [[nodiscard]] core::utils::JsonValue config_state() const override;

    /// Asks the editor the questions of an ask_user_question call and fulfils
    /// the request's promise. Called on the tool's thread during a turn.
    void answer_question(core::tools::QuestionRequest request);

private:
    struct LiveCall {
        std::string id;
        std::string name;
        std::string args;
        bool finished = false;
        /// Files the call may change, as they were before it ran.
        std::vector<std::pair<std::filesystem::path, FileSide>> before;
    };

    [[nodiscard]] std::string provider_name() const;
    void install_permissions();
    void emit(const core::utils::JsonValue& params);
    /// A status line from the turn: a notice when the editor shows them,
    /// otherwise a thought, so it never lands in the answer.
    void emit_status(std::string_view text);
    void emit_state(ClientChannel& client);
    void capture_before(LiveCall& call) const;
    void publish_tool(const LiveCall& call, std::string status, std::string output, bool initial);
    void publish_plan();
    void publish_usage();
    void publish_info(std::string_view prompt_text);
    void save() const;
    [[nodiscard]] std::string run_slash(std::string_view name,
                                        std::string_view argument,
                                        ClientChannel& client);
    [[nodiscard]] std::vector<ConfigSelect> config_selects() const;
    [[nodiscard]] std::vector<ModeChoice> modes() const;
    [[nodiscard]] static std::vector<CommandSpec> commands();

    std::string id_;
    std::filesystem::path cwd_;
    std::filesystem::path sessions_dir_;
    std::string provider_name_;  // guarded by state_mu_
    std::string created_at_;
    Options options_;
    std::vector<core::llm::Message> replay_;
    std::vector<std::string> allow_rules_;

    // Declared before the agent: tool threads use them, and the agent is
    // destroyed first.
    mutable std::mutex state_mu_;
    ClientChannel* channel_ = nullptr;
    std::vector<LiveCall> calls_;
    std::string permission_call_id_;
    std::atomic<bool> cancel_requested_{false};
    std::atomic<bool> discarded_{false};
    std::atomic<bool> auth_required_{false};
    /// The mode to return to when the editor leaves plan mode. Guarded by state_mu_.
    std::string mode_before_plan_ = "BUILD";
    /// Set once the editor has been given a title for this conversation.
    bool titled_ = false;

    std::shared_ptr<core::agent::Agent> agent_;
};

/// Hands an ask_user_question call to the session that made it (by
/// `request.session_id`); with no such session the question is dismissed.
/// This is the question tool's callback in an ACP process.
void route_question(core::tools::QuestionRequest request);

} // namespace exec::acp
