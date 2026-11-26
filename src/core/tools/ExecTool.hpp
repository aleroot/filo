#pragma once

#include "Tool.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::llm {
struct Tool;
}

namespace core::tools {

// `exec` runs a JavaScript script whose `tools.<name>(input)` calls reach the
// tools a step offers, and whose `text(...)` calls make up the result. A script
// that is still running when its yield time elapses stays alive as a cell;
// `wait` resumes that cell. How either tool is put on the wire is the provider
// protocol's business.

/// Whether this build has the JavaScript runtime (CMake FILO_ENABLE_JAVASCRIPT).
/// Without it no model is offered `exec`.
[[nodiscard]] constexpr bool tool_script_available() noexcept {
#ifdef FILO_ENABLE_JAVASCRIPT
    return true;
#else
    return false;
#endif
}

/// How long `exec` runs before yielding a still-running script, unless the
/// script's `// @exec:` line asks for another. Codex's default.
inline constexpr std::uint64_t kDefaultExecYieldMs = 10'000;

/// The argument contract of an `exec` call: the script, as `{"source": ...}`.
[[nodiscard]] ToolDefinition exec_argument_definition();
[[nodiscard]] std::string arguments_for_source(std::string_view source);
/// The script of an `exec` call, or nothing when @p arguments is not the
/// object arguments_for_source builds.
[[nodiscard]] std::optional<std::string> source_from_arguments(std::string_view arguments);

/// What the model reads about `exec`: how scripts run, the helpers, and a
/// TypeScript declaration of every tool in @p tools.
[[nodiscard]] std::string tool_script_description(const std::vector<core::llm::Tool>& tools);

/// What the model reads about `wait`: resume a cell `exec` yielded.
[[nodiscard]] std::string wait_tool_description();

/// The JavaScript identifier a tool is reachable under: `tools.<identifier>`.
[[nodiscard]] std::string script_identifier(std::string_view tool_name);

/// Arguments a script passed to a nested tool, as the tool expects them. A
/// script may pass a bare string to a tool with exactly one required string
/// parameter (`tools.apply_patch(patch)`); it becomes that parameter.
[[nodiscard]] std::string nested_tool_arguments(const ToolDefinition& definition,
                                                std::string arguments);

/// One `tools.<name>(input)` call. `arguments` is JSON: an object, or a
/// string when the script passed one.
struct ToolScriptCall {
    std::string tool_name;
    std::string arguments;
};

/// The answer to one call. A call that ran resolves with its result (parsed
/// when it is JSON); a call that could not run rejects with @c value as the
/// error message.
struct ToolScriptReply {
    bool ran = false;
    std::string value;
};

/// Runs the tools a script calls. The runtime hands over every call the
/// script issued before it next waited, so independent calls batched with
/// Promise.all arrive together and may run in parallel.
class ToolScriptHost {
public:
    virtual ~ToolScriptHost() = default;
    /// One reply per call, in call order.
    [[nodiscard]] virtual std::vector<ToolScriptReply> run(std::span<const ToolScriptCall> calls) = 0;
    [[nodiscard]] virtual bool cancelled() const { return false; }
};

/// What `store(key, value)` keeps for later scripts of the same conversation.
/// Values are JSON text; nothing (`std::nullopt`) removes the key.
class ToolScriptStore {
public:
    /// Keys and values together, so scripts cannot grow the session without bound.
    static constexpr std::size_t kMaxBytes = 16 * 1024 * 1024;

    /// False, and nothing changes, when the value would exceed kMaxBytes.
    [[nodiscard]] bool store(std::string key, std::optional<std::string> json);
    [[nodiscard]] std::optional<std::string> load(const std::string& key) const;
    /// Forgets every value: the conversation the scripts belonged to is gone.
    void clear();

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> values_;
    std::size_t bytes_ = 0;
};

struct ToolScriptOutcome {
    bool completed = false;
    /// True when the script is still running and @c output names its cell.
    bool running = false;
    /// The `exec` or `wait` result the model reads: status, wall time, then
    /// the text the script reported, and the error when it failed.
    std::string output;
};

/// The exec cells of one conversation. A script that yields stays here until
/// `wait` finishes it, terminates it, or @c clear drops the conversation.
class ToolScriptSession {
public:
    ToolScriptSession();
    ~ToolScriptSession();
    ToolScriptSession(const ToolScriptSession&) = delete;
    ToolScriptSession& operator=(const ToolScriptSession&) = delete;

    void clear();
    [[nodiscard]] ToolScriptOutcome execute(std::string_view source,
                                            const std::vector<core::llm::Tool>& tools,
                                            ToolScriptHost& host);
    /// @p arguments is `wait`'s JSON object: cell_id, and optionally
    /// yield_time_ms, max_tokens, and terminate.
    [[nodiscard]] ToolScriptOutcome resume(std::string_view arguments, ToolScriptHost& host);

private:
    struct State;
    std::unique_ptr<State> state_;
};

/// Runs one script to completion. A script that would yield is not kept: this
/// is the one-shot path tests use. A conversation uses ToolScriptSession.
[[nodiscard]] ToolScriptOutcome run_tool_script(std::string_view source,
                                                const std::vector<core::llm::Tool>& tools,
                                                ToolScriptHost& host,
                                                ToolScriptStore& store);

} // namespace core::tools
