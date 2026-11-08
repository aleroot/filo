#include "AcpServer.hpp"

#include "core/jsonrpc/StdioLineReader.hpp"
#include "core/auth/AuthLogin.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/logging/Logger.hpp"
#include "core/session/SessionStore.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace exec::acp {
namespace {

class Connection final : public ClientChannel {
public:
    explicit Connection(AcpTransport& transport) : transport_(transport) {}

    void notify(std::string_view method, const core::utils::JsonValue& params) override {
        write(encode_notification(method, params));
    }

    core::utils::JsonValue request(std::string_view method,
                                   const core::utils::JsonValue& params) override {
        RpcId id;
        id.kind = RpcId::Kind::number;
        id.number = next_id_.fetch_add(1, std::memory_order_relaxed);
        auto waiter = std::make_shared<Waiter>();
        {
            std::lock_guard lock(waiters_mu_);
            waiters_.push_back(Pending{
                .key = id.key(),
                .session_id = session_of(params),
                .waiter = waiter,
            });
        }
        write(encode_request(id, method, params));
        return waiter->get();
    }

    [[nodiscard]] ClientCapabilities capabilities() const override {
        std::lock_guard lock(capabilities_mu_);
        return capabilities_;
    }

    void set_capabilities(ClientCapabilities capabilities) {
        std::lock_guard lock(capabilities_mu_);
        capabilities_ = capabilities;
    }

    void reply(const RpcId& id, const core::utils::JsonValue& result) {
        write(encode_result(id, result));
    }

    void reply_error(const RpcId& id,
                     int code,
                     std::string_view message,
                     const core::utils::JsonValue* data = nullptr) {
        write(encode_error(id, code, message, data));
    }

    void complete(const RpcMessage& message) {
        std::shared_ptr<Waiter> waiter;
        {
            std::lock_guard lock(waiters_mu_);
            const auto key = message.id.key();
            const auto it = std::ranges::find_if(
                waiters_, [&](const Pending& pending) { return pending.key == key; });
            if (it == waiters_.end()) return;
            waiter = it->waiter;
            waiters_.erase(it);
        }
        if (message.has_error || message.result.is_null()) waiter->set({});
        else waiter->set(message.result);
    }

    void cancel_session(std::string_view session_id) {
        std::vector<std::shared_ptr<Waiter>> waiters;
        {
            std::lock_guard lock(waiters_mu_);
            std::erase_if(waiters_, [&](const Pending& pending) {
                if (pending.session_id != session_id) return false;
                waiters.push_back(pending.waiter);
                return true;
            });
        }
        for (const auto& waiter : waiters) waiter->set(cancelled_permission());
    }

    void cancel_all() {
        std::vector<std::shared_ptr<Waiter>> waiters;
        {
            std::lock_guard lock(waiters_mu_);
            for (auto& pending : waiters_) waiters.push_back(pending.waiter);
            waiters_.clear();
        }
        for (const auto& waiter : waiters) waiter->set(cancelled_permission());
    }

private:
    struct Waiter {
        std::promise<core::utils::JsonValue> promise;
        std::atomic<bool> done{false};

        void set(core::utils::JsonValue value) {
            if (done.exchange(true, std::memory_order_acq_rel)) return;
            promise.set_value(std::move(value));
        }

        [[nodiscard]] core::utils::JsonValue get() { return promise.get_future().get(); }
    };

    struct Pending {
        std::string key;
        std::string session_id;
        std::shared_ptr<Waiter> waiter;
    };

    static core::utils::JsonValue cancelled_permission() {
        auto result = core::utils::JsonValue::object();
        auto outcome = core::utils::JsonValue::object();
        outcome.set("outcome", "cancelled");
        result.set("outcome", std::move(outcome));
        return result;
    }

    static std::string session_of(const core::utils::JsonValue& params) {
        const auto* id = params.is_object() ? params.find("sessionId") : nullptr;
        if (!id || !id->is_string()) return {};
        return std::string(id->as_string());
    }

    void write(std::string line) {
        std::lock_guard lock(write_mu_);
        transport_.write_line(line);
    }

    AcpTransport& transport_;
    std::mutex write_mu_;
    std::mutex waiters_mu_;
    std::vector<Pending> waiters_;
    std::atomic<long long> next_id_{1};
    mutable std::mutex capabilities_mu_;
    ClientCapabilities capabilities_;
};

class StdioTransport final : public AcpTransport {
public:
    /// `trace_path`, when set, receives every protocol line in both
    /// directions. It holds prompts, file contents and the credentials of MCP
    /// servers the editor passes, so it is opt-in and readable by the owner only.
    explicit StdioTransport(const char* trace_path) : reader_(STDIN_FILENO) {
        if (trace_path && *trace_path) {
            const int fd = ::open(trace_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
            if (fd >= 0) ::close(fd);
            trace_.open(trace_path, std::ios::app);
            if (!trace_) core::logging::warn("ACP trace file {} could not be opened", trace_path);
        }
        protocol_fd_ = ::dup(STDOUT_FILENO);
        if (protocol_fd_ >= 0) {
            ::fcntl(protocol_fd_, F_SETFD, FD_CLOEXEC);
            // Anything that prints to stdout after this (a library, a tool that
            // inherits the process) must not corrupt the protocol stream.
            ::dup2(STDERR_FILENO, STDOUT_FILENO);
        }
    }

    ~StdioTransport() override {
        if (protocol_fd_ >= 0) ::close(protocol_fd_);
    }

    StdioTransport(const StdioTransport&) = delete;
    StdioTransport& operator=(const StdioTransport&) = delete;

    [[nodiscard]] std::optional<std::string> read_line() override {
        const auto read = reader_.read_until(std::nullopt);
        if (read.status != core::jsonrpc::StdioLineReader::Status::line) return std::nullopt;
        trace("<- ", read.line);
        return read.line;
    }

    void write_line(std::string_view line) override {
        if (protocol_fd_ < 0) return;
        std::lock_guard lock(write_mu_);
        write_all(line);
        write_all("\n");
        trace("-> ", line);
    }

private:
    void trace(std::string_view direction, std::string_view line) {
        if (!trace_.is_open()) return;
        std::lock_guard lock(trace_mu_);
        trace_ << direction << line << '\n';
        trace_.flush();
    }

    void write_all(std::string_view bytes) {
        while (!bytes.empty()) {
            const auto wrote = ::write(protocol_fd_, bytes.data(), bytes.size());
            if (wrote < 0) {
                if (errno == EINTR) continue;
                return;
            }
            bytes.remove_prefix(static_cast<std::size_t>(wrote));
        }
    }

    core::jsonrpc::StdioLineReader reader_;
    int protocol_fd_ = -1;
    std::mutex write_mu_;
    std::mutex trace_mu_;
    std::ofstream trace_;
};

struct Slot {
    std::unique_ptr<Session> session;
    std::atomic<bool> busy{false};
};

[[nodiscard]] const core::utils::JsonValue* object_field(
    const core::utils::JsonValue& object,
    std::string_view key) {
    if (!object.is_object()) return nullptr;
    return object.find(key);
}

[[nodiscard]] std::string text_field(const core::utils::JsonValue& object, std::string_view key) {
    const auto* value = object_field(object, key);
    if (!value || !value->is_string()) return {};
    return std::string(value->as_string());
}

[[nodiscard]] std::vector<std::filesystem::path> directories_field(
    const core::utils::JsonValue& object) {
    std::vector<std::filesystem::path> paths;
    const auto* value = object_field(object, "additionalDirectories");
    if (!value || !value->is_array()) return paths;
    for (const auto& entry : value->items()) {
        if (entry.is_string() && !entry.as_string().empty()) {
            paths.emplace_back(entry.as_string());
        }
    }
    return paths;
}

} // namespace

int serve(AcpTransport& transport, const SessionFactory& open, ServeOptions options) {
    SessionCatalog* const catalog = options.catalog;
    Connection connection(transport);
    bool initialized = false;
    // Shared so a prompt still running keeps its session alive after
    // session/close drops it from the map.
    std::map<std::string, std::shared_ptr<Slot>> sessions;
    std::mutex sessions_mu;
    struct Worker {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> done;
    };
    std::vector<Worker> workers;
    std::mutex workers_mu;
    // Joins prompt threads that have finished, so a long editor session does
    // not accumulate one per prompt. Caller holds workers_mu.
    const auto reap_workers = [&] {
        std::erase_if(workers, [](Worker& worker) {
            if (!worker.done->load(std::memory_order_acquire)) return false;
            if (worker.thread.joinable()) worker.thread.join();
            return true;
        });
    };
    const auto find_session = [&](const std::string& session_id) -> std::shared_ptr<Slot> {
        std::lock_guard lock(sessions_mu);
        const auto it = sessions.find(session_id);
        return it == sessions.end() ? nullptr : it->second;
    };
    struct LivePrompt {
        RpcId id;
        std::string session_id;
    };
    std::mutex live_mu;
    std::vector<LivePrompt> live_prompts;

    auto fail = [&](const RpcId& id,
                    int code,
                    std::string_view message,
                    const core::utils::JsonValue* data = nullptr) {
        connection.reply_error(id, code, message, data);
    };
    auto session_missing = [&](const RpcId& id, std::string session_id) {
        // Same code and data the Agent Client Protocol SDK uses for an
        // unknown session. -32602 is reserved for malformed parameters.
        auto data = core::utils::JsonValue::object();
        data.set("sessionId", std::move(session_id));
        connection.reply_error(id, -32002, "Resource not found", &data);
    };
    auto cancel_session = [&](const std::string& session_id) {
        if (session_id.empty()) return;
        {
            std::lock_guard lock(sessions_mu);
            if (const auto it = sessions.find(session_id); it != sessions.end()) {
                it->second->session->cancel();
            }
        }
        connection.cancel_session(session_id);
    };
    auto remember_prompt = [&](RpcId id, std::string session_id) {
        std::lock_guard lock(live_mu);
        live_prompts.push_back(LivePrompt{std::move(id), std::move(session_id)});
    };
    auto forget_prompt = [&](const RpcId& id) {
        const auto key = id.key();
        std::lock_guard lock(live_mu);
        std::erase_if(live_prompts, [&](const LivePrompt& prompt) {
            return prompt.id.key() == key;
        });
    };
    auto prompt_session = [&](const RpcId& id) {
        const auto key = id.key();
        std::lock_guard lock(live_mu);
        const auto it = std::ranges::find_if(live_prompts, [&](const LivePrompt& prompt) {
            return prompt.id.key() == key;
        });
        return it == live_prompts.end() ? std::string{} : it->session_id;
    };

    const auto loop = [&] {
        while (auto line = transport.read_line()) {
            if (core::utils::str::trim_ascii_copy(*line).empty()) continue;
            const auto message = parse_message(*line);
            if (message.kind == RpcMessage::Kind::invalid) {
                fail(message.id, -32700,
                     message.parse_error.empty() ? "Parse error" : message.parse_error);
                continue;
            }
            if (message.kind == RpcMessage::Kind::response) {
                connection.complete(message);
                continue;
            }
            if (message.kind == RpcMessage::Kind::notification) {
                if (message.method == "session/cancel") {
                    cancel_session(text_field(message.params, "sessionId"));
                } else if (message.method == "$/cancel_request") {
                    // Cancels the request itself. A prompt answers with
                    // stopReason "cancelled" once its turn notices.
                    cancel_session(prompt_session(rpc_id(object_field(message.params, "requestId"))));
                }
                continue;
            }

            try {
                if (message.method == "initialize") {
                    std::string version_error;
                    const auto version = negotiated_protocol_version(message.params, &version_error);
                    if (!version) {
                        fail(message.id, -32602,
                             version_error.empty() ? "Invalid params" : version_error);
                        continue;
                    }
                    initialized = true;
                    const auto client = parse_client_capabilities(message.params);
                    connection.set_capabilities(client);
                    connection.reply(message.id,
                                     initialize_result(*version,
                                                       client.terminal_auth ? options.login_provider
                                                                            : std::string{}));
                    continue;
                }
                if (!initialized) {
                    fail(message.id, -32600, "Invalid request");
                    continue;
                }
                // No auth methods are advertised. A client that asks anyway is
                // already signed in through Filo's own credential store.
                if (message.method == "authenticate") {
                    connection.reply(message.id, core::utils::JsonValue::object());
                    continue;
                }
                if (message.method == "session/new" || message.method == "session/load"
                    || message.method == "session/resume") {
                    if (!message.params.is_object()) {
                        fail(message.id, -32602, "Invalid params");
                        continue;
                    }
                    // Not advertised. Applying it silently would make the editor
                    // believe instructions the model never saw.
                    if (const auto* system_prompt = object_field(message.params, "systemPrompt");
                        system_prompt && !system_prompt->is_null()) {
                        fail(message.id, -32602, "systemPrompt is not supported");
                        continue;
                    }
                    OpenRequest request;
                    request.cwd = text_field(message.params, "cwd");
                    request.additional_directories = directories_field(message.params);
                    std::string mcp_error;
                    auto servers = parse_mcp_servers(object_field(message.params, "mcpServers"), &mcp_error);
                    if (!servers) {
                        fail(message.id, -32602, mcp_error.empty() ? "Invalid mcpServers" : mcp_error);
                        continue;
                    }
                    request.mcp_servers = std::move(*servers);
                    request.client = connection.capabilities();
                    const bool loading = message.method == "session/load";
                    const bool resuming = message.method == "session/resume";
                    if (loading || resuming) {
                        request.resume_id = text_field(message.params, "sessionId");
                        if (request.resume_id->empty()) {
                            fail(message.id, -32602, "sessionId is required");
                            continue;
                        }
                    }
                    auto opened = open(request);
                    if (!opened) {
                        const auto& error = opened.error();
                        if (error.kind == OpenError::Kind::not_found) {
                            session_missing(message.id, request.resume_id.value_or(""));
                        } else if (error.kind == OpenError::Kind::internal) {
                            fail(message.id, -32603, error.message);
                        } else {
                            fail(message.id, -32602, error.message);
                        }
                        continue;
                    }
                    const auto id = (*opened)->id();
                    auto slot = std::make_shared<Slot>();
                    slot->session = std::move(*opened);
                    {
                        std::lock_guard lock(sessions_mu);
                        if (sessions.contains(id)) {
                            fail(message.id, -32602, "Session is already open");
                            continue;
                        }
                        sessions.emplace(id, std::move(slot));
                    }
                    const auto opened_slot = find_session(id);
                    auto& session = *opened_slot->session;
                    // History is replayed before the load answer; resume never replays.
                    if (loading) session.replay(connection);
                    auto result = session.opened_result();
                    // session/load and session/resume already name the session.
                    // Their responses have modes and config options, not sessionId.
                    if (loading || resuming) result.erase("sessionId");
                    connection.reply(message.id, result);
                    // After the answer: until then the editor does not know the
                    // session id, and may drop updates addressed to it.
                    session.announce(connection);
                    continue;
                }
                if (message.method == "session/prompt") {
                    const auto session_id = text_field(message.params, "sessionId");
                    const auto* prompt = object_field(message.params, "prompt");
                    std::shared_ptr<Slot> slot;
                    {
                        std::lock_guard lock(sessions_mu);
                        const auto it = sessions.find(session_id);
                        if (it == sessions.end()) {
                            session_missing(message.id, session_id);
                            continue;
                        }
                        if (it->second->busy.exchange(true, std::memory_order_acq_rel)) {
                            fail(message.id, -32602, "A prompt is already running");
                            continue;
                        }
                        slot = it->second;
                    }
                    if (!prompt || !prompt->is_array()) {
                        slot->busy.store(false, std::memory_order_release);
                        fail(message.id, -32602, "prompt must be an array of content blocks");
                        continue;
                    }
                    auto prompt_blocks = *prompt;
                    const auto request_id = message.id;
                    remember_prompt(request_id, session_id);
                    std::lock_guard lock(workers_mu);
                    reap_workers();
                    auto done = std::make_shared<std::atomic<bool>>(false);
                    std::thread thread([slot,
                                        done,
                                        prompt_blocks = std::move(prompt_blocks),
                                        request_id,
                                        &connection,
                                        &forget_prompt] {
                        struct Idle {
                            Slot* slot;
                            const RpcId* id;
                            decltype(forget_prompt)* forget;
                            std::atomic<bool>* done;
                            ~Idle() {
                                (*forget)(*id);
                                slot->busy.store(false, std::memory_order_release);
                                done->store(true, std::memory_order_release);
                            }
                        } idle{slot.get(), &request_id, &forget_prompt, done.get()};
                        try {
                            const auto reason = slot->session->prompt(prompt_blocks, connection);
                            auto result = core::utils::JsonValue::object();
                            result.set("stopReason", reason);
                            connection.reply(request_id, result);
                        } catch (const AuthRequired& ex) {
                            connection.reply_error(request_id, kAuthRequiredCode, ex.what());
                        } catch (const std::exception& ex) {
                            connection.reply_error(request_id, -32603, ex.what());
                        }
                    });
                    workers.push_back(Worker{std::move(thread), std::move(done)});
                    continue;
                }
                if (message.method == "session/close") {
                    // Idempotent: closing an unknown or closed session succeeds.
                    const auto session_id = text_field(message.params, "sessionId");
                    std::shared_ptr<Slot> slot;
                    {
                        std::lock_guard lock(sessions_mu);
                        if (const auto it = sessions.find(session_id); it != sessions.end()) {
                            slot = std::move(it->second);
                            sessions.erase(it);
                        }
                    }
                    if (slot) {
                        slot->session->cancel();
                        connection.cancel_session(session_id);
                    }
                    connection.reply(message.id, core::utils::JsonValue::object());
                    continue;
                }
                if (message.method == "session/list") {
                    if (!catalog) {
                        connection.reply(message.id, session_list_result({}));
                        continue;
                    }
                    std::optional<std::filesystem::path> cwd;
                    if (const auto filter = text_field(message.params, "cwd"); !filter.empty()) {
                        if (!std::filesystem::path(filter).is_absolute()) {
                            fail(message.id, -32602, "cwd must be an absolute path");
                            continue;
                        }
                        cwd = filter;
                    }
                    auto page = catalog->list(cwd, text_field(message.params, "cursor"));
                    if (!page) {
                        fail(message.id, -32602, page.error());
                        continue;
                    }
                    connection.reply(message.id, session_list_result(*page));
                    continue;
                }
                if (message.method == "session/delete") {
                    const auto session_id = text_field(message.params, "sessionId");
                    if (session_id.empty()) {
                        fail(message.id, -32602, "sessionId is required");
                        continue;
                    }
                    // An open session is closed first, so nothing saves it back.
                    std::shared_ptr<Slot> slot;
                    {
                        std::lock_guard lock(sessions_mu);
                        if (const auto it = sessions.find(session_id); it != sessions.end()) {
                            slot = std::move(it->second);
                            sessions.erase(it);
                        }
                    }
                    if (slot) {
                        slot->session->discard();
                        slot->session->cancel();
                        connection.cancel_session(session_id);
                    }
                    if (catalog) catalog->remove(session_id);
                    connection.reply(message.id, core::utils::JsonValue::object());
                    continue;
                }
                if (message.method == "session/set_mode" || message.method == "session/set_config_option"
                    || message.method == "session/set_model") {
                    const auto session_id = text_field(message.params, "sessionId");
                    const auto slot = find_session(session_id);
                    if (!slot) {
                        session_missing(message.id, session_id);
                        continue;
                    }
                    auto& session = *slot->session;
                    std::optional<std::string> error;
                    if (message.method == "session/set_mode") {
                        error = session.set_mode(text_field(message.params, "modeId"), connection);
                        if (!error) connection.reply(message.id, core::utils::JsonValue::object());
                    } else if (message.method == "session/set_model") {
                        // The older model picker; the same switch as the config option.
                        error = session.set_config(
                            "model", core::utils::JsonValue{text_field(message.params, "modelId")}, connection);
                        if (!error) connection.reply(message.id, core::utils::JsonValue::object());
                    } else {
                        const auto* value = object_field(message.params, "value");
                        error = session.set_config(
                            text_field(message.params, "configId"),
                            value ? *value : core::utils::JsonValue{},
                            connection);
                        if (!error) {
                            auto result = core::utils::JsonValue::object();
                            result.set("configOptions", session.config_state());
                            connection.reply(message.id, result);
                        }
                    }
                    if (error) fail(message.id, -32602, *error);
                    continue;
                }
                fail(message.id, -32601, "Method not found");
            } catch (const std::exception& ex) {
                fail(message.id, -32603, ex.what());
            }
        }
    };

    try {
        loop();
    } catch (const std::exception& ex) {
        core::logging::error("ACP server stopped: {}", ex.what());
    }

    {
        std::lock_guard lock(sessions_mu);
        for (auto& [id, slot] : sessions) {
            slot->session->cancel();
            connection.cancel_session(id);
        }
    }
    connection.cancel_all();
    std::lock_guard lock(workers_mu);
    for (auto& worker : workers) {
        if (worker.thread.joinable()) worker.thread.join();
    }
    return 0;
}

namespace {

/// The provider an editor's terminal sign-in should log in to: the default
/// one, when it has a sign-in at all. A local model needs none.
[[nodiscard]] std::string terminal_login_provider() {
    auto& config = core::config::ConfigManager::get_instance();
    const auto provider = config.get_config().default_provider;
    if (provider.empty()) return {};
    try {
        const auto known = core::auth::available_login_providers(config.get_config_dir());
        if (std::ranges::contains(known, provider)) return provider;
    } catch (const std::exception& ex) {
        core::logging::warn("ACP sign-in providers unavailable: {}", ex.what());
    }
    return {};
}

} // namespace

int run(const HostOptions& options) {
    core::logging::Logger::get_instance().use_stderr();
    StdioTransport transport(std::getenv("FILO_ACP_TRACE"));
    const auto catalog = make_store_catalog(core::session::SessionStore::default_sessions_dir());
    return serve(
        transport,
        [&](const OpenRequest& request) { return open_session(request, options); },
        ServeOptions{
            .catalog = catalog.get(),
            .login_provider = terminal_login_provider(),
        });
}

} // namespace exec::acp
