#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "exec/Prompter.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/session/SessionData.hpp"
#include "core/session/SessionStore.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <mutex>
#include <optional>
#include <simdjson.h>
#include <sstream>
#include <string>
#include <vector>

using namespace Catch::Matchers;

namespace {

class MockProvider final : public core::llm::LLMProvider {
public:
    core::llm::ChatRequest last_request;
    core::llm::ProviderCapabilities caps{.supports_tool_calls = true, .is_local = false};

    std::function<void(const core::llm::ChatRequest&,
                       std::function<void(const core::llm::StreamChunk&)>)> on_stream;

    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        last_request = request;
        if (on_stream) {
            on_stream(request, callback);
            return;
        }
        callback(core::llm::StreamChunk::make_content("ok"));
        callback(core::llm::StreamChunk::make_final());
    }

    [[nodiscard]] core::llm::ProviderCapabilities capabilities() const override {
        return caps;
    }
};

class RotatingPrompterProvider final : public core::llm::LLMProvider {
public:
    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }

        const int call = calls_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (call <= 10) {
            set_last_usage(call <= 4 ? 4'000 : 26'000, call <= 4 ? 400 : 800);

            core::llm::ToolCall tool_call;
            tool_call.index = 0;
            tool_call.id = std::format("prompter-rotate-{}", call);
            tool_call.type = "function";
            tool_call.function.name = "get_current_time";
            tool_call.function.arguments = "{}";

            core::llm::StreamChunk chunk;
            chunk.tools = {tool_call};
            chunk.is_final = true;
            callback(chunk);
            return;
        }

        set_last_usage(3'000, 300);
        callback(core::llm::StreamChunk::make_content("rotation complete"));
        callback(core::llm::StreamChunk::make_final());
    }

    void reset_conversation_state() override {
        reset_calls_.fetch_add(1, std::memory_order_acq_rel);
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

    [[nodiscard]] int reset_count() const noexcept {
        return reset_calls_.load(std::memory_order_acquire);
    }

private:
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
    std::atomic<int> calls_{0};
    std::atomic<int> reset_calls_{0};
};

struct TempDir {
    std::filesystem::path path;

    explicit TempDir(std::string_view suffix)
        : path(std::filesystem::temp_directory_path() / std::string(suffix)) {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        std::filesystem::create_directories(path, ec);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

exec::prompter::RuntimeContext make_runtime(std::shared_ptr<core::llm::LLMProvider> provider) {
    exec::prompter::RuntimeContext runtime;
    runtime.provider = std::move(provider);
    runtime.provider_name = "mock";
    runtime.model_name = "mock-model";
    runtime.default_mode = "BUILD";
    return runtime;
}

core::session::SessionData make_session(std::string id,
                                        std::string created,
                                        std::string working_dir,
                                        std::string user_msg,
                                        std::string asst_msg,
                                        std::string last_active = {}) {
    core::session::SessionData data;
    data.session_id = std::move(id);
    data.created_at = std::move(created);
    data.last_active_at = last_active.empty() ? data.created_at : std::move(last_active);
    data.working_dir = std::move(working_dir);
    data.provider = "mock";
    data.model = "mock-model";
    data.mode = "BUILD";
    data.messages = {
        core::llm::Message{.role = "user", .content = std::move(user_msg)},
        core::llm::Message{.role = "assistant", .content = std::move(asst_msg)},
    };
    return data;
}

std::vector<std::string> extract_stream_event_session_ids(std::string_view output) {
    std::vector<std::string> session_ids;
    std::istringstream lines{std::string(output)};
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) {
            continue;
        }

        simdjson::dom::parser parser;
        simdjson::padded_string padded(line);
        simdjson::dom::element doc;
        REQUIRE(parser.parse(padded).get(doc) == simdjson::SUCCESS);

        std::string_view session_id;
        REQUIRE(doc["session_id"].get(session_id) == simdjson::SUCCESS);
        session_ids.emplace_back(session_id);
    }
    return session_ids;
}

} // namespace

TEST_CASE("prompter::should_run detects non-interactive triggers", "[prompter][cli]") {
    CHECK(exec::prompter::should_run(true, false, false, false, false, false, false));
    CHECK(exec::prompter::should_run(false, true, false, false, false, false, false));
    CHECK(exec::prompter::should_run(false, false, true, false, false, false, false));
    CHECK(exec::prompter::should_run(false, false, false, true, false, false, false));

    CHECK_FALSE(exec::prompter::should_run(false, false, false, false, false, false, false));
}

TEST_CASE("prompter text mode streams response", "[prompter][text]") {
    auto provider = std::make_shared<MockProvider>();
    provider->on_stream = [](const auto&, auto callback) {
        callback(core::llm::StreamChunk::make_content("hello"));
        callback(core::llm::StreamChunk::make_final());
    };

    exec::prompter::RunOptions opts;
    opts.prompt = "Say hi";
    opts.prompt_was_provided = true;

    exec::prompter::StreamInput input;

    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_text_test"};

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(diag.final_prompt == "Say hi");
    REQUIRE_THAT(diag.final_text_response, ContainsSubstring("hello"));
    REQUIRE_THAT(out.str(), ContainsSubstring("hello"));
    REQUIRE(err.str().empty());
    REQUIRE_FALSE(diag.session_id.empty());
}

TEST_CASE("prompter uses stdin when prompt is omitted", "[prompter][stdin]") {
    auto provider = std::make_shared<MockProvider>();
    provider->on_stream = [](const auto&, auto callback) {
        callback(core::llm::StreamChunk::make_content("processed"));
        callback(core::llm::StreamChunk::make_final());
    };

    exec::prompter::RunOptions opts;

    exec::prompter::StreamInput input;
    input.has_stdin_data = true;
    input.stdin_data = "from stdin\n";

    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_stdin_test"};

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(diag.final_prompt == "from stdin");

    bool saw_user_stdin = false;
    for (const auto& msg : provider->last_request.messages) {
        if (msg.role == "user" && msg.content == "from stdin") {
            saw_user_stdin = true;
            break;
        }
    }
    REQUIRE(saw_user_stdin);
}

TEST_CASE("prompter combines prompt and stdin context", "[prompter][stdin]") {
    auto provider = std::make_shared<MockProvider>();
    provider->on_stream = [](const auto&, auto callback) {
        callback(core::llm::StreamChunk::make_content("ok"));
        callback(core::llm::StreamChunk::make_final());
    };

    exec::prompter::RunOptions opts;
    opts.prompt = "Summarize";
    opts.prompt_was_provided = true;

    exec::prompter::StreamInput input;
    input.has_stdin_data = true;
    input.stdin_data = "README CONTENT";

    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_combine_test"};

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE_THAT(diag.final_prompt, ContainsSubstring("Summarize"));
    REQUIRE_THAT(diag.final_prompt, ContainsSubstring("README CONTENT"));
    REQUIRE_THAT(diag.final_prompt, ContainsSubstring("Input from stdin"));
}

TEST_CASE("prompter trust-tools blocks untrusted sensitive tools", "[prompter][trust]") {
    auto provider = std::make_shared<MockProvider>();
    TempDir tmp{"filo_prompter_trust_blocked_test"};
    const auto blocked_path = tmp.path / "blocked-write.txt";
    auto provider_call_count = std::make_shared<std::atomic<int>>(0);

    provider->on_stream = [blocked_path, provider_call_count](const auto&, auto callback) {
        const int call_count = ++(*provider_call_count);
        if (call_count == 1) {
            core::llm::ToolCall tool_call;
            tool_call.index = 0;
            tool_call.id = "tool-write-1";
            tool_call.type = "function";
            tool_call.function.name = "write_file";
            tool_call.function.arguments = std::format(
                R"({{"file_path":"{}","content":"blocked"}})",
                blocked_path.string());

            core::llm::StreamChunk chunk;
            chunk.tools = {tool_call};
            chunk.is_final = true;
            callback(chunk);
            return;
        }

        callback(core::llm::StreamChunk::make_content("done"));
        callback(core::llm::StreamChunk::make_final());
    };

    exec::prompter::RunOptions opts;
    opts.prompt = "Try writing a file";
    opts.prompt_was_provided = true;
    opts.trusted_tools = {"run_terminal_command"};

    exec::prompter::StreamInput input;

    std::ostringstream out;
    std::ostringstream err;
    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE_FALSE(std::filesystem::exists(blocked_path));
    REQUIRE(provider_call_count->load(std::memory_order_acquire) == 1);
    REQUIRE(diag.final_text_response.empty());
}

TEST_CASE("prompter yolo allows sensitive tools", "[prompter][trust]") {
    auto provider = std::make_shared<MockProvider>();
    TempDir tmp{"filo_prompter_trust_yolo_test"};
    const auto allowed_path = tmp.path / "allowed-write.txt";

    provider->on_stream = [allowed_path, call_count = 0](const auto&, auto callback) mutable {
        ++call_count;
        if (call_count == 1) {
            core::llm::ToolCall tool_call;
            tool_call.index = 0;
            tool_call.id = "tool-write-1";
            tool_call.type = "function";
            tool_call.function.name = "write_file";
            tool_call.function.arguments = std::format(
                R"({{"file_path":"{}","content":"trusted"}})",
                allowed_path.string());

            core::llm::StreamChunk chunk;
            chunk.tools = {tool_call};
            chunk.is_final = true;
            callback(chunk);
            return;
        }

        callback(core::llm::StreamChunk::make_content("done"));
        callback(core::llm::StreamChunk::make_final());
    };

    exec::prompter::RunOptions opts;
    opts.prompt = "Write a file";
    opts.prompt_was_provided = true;
    opts.yolo = true;

    exec::prompter::StreamInput input;

    std::ostringstream out;
    std::ostringstream err;
    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(std::filesystem::exists(allowed_path));

    std::ifstream ifs(allowed_path);
    REQUIRE(ifs.good());
    std::string file_text;
    std::getline(ifs, file_text);
    REQUIRE(file_text == "trusted");
}

TEST_CASE("prompter json mode emits structured array", "[prompter][json]") {
    auto provider = std::make_shared<MockProvider>();
    provider->on_stream = [](const auto&, auto callback) {
        callback(core::llm::StreamChunk::make_content("The capital is Paris."));
        callback(core::llm::StreamChunk::make_final());
    };

    exec::prompter::RunOptions opts;
    opts.prompt = "Capital of France?";
    opts.prompt_was_provided = true;
    opts.output_format = "json";
    opts.output_format_was_provided = true;

    exec::prompter::StreamInput input;

    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_json_test"};

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(err.str().empty());
    REQUIRE_THAT(out.str(), ContainsSubstring("\"type\":\"system\""));
    REQUIRE_THAT(out.str(), ContainsSubstring("\"subtype\":\"session_start\""));
    REQUIRE_THAT(out.str(), ContainsSubstring("\"type\":\"assistant\""));
    REQUIRE_THAT(out.str(), ContainsSubstring("\"type\":\"result\""));
    REQUIRE_THAT(out.str(), ContainsSubstring("Paris"));
}

TEST_CASE("prompter stream-json emits partial events", "[prompter][stream-json]") {
    auto provider = std::make_shared<MockProvider>();
    provider->on_stream = [](const auto&, auto callback) {
        callback(core::llm::StreamChunk::make_content("Hel"));
        callback(core::llm::StreamChunk::make_content("lo"));
        callback(core::llm::StreamChunk::make_final());
    };

    exec::prompter::RunOptions opts;
    opts.prompt = "Greet";
    opts.prompt_was_provided = true;
    opts.output_format = "stream-json";
    opts.output_format_was_provided = true;
    opts.include_partial_messages = true;

    exec::prompter::StreamInput input;

    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_stream_json_test"};

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(err.str().empty());
    REQUIRE_THAT(out.str(), ContainsSubstring("\"subtype\":\"content_delta\""));
    REQUIRE_THAT(out.str(), ContainsSubstring("\"type\":\"result\""));
}

TEST_CASE("prompter stream-json keeps a stable public session id across rotation",
          "[prompter][stream-json][rotation]") {
    auto provider = std::make_shared<RotatingPrompterProvider>();

    exec::prompter::RunOptions opts;
    opts.prompt = std::string(200'000, 'x');
    opts.prompt_was_provided = true;
    opts.output_format = "stream-json";
    opts.output_format_was_provided = true;
    opts.include_partial_messages = true;

    exec::prompter::StreamInput input;

    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_rotation_stream_json_test"};

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(err.str().empty());
    REQUIRE(provider->call_count() == 11);
    REQUIRE(provider->reset_count() == 1);

    const auto session_ids = extract_stream_event_session_ids(out.str());
    REQUIRE(session_ids.size() >= 4);
    CHECK(std::all_of(
        session_ids.begin(),
        session_ids.end(),
        [&](const std::string& value) { return value == session_ids.front(); }));
    CHECK(diag.session_id != session_ids.front());
}

TEST_CASE("prompter skips rotation when archival save fails", "[prompter][rotation][save]") {
    auto provider = std::make_shared<RotatingPrompterProvider>();

    exec::prompter::RunOptions opts;
    opts.prompt = std::string(200'000, 'x');
    opts.prompt_was_provided = true;

    exec::prompter::StreamInput input;

    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_rotation_save_failure_test"};
    const auto blocked_path = tmp.path / "sessions-blocked";
    {
        std::ofstream blocked(blocked_path);
        REQUIRE(blocked.good());
        blocked << "not a directory";
    }

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, blocked_path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(err.str().empty());
    REQUIRE(provider->call_count() == 11);
    REQUIRE(provider->reset_count() == 0);

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 11);

    bool saw_original_user_prompt = false;
    for (const auto& message : requests.back().messages) {
        if (message.role == "user" && message.content == *opts.prompt) {
            saw_original_user_prompt = true;
            break;
        }
    }
    CHECK(saw_original_user_prompt);
}

TEST_CASE("prompter validates incompatible partial-message flag", "[prompter][validation]") {
    auto provider = std::make_shared<MockProvider>();

    exec::prompter::RunOptions opts;
    opts.prompt = "hello";
    opts.prompt_was_provided = true;
    opts.include_partial_messages = true;
    opts.output_format = "text";

    exec::prompter::StreamInput input;
    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_partial_validation_test"};

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 2);
    REQUIRE_THAT(err.str(), ContainsSubstring("requires --output-format stream-json"));
}

TEST_CASE("prompter errors when no prompt and no stdin", "[prompter][validation]") {
    auto provider = std::make_shared<MockProvider>();

    exec::prompter::RunOptions opts;
    exec::prompter::StreamInput input;

    std::ostringstream out;
    std::ostringstream err;
    TempDir tmp{"filo_prompter_no_prompt_test"};

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 2);
    REQUIRE_THAT(err.str(), ContainsSubstring("requires a prompt"));
}

TEST_CASE("prompter --continue starts a fresh segment from project handoff", "[prompter][resume]") {
    auto provider = std::make_shared<MockProvider>();
    provider->on_stream = [](const auto&, auto callback) {
        callback(core::llm::StreamChunk::make_content("continued"));
        callback(core::llm::StreamChunk::make_final());
    };

    TempDir tmp{"filo_prompter_continue_test"};
    core::session::SessionStore store(tmp.path);

    const auto cwd = std::filesystem::current_path().string();

    auto wrong = make_session(
        "wrong123",
        "2026-03-27T12:00:00Z",
        "/tmp/not-this-project",
        "wrong user",
        "wrong answer",
        "2026-03-27T12:00:00Z");
    REQUIRE(store.save(wrong));

    auto right = make_session(
        "right123",
        "2026-03-26T12:00:00Z",
        cwd,
        "old user",
        "old answer",
        "2026-03-26T12:00:00Z");
    REQUIRE(store.save(right));

    exec::prompter::RunOptions opts;
    opts.prompt = "new task";
    opts.prompt_was_provided = true;
    opts.continue_last = true;

    exec::prompter::StreamInput input;
    std::ostringstream out;
    std::ostringstream err;

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(diag.used_resumed_session);
    REQUIRE(diag.resumed_session_id == "right123");
    REQUIRE(diag.session_id != "right123");

    bool saw_old_user = false;
    bool saw_new_user = false;
    bool saw_old_task_in_handoff = false;
    for (const auto& msg : provider->last_request.messages) {
        if (msg.role == "user" && msg.content == "old user") {
            saw_old_user = true;
        }
        if (msg.role == "user" && msg.content == "new task") {
            saw_new_user = true;
        }
        if (msg.role == "system" && msg.content.find("old user") != std::string::npos) {
            saw_old_task_in_handoff = true;
        }
    }
    REQUIRE_FALSE(saw_old_user);
    REQUIRE(saw_old_task_in_handoff);
    REQUIRE(saw_new_user);
}

TEST_CASE("prompter --resume restores the raw transcript", "[prompter][resume]") {
    auto provider = std::make_shared<MockProvider>();
    provider->on_stream = [](const auto&, auto callback) {
        callback(core::llm::StreamChunk::make_content("continued"));
        callback(core::llm::StreamChunk::make_final());
    };

    TempDir tmp{"filo_prompter_resume_raw_test"};
    core::session::SessionStore store(tmp.path);

    auto data = make_session(
        "resume123",
        "2026-03-26T12:00:00Z",
        std::filesystem::current_path().string(),
        "old user",
        "old answer",
        "2026-03-26T12:00:00Z");
    REQUIRE(store.save(data));

    exec::prompter::RunOptions opts;
    opts.prompt = "new task";
    opts.prompt_was_provided = true;
    opts.resume_session = "resume123";

    exec::prompter::StreamInput input;
    std::ostringstream out;
    std::ostringstream err;

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 0);
    REQUIRE(diag.used_resumed_session);
    REQUIRE(diag.session_id == "resume123");

    bool saw_old_user = false;
    bool saw_new_user = false;
    for (const auto& msg : provider->last_request.messages) {
        if (msg.role == "user" && msg.content == "old user") {
            saw_old_user = true;
        }
        if (msg.role == "user" && msg.content == "new task") {
            saw_new_user = true;
        }
    }
    REQUIRE(saw_old_user);
    REQUIRE(saw_new_user);
}

TEST_CASE("prompter --resume fails cleanly for unknown session", "[prompter][resume]") {
    auto provider = std::make_shared<MockProvider>();
    TempDir tmp{"filo_prompter_resume_missing_test"};

    exec::prompter::RunOptions opts;
    opts.prompt = "resume";
    opts.prompt_was_provided = true;
    opts.resume_session = "does-not-exist";

    exec::prompter::StreamInput input;
    std::ostringstream out;
    std::ostringstream err;

    const auto diag = exec::prompter::run_for_test(
        opts, make_runtime(provider), input, out, err, tmp.path);

    REQUIRE(diag.exit_code == 1);
    REQUIRE_THAT(err.str(), ContainsSubstring("Could not find session"));
}
