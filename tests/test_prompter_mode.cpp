#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "exec/Prompter.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/session/SessionData.hpp"
#include "core/session/SessionStore.hpp"

#include <chrono>
#include <filesystem>
#include <future>
#include <optional>
#include <sstream>
#include <string>

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

exec::prompter::RuntimeContext make_runtime(std::shared_ptr<MockProvider> provider) {
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

TEST_CASE("prompter --continue resumes project-scoped session", "[prompter][resume]") {
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
