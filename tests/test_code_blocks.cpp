#include <catch2/catch_test_macros.hpp>

#include "core/code/CodeBlock.hpp"
#include "core/code/InteractiveCodeRunner.hpp"
#include "tui/CodeBlockRunner.hpp"
#include "tui/CodeBlockRunServices.hpp"
#include "tui/Conversation.hpp"

#include <filesystem>
#include <format>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace core::code;

namespace {

class FakeCodeBlockRunServices final : public tui::ICodeBlockRunServices {
public:
    std::expected<CodeRunResult, std::string> run(
        const ExecutionPlan& plan) override {
        ++run_count;
        last_interpreter = plan.interpreter;
        return CodeRunResult{.exit_code = 0, .output = "done\n"};
    }

    std::expected<void, std::string> copy(std::string_view text) override {
        copied = text;
        return {};
    }

    std::expected<std::filesystem::path, std::string> store_transcript(
        const CodeRunResult&) override {
        return std::filesystem::path("/tmp/filo-test-output.log");
    }

    int run_count = 0;
    std::string last_interpreter;
    std::string copied;
};

} // namespace

TEST_CASE("Fenced code extraction follows Markdown fence boundaries", "[code_blocks]") {
    const std::string markdown =
        "Before\n"
        "```bash title=demo\n"
        "printf 'one'\n"
        "```\n"
        "~~~python\r\n"
        "print('two')\r\n"
        "~~~~\r\n"
        "```broken`info\n"
        "ignored\n";

    const auto blocks = extract_fenced_code_blocks(markdown);
    REQUIRE(blocks.size() == 2);
    CHECK(blocks[0].ordinal == 1);
    CHECK(blocks[0].language == "bash");
    CHECK(blocks[0].info == "bash title=demo");
    CHECK(blocks[0].source == "printf 'one'\n");
    CHECK(blocks[0].first_line == 3);
    CHECK(blocks[0].last_line == 3);
    CHECK(blocks[1].language == "python");
    CHECK(blocks[1].source == "print('two')\n");
}

TEST_CASE("Unclosed fences are not offered for execution", "[code_blocks]") {
    CHECK(extract_fenced_code_blocks("```sh\necho incomplete\n").empty());
}

TEST_CASE("Code execution reads canonical assistant bytes instead of display markers",
          "[code_blocks][tui]") {
    auto message = tui::make_assistant_message(
        "```sh\nprintf 'hel\n💡\nlo'\n```\n", "", false);
    message.assistant_source_text = "```sh\nprintf 'hello'\n```\n";

    const auto source = tui::latest_completed_assistant_source({message});
    const auto blocks = extract_fenced_code_blocks(source);
    REQUIRE(blocks.size() == 1);
    CHECK(blocks.front().source == "printf 'hello'\n");
}

TEST_CASE("Execution planning is explicit and language aware", "[code_blocks]") {
    FencedCodeBlock shell{
        .ordinal = 1,
        .language = "console",
        .source = "$ printf 'ready\\n'\n$ printf 'set\\n'\n",
    };
    auto plan = plan_execution(std::move(shell));
    REQUIRE(plan.has_value());
    CHECK(plan->interpreter == "bash");
    CHECK(plan->interpreter_path.is_absolute());
    CHECK(plan->script_extension == "sh");
    CHECK(plan->prepared_source == "printf 'ready\\n'\nprintf 'set\\n'\n");

    auto unsupported = plan_execution(FencedCodeBlock{
        .ordinal = 2,
        .language = "cpp",
        .source = "int main() {}",
    });
    REQUIRE_FALSE(unsupported.has_value());
    CHECK(unsupported.error().reason.find("not executable") != std::string::npos);
}

TEST_CASE("Terminal transcript sanitization removes escape and control sequences",
          "[code_blocks]") {
    const std::string raw = "\x1b[31mred\x1b[0m\rnext\x01\n"
                            "\x1b]0;secret title\x07visible";
    CHECK(sanitize_terminal_output(raw) == "red\nnext\nvisible");
}

TEST_CASE("Code block runner requires selection and confirmation", "[code_blocks][tui]") {
    FakeCodeBlockRunServices services;
    tui::CodeBlockRunnerController controller(services);
    REQUIRE(controller.open(
        "```sh\nprintf ok\n```\n```cpp\nint main() {}\n```").has_value());
    REQUIRE(controller.state().items.size() == 2);
    CHECK(controller.state().mode == tui::CodeBlockRunnerMode::Select);
    CHECK(controller.state().items[0].plan.has_value());
    CHECK_FALSE(controller.state().items[1].plan.has_value());

    auto selected = controller.handle(ftxui::Event::Return, false);
    CHECK(selected.handled);
    CHECK(controller.state().mode == tui::CodeBlockRunnerMode::Confirm);
    CHECK(services.run_count == 0);

    auto confirmed = controller.handle(ftxui::Event::Return, false);
    CHECK(confirmed.handled);
    CHECK(services.run_count == 1);
    CHECK(services.last_interpreter == "bash");
    CHECK(controller.state().mode == tui::CodeBlockRunnerMode::Complete);
}

TEST_CASE("POSIX code runner captures output and exit status", "[code_blocks][runner]") {
    auto plan = plan_execution(FencedCodeBlock{
        .ordinal = 1,
        .language = "sh",
        .source = "printf 'filo-code-runner\\n'\nexit 7\n",
    });
    REQUIRE(plan.has_value());

    auto runner = make_interactive_code_runner();
    auto result = runner->run(
        *plan,
        std::filesystem::current_path(),
        std::filesystem::temp_directory_path() / "filo-code-runner-tests",
        core::landrun::LandrunPolicy{});
    REQUIRE(result.has_value());
    CHECK(result->exit_code == 7);
    CHECK(result->terminating_signal == 0);
    CHECK(result->output == "filo-code-runner\n");
    CHECK(result->output_bytes == std::string_view("filo-code-runner\n").size());
    CHECK_FALSE(result->output_truncated);
}

TEST_CASE("Sandboxed code runs select the private Landrun temp root",
          "[code_blocks][runner]") {
    const std::filesystem::path runtime = "/private/runtime";
    const std::filesystem::path process_temp = "/excluded/process-temp";
    CHECK(tui::select_code_run_script_directory(
              core::landrun::LandrunMode::workspace_write,
              runtime,
              process_temp)
          == runtime / "tmp" / "code-runs");
    CHECK(tui::select_code_run_script_directory(
              core::landrun::LandrunMode::off,
              runtime,
              process_temp)
          == process_temp / "filo" / "code-runs");
}

#if !defined(_WIN32)
TEST_CASE("Code runner closes unrelated inherited descriptors before exec",
          "[code_blocks][runner]") {
    int inherited_pipe[2]{};
    REQUIRE(::pipe(inherited_pipe) == 0);
    const auto source = std::format(
        "if ( : >&{} ) 2>/dev/null; then printf leaked; else printf closed; fi\n",
        inherited_pipe[1]);
    auto plan = plan_execution(FencedCodeBlock{
        .ordinal = 1,
        .language = "bash",
        .source = source,
    });
    REQUIRE(plan.has_value());

    auto runner = make_interactive_code_runner();
    auto result = runner->run(
        *plan,
        std::filesystem::current_path(),
        std::filesystem::temp_directory_path() / "filo-code-runner-tests",
        core::landrun::LandrunPolicy{});
    ::close(inherited_pipe[0]);
    ::close(inherited_pipe[1]);

    REQUIRE(result.has_value());
    CHECK(result->output == "closed");
}
#endif
