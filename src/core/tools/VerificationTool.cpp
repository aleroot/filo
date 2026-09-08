#include "VerificationTool.hpp"

#include "../context/SessionContext.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "ToolNames.hpp"
#include "shell/ShellUtils.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>
#include <ranges>
#include <simdjson.h>

namespace core::tools {

namespace {

constexpr std::size_t kMaxArguments = 128;
constexpr std::size_t kMaxArgumentBytes = 16 * 1024;

[[nodiscard]] std::string error_result(std::string_view message) {
  core::utils::JsonWriter writer;
  {
    auto object = writer.object();
    writer.kv_str("error", message);
  }
  return std::move(writer).take();
}

[[nodiscard]] bool safe_argv_token(std::string_view token) noexcept {
  if (token.empty())
    return true;
  return std::ranges::none_of(token, [](const unsigned char ch) {
    return ch == 0 || ch == '\n' || ch == '\r' || ch < 0x20;
  });
}

[[nodiscard]] bool is_shell_interpreter(std::string_view executable) {
  const auto separator = executable.find_last_of("/\\");
  if (separator != std::string_view::npos)
    executable.remove_prefix(separator + 1);
  std::string lower;
  lower.reserve(executable.size());
  for (const unsigned char ch : executable) {
    lower.push_back(static_cast<char>(std::tolower(ch)));
  }
  return lower == "sh" || lower == "bash" || lower == "zsh" ||
         lower == "fish" || lower == "cmd" || lower == "cmd.exe" ||
         lower == "powershell" || lower == "powershell.exe" ||
         lower == "pwsh" || lower == "pwsh.exe";
}

[[nodiscard]] std::vector<std::string>
parse_arguments(const simdjson::dom::object &object, std::string &error) {
  std::vector<std::string> arguments;
  simdjson::dom::array array;
  const auto status = object["arguments"].get(array);
  if (status == simdjson::NO_SUCH_FIELD)
    return arguments;
  if (status != simdjson::SUCCESS) {
    error = "arguments must be an array of strings";
    return arguments;
  }
  std::size_t bytes = 0;
  for (const auto item : array) {
    if (arguments.size() >= kMaxArguments) {
      error = "arguments exceeds the 128 item limit";
      return {};
    }
    std::string_view value;
    if (item.get(value) != simdjson::SUCCESS || !safe_argv_token(value)) {
      error = "arguments must contain only non-control string values";
      return {};
    }
    bytes += value.size();
    if (bytes > kMaxArgumentBytes) {
      error = "arguments exceeds the 16 KiB limit";
      return {};
    }
    arguments.emplace_back(value);
  }
  return arguments;
}

[[nodiscard]] std::string
shell_command(const core::verification::CommandSpec &command) {
  const auto quote = [](std::string_view token) {
    return std::format("'{}'", detail::shell_single_quote(token));
  };
  std::string result = quote(command.executable);
  for (const auto &argument : command.arguments) {
    result += " ";
    result += quote(argument);
  }
  return result;
}

[[nodiscard]] std::string
shell_arguments_json(const core::verification::CommandSpec &command,
                     const std::filesystem::path &working_directory) {
  core::utils::JsonWriter writer;
  {
    auto object = writer.object();
    writer.kv_str("command", shell_command(command)).comma();
    writer.kv_str("working_dir", working_directory.string()).comma();
    writer.kv_num("timeout_seconds", command.timeout_seconds);
  }
  return std::move(writer).take();
}

} // namespace

VerificationTool::VerificationTool() = default;

VerificationTool::VerificationTool(
    std::unique_ptr<shell::IShellExecutor> executor)
    : shell_(std::move(executor)) {}

ToolDefinition VerificationTool::get_definition() const {
  return {
      .name = std::string(names::kRunVerification),
      .title = "Run Verification",
      .description = "Run a deterministic build, test, lint, type-check, or "
                     "format check and return "
                     "a typed verification receipt. Use recipe_id only for "
                     "checks listed in a repository verification catalog when "
                     "the environment provides one; otherwise provide kind, "
                     "executable, and an argv arguments array. This tool does "
                     "not invoke a caller-supplied "
                     "shell string, so shell operators cannot mask failures.",
      .parameters =
          {
              {"recipe_id", "string",
               "Repository recipe id shown in the verification catalog.",
               false},
              {"kind", "string",
               "Required for a custom check: build, test, lint, typecheck, "
               "format, workflow, or custom.",
               false,
               R"({"type":"string","enum":["build","test","lint","typecheck","format","workflow","custom"]})"},
              {"executable", "string",
               "Required for a custom check. A single executable token, not a "
               "shell command.",
               false},
              {"arguments",
               "array",
               "Arguments for a custom executable. Each item is passed as one "
               "quoted argv token.",
               false,
               {},
               R"({"type":"string"})"},
              {"working_dir", "string",
               "Custom check directory relative to the workspace root; "
               "defaults to the root.",
               false},
              {"timeout_seconds", "integer",
               "Custom check timeout from 1 to 3600 seconds; defaults to 600.",
               false},
          },
      .output_schema =
          R"({"type":"object","properties":{"output":{"type":"string"},"exit_code":{"type":"integer"},"verification_receipt":{"type":"object","properties":{"schema_version":{"type":"integer"},"receipt_id":{"type":"string"},"recipe_id":{"type":"string"},"kind":{"type":"string"},"source":{"type":"string"},"command":{"type":"string"},"working_directory":{"type":"string"},"exit_code":{"type":"integer"},"duration_ms":{"type":"integer"}},"required":["schema_version","receipt_id","recipe_id","kind","source","command","working_directory","exit_code","duration_ms"],"additionalProperties":false}},"required":["output","exit_code","verification_receipt"],"additionalProperties":false})",
      .annotations =
          {
              .destructive_hint = true,
              .open_world_hint = true,
          },
      .trusted_verification_receipts = true,
  };
}

std::string
VerificationTool::execute(const std::string &json_args,
                          const core::context::SessionContext &context) {
  return execute_impl(json_args,
                      ToolInvocationContext{.session_context = context});
}

std::string VerificationTool::execute(const std::string &json_args,
                                      const ToolInvocationContext &invocation) {
  return execute_impl(json_args, invocation);
}

void VerificationTool::clear_session_state(std::string_view session_id) {
  shell_.clear_session_state(session_id);
}

std::string
VerificationTool::make_receipt_id(std::string_view tool_call_id) {
  if (!tool_call_id.empty())
    return std::string(tool_call_id);
  return std::format(
      "verification-{}",
      receipt_sequence_.fetch_add(1, std::memory_order_relaxed) + 1);
}

std::string
VerificationTool::execute_impl(const std::string &json_args,
                               const ToolInvocationContext &invocation) {
  simdjson::dom::parser parser;
  simdjson::dom::element document;
  if (parser.parse(json_args).get(document) != simdjson::SUCCESS) {
    return error_result("Invalid JSON arguments provided to run_verification.");
  }
  simdjson::dom::object object;
  if (document.get(object) != simdjson::SUCCESS) {
    return error_result("run_verification arguments must be an object.");
  }

  const auto &workspace = invocation.session_context.workspace_view();
  const auto project_root = workspace.primary();
  if (project_root.empty()) {
    return error_result("run_verification requires a workspace root.");
  }

  core::verification::Recipe recipe;
  const std::string receipt_id = make_receipt_id(invocation.tool_call_id);
  const std::string recipe_id =
      core::utils::json::string_field(object, "recipe_id");
  if (!recipe_id.empty()) {
    const auto catalog = catalog_.discover(project_root);
    const auto found = std::ranges::find(catalog.recipes, recipe_id,
                                         &core::verification::Recipe::id);
    if (found == catalog.recipes.end()) {
      return error_result(std::format("Unknown verification recipe '{}'. "
                                      "Refresh the repository catalog or use a "
                                      "typed custom check.",
                                      recipe_id));
    }
    recipe = *found;
  } else {
    const auto kind = core::verification::kind_from_string(
        core::utils::json::string_field(object, "kind"));
    const std::string executable =
        core::utils::json::string_field(object, "executable");
    if (!kind.has_value() || executable.empty()) {
      return error_result(
          "A custom verification requires both kind and executable.");
    }
    if (!safe_argv_token(executable) || executable.size() > 4096) {
      return error_result(
          "Custom verification executable contains invalid characters.");
    }
    if (is_shell_interpreter(executable)) {
      return error_result("Custom verification cannot invoke a shell "
                          "interpreter. Define a repository "
                          "recipe or invoke the build/test executable directly "
                          "with an argv array.");
    }
    std::string argument_error;
    auto arguments = parse_arguments(object, argument_error);
    if (!argument_error.empty())
      return error_result(argument_error);

    int timeout = core::utils::json::int_field(object, "timeout_seconds", 600);
    if (timeout <= 0 || timeout > 3600) {
      return error_result("timeout_seconds must be between 1 and 3600.");
    }
    recipe = {
        .id = std::format("custom:{}", receipt_id),
        .display_name = "Agent-proposed verification",
        .kind = *kind,
        .source = core::verification::Source::AgentProposed,
        .command =
            {
                .executable = executable,
                .arguments = std::move(arguments),
                .working_directory =
                    core::utils::json::string_field(object, "working_dir"),
                .timeout_seconds = timeout,
            },
    };
  }

  std::filesystem::path working_directory = recipe.command.working_directory;
  if (working_directory.empty())
    working_directory = project_root;
  else if (working_directory.is_relative())
    working_directory = project_root / working_directory;

  const auto started = std::chrono::steady_clock::now();
  const std::string raw_result = shell_.execute(
      shell_arguments_json(recipe.command, working_directory), invocation);
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  if (core::utils::json::first_string_field(raw_result, {"error"})
          .has_value()) {
    return raw_result;
  }
  const auto exit_code = core::utils::json::first_int64_field(
      raw_result, {"exit_code", "exitCode"});
  if (!exit_code.has_value()) {
    return error_result("Verification executor returned no exit code.");
  }
  const std::string output =
      core::utils::json::string_field(raw_result, "output");
  core::utils::JsonWriter writer(output.size() + 512);
  {
    auto root = writer.object();
    writer.kv_str("output", output).comma();
    writer.kv_num("exit_code", *exit_code).comma();
    writer.key("verification_receipt");
    {
      auto receipt = writer.object();
      writer.kv_num("schema_version", core::verification::kReceiptSchemaVersion)
          .comma();
      writer.kv_str("receipt_id", receipt_id).comma();
      writer.kv_str("recipe_id", recipe.id).comma();
      writer.kv_str("kind", core::verification::to_string(recipe.kind)).comma();
      writer.kv_str("source", core::verification::to_string(recipe.source))
          .comma();
      writer
          .kv_str("command",
                  core::verification::display_command(recipe.command))
          .comma();
      writer.kv_str("working_directory", working_directory.string()).comma();
      writer.kv_num("exit_code", *exit_code).comma();
      writer.kv_num("duration_ms", duration.count());
    }
  }
  return std::move(writer).take();
}

} // namespace core::tools
