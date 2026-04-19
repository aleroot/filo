#pragma once

#include "core/llm/LLMProvider.hpp"
#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace exec::prompter {

struct RunOptions {
    std::optional<std::string> prompt;
    bool prompt_was_provided = false;

    std::string output_format = "text";
    bool output_format_was_provided = false;

    std::string input_format = "text";
    bool input_format_was_provided = false;

    bool include_partial_messages = false;
    bool continue_last = false;
    bool yolo = false;
    std::vector<std::string> trusted_tools;

    // When present and empty, resume the most recent session.
    std::optional<std::string> resume_session;
};

struct RuntimeContext {
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string provider_name;
    std::string model_name;
    std::string default_mode = "BUILD";
};

struct StreamInput {
    bool has_stdin_data = false;
    std::string stdin_data;
};

struct RunDiagnostics {
    int exit_code = 0;
    std::string final_prompt;
    std::string final_text_response;
    std::string rendered_output;

    std::string session_id;
    std::string session_file_path;
    bool used_resumed_session = false;
    std::string resumed_session_id;
};

[[nodiscard]] bool should_run(bool explicit_prompter,
                              bool prompt_provided,
                              bool has_stdin_data,
                              bool output_format_provided,
                              bool input_format_provided,
                              bool include_partial_messages,
                              bool continue_last);

int run(const RunOptions& options);

RunDiagnostics run_for_test(const RunOptions& options,
                            const RuntimeContext& runtime,
                            const StreamInput& input,
                            std::ostream& out,
                            std::ostream& err,
                            std::optional<std::filesystem::path> session_dir_override = std::nullopt);

} // namespace exec::prompter
