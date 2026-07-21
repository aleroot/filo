#include "CodeBlockRunServices.hpp"

#include <cerrno>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tui {
namespace {

[[nodiscard]] std::string serialize_transcript(
    const core::code::CodeRunResult& result) {
    return std::format(
        "Filo code-block execution\n"
        "exit_code: {}\n"
        "terminating_signal: {}\n"
        "duration_ms: {}\n"
        "captured_bytes: {}\n"
        "truncated: {}\n\n{}",
        result.exit_code,
        result.terminating_signal,
        result.duration.count(),
        result.output_bytes,
        result.output_truncated,
        result.output);
}

[[nodiscard]] std::expected<std::filesystem::path, std::string>
write_private_transcript(const std::filesystem::path& directory,
                         std::string_view contents) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        return std::unexpected("could not create transcript directory: " + error.message());
    }
    std::filesystem::permissions(
        directory,
        std::filesystem::perms::owner_all,
        std::filesystem::perm_options::replace,
        error);
    if (error) {
        return std::unexpected("could not protect transcript directory: " + error.message());
    }

#if !defined(_WIN32)
    std::string pattern = (directory / "output-XXXXXX.log").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = ::mkstemps(writable.data(), 4);
    if (descriptor < 0) {
        return std::unexpected(std::format(
            "could not create transcript: {}", std::strerror(errno)));
    }
    const std::filesystem::path path(writable.data());
    if (::fchmod(descriptor, S_IRUSR | S_IWUSR) != 0) {
        const std::string message = std::format(
            "could not protect transcript: {}", std::strerror(errno));
        ::close(descriptor);
        std::filesystem::remove(path, error);
        return std::unexpected(message);
    }
    while (!contents.empty()) {
        const auto written = ::write(descriptor, contents.data(), contents.size());
        if (written < 0) {
            if (errno == EINTR) continue;
            const std::string message = std::format(
                "could not write transcript: {}", std::strerror(errno));
            ::close(descriptor);
            std::filesystem::remove(path, error);
            return std::unexpected(message);
        }
        contents.remove_prefix(static_cast<std::size_t>(written));
    }
    if (::close(descriptor) != 0) {
        const std::string message = std::format(
            "could not finish transcript: {}", std::strerror(errno));
        std::filesystem::remove(path, error);
        return std::unexpected(message);
    }
    return path;
#else
    std::random_device random;
    for (int attempt = 0; attempt < 32; ++attempt) {
        const auto path = directory / std::format("output-{:08x}.log", random());
        if (std::filesystem::exists(path, error)) continue;
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) continue;
        output << contents;
        output.close();
        if (!output) {
            std::filesystem::remove(path, error);
            continue;
        }
        std::filesystem::permissions(
            path,
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace,
            error);
        if (!error) return path;
        std::filesystem::remove(path, error);
    }
    return std::unexpected("could not create a unique private transcript");
#endif
}

} // namespace

CodeBlockRunServices::CodeBlockRunServices(CodeBlockRunDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

std::filesystem::path select_code_run_script_directory(
    core::landrun::LandrunMode mode,
    const std::filesystem::path& runtime_root,
    const std::filesystem::path& process_temp_directory) {
    if (core::landrun::landrun_enabled(mode) && !runtime_root.empty()) {
        return runtime_root / "tmp" / "code-runs";
    }
    return process_temp_directory / "filo" / "code-runs";
}

std::expected<core::code::CodeRunResult, std::string>
CodeBlockRunServices::run(const core::code::ExecutionPlan& plan) {
    if (!dependencies_.runner
        || !dependencies_.working_directory
        || !dependencies_.landrun_policy
        || !dependencies_.with_restored_terminal) {
        return std::unexpected("code-run services are not fully configured");
    }

    std::expected<core::code::CodeRunResult, std::string> result =
        std::unexpected("code runner did not start");
    dependencies_.with_restored_terminal([&]() {
        std::cout << std::format(
            "\n── Filo · block {} · {} ──\n",
            plan.block.ordinal,
            plan.interpreter) << std::flush;
        result = dependencies_.runner->run(
            plan,
            dependencies_.working_directory(),
            dependencies_.script_directory,
            dependencies_.landrun_policy());
        if (result) {
            std::cout << std::format(
                "\n── {} in {:.2f}s ──\n",
                result->terminating_signal != 0
                    ? std::format("signal {}", result->terminating_signal)
                    : std::format("exit {}", result->exit_code),
                static_cast<double>(result->duration.count()) / 1000.0)
                      << std::flush;
        } else {
            std::cout << "\nfilo: " << result.error() << "\n" << std::flush;
        }
    });
    return result;
}

std::expected<void, std::string> CodeBlockRunServices::copy(std::string_view text) {
    if (!dependencies_.copy_to_clipboard) {
        return std::unexpected("clipboard integration is unavailable");
    }
    if (auto error = dependencies_.copy_to_clipboard(text); error.has_value()) {
        return std::unexpected(std::move(*error));
    }
    return {};
}

std::expected<std::filesystem::path, std::string>
CodeBlockRunServices::store_transcript(const core::code::CodeRunResult& result) {
    if (dependencies_.transcript_directory.empty()) {
        return std::unexpected("transcript directory is unavailable");
    }
    return write_private_transcript(
        dependencies_.transcript_directory,
        serialize_transcript(result));
}

} // namespace tui
