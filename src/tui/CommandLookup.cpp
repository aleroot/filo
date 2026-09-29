#include "CommandLookup.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace tui {
namespace {

// Invokes `visit` for each non-empty field of `text`, stopping early when it
// returns true. Returns whether `visit` ever accepted a field.
template <typename Visit>
[[nodiscard]] bool any_field(std::string_view text, char separator, Visit&& visit) {
    for (std::size_t begin = 0; begin <= text.size();) {
        const std::size_t end = std::min(text.find(separator, begin), text.size());
        if (end > begin && visit(text.substr(begin, end - begin))) {
            return true;
        }
        begin = end + 1;
    }
    return false;
}

} // namespace

std::optional<std::vector<std::string>> shell_words(std::string_view command) {
    std::vector<std::string> words;
    std::string word;
    char quote = '\0';
    bool word_started = false;

    for (std::size_t index = 0; index < command.size(); ++index) {
        const char character = command[index];
        if (quote == '\'') {
            if (character == '\'') {
                quote = '\0';
            } else {
                word += character;
            }
            continue;
        }
        if (quote == '"') {
            if (character == '"') {
                quote = '\0';
            } else if (character == '\\') {
                if (++index == command.size()) {
                    return std::nullopt;
                }
                word += command[index];
            } else {
                word += character;
            }
            continue;
        }

        if (std::isspace(static_cast<unsigned char>(character))) {
            if (word_started) {
                words.push_back(std::move(word));
                word.clear();
                word_started = false;
            }
        } else if (character == '\'' || character == '"') {
            quote = character;
            word_started = true;
        } else if (character == '\\') {
            if (++index == command.size()) {
                return std::nullopt;
            }
            word += command[index];
            word_started = true;
        } else {
            word += character;
            word_started = true;
        }
    }

    if (quote != '\0') {
        return std::nullopt;
    }
    if (word_started) {
        words.push_back(std::move(word));
    }
    return words;
}

bool command_is_executable(std::string_view command) {
    const auto words = shell_words(command);
    if (!words || words->empty() || words->front().empty()) {
        return false;
    }

#if defined(_WIN32)
    // Executable extensions and PATH semantics differ enough that probing here
    // would be guesswork; the launch itself reports an unusable command.
    return true;
#else
    const std::string& program = words->front();
    if (program.contains('/')) {
        return ::access(program.c_str(), X_OK) == 0;
    }

    const char* path = std::getenv("PATH");
    if (path == nullptr) {
        return false;
    }
    return any_field(path, ':', [&program](std::string_view directory) {
        const std::filesystem::path candidate = std::filesystem::path(directory) / program;
        return ::access(candidate.c_str(), X_OK) == 0;
    });
#endif
}

int child_exit_code(int status) {
#if defined(_WIN32)
    return status;
#else
    if (status == -1) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return status;
#endif
}

} // namespace tui
