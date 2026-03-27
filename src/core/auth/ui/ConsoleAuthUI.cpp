#include "ConsoleAuthUI.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>
#if !defined(_WIN32)
#include <termios.h>
#include <unistd.h>
#endif

namespace core::auth::ui {

// Minimal ANSI colors
static const char* ANSI_BOLD = "\033[1m";
static const char* ANSI_RESET = "\033[0m";
static const char* ANSI_YELLOW = "\033[33m";
static const char* ANSI_GREEN = "\033[32m";
static const char* ANSI_RED = "\033[31m";
static const char* ANSI_BLUE = "\033[34m";

void ConsoleAuthUI::show_header(const std::string& title) {
    std::cout << "\n" << ANSI_BOLD << ANSI_BLUE << "=== " << title << " ===" << ANSI_RESET << "\n\n";
}

void ConsoleAuthUI::show_instructions(const std::string& text) {
    std::cout << text << "\n\n";
}

void ConsoleAuthUI::show_url(const std::string& url, const std::string& label) {
    std::cout << ANSI_BOLD << label << ANSI_RESET << "\n";
    std::cout << ANSI_BLUE << url << ANSI_RESET << "\n\n";
}

std::string ConsoleAuthUI::prompt_secret(const std::string& prompt_label) {
    std::cout << ANSI_BOLD << ANSI_YELLOW << prompt_label << ANSI_RESET << " ";
    std::cout.flush();
    std::string input;

#if !defined(_WIN32)
    termios oldt{};
    bool restore_echo = false;
    if (isatty(STDIN_FILENO) == 1 && tcgetattr(STDIN_FILENO, &oldt) == 0) {
        termios newt = oldt;
        newt.c_lflag &= static_cast<tcflag_t>(~ECHO);
        if (tcsetattr(STDIN_FILENO, TCSANOW, &newt) == 0) {
            restore_echo = true;
        }
    }
#endif

    std::getline(std::cin, input);

#if !defined(_WIN32)
    if (restore_echo) {
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        std::cout << "\n";
    }
#endif

    // Trim whitespace
    input.erase(input.begin(), std::find_if(input.begin(), input.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    input.erase(std::find_if(input.rbegin(), input.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), input.end());

    return input;
}

void ConsoleAuthUI::show_success(const std::string& message) {
    std::cout << "\n" << ANSI_BOLD << ANSI_GREEN << "Success: " << message << ANSI_RESET << "\n";
}

void ConsoleAuthUI::show_error(const std::string& message) {
    std::cerr << "\n" << ANSI_BOLD << ANSI_RED << "Error: " << message << ANSI_RESET << "\n";
}

} // namespace core::auth::ui
