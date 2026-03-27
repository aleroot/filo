#pragma once

#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace core::auth::ui {

// Abstract interface for authentication UI interactions.
// Implementations can be TUI-based (interactive) or CLI-based (simple streams).
class AuthUI {
public:
    virtual ~AuthUI() = default;

    // Display a major header/title for the auth flow
    virtual void show_header(const std::string& title) = 0;

    // Display instructional text (multiline supported)
    virtual void show_instructions(const std::string& text) = 0;

    // Display a URL that the user should visit
    virtual void show_url(const std::string& url, const std::string& label = "Open this URL in your browser:") = 0;

    // Prompt the user for a secret (e.g., token, code) securely (masked input if possible)
    // Returns the trimmed input string.
    virtual std::string prompt_secret(const std::string& prompt_label) = 0;

    // Display a success message
    virtual void show_success(const std::string& message) = 0;

    // Display an error message
    virtual void show_error(const std::string& message) = 0;
};

} // namespace core::auth::ui
