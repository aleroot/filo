#include "ClaudeOAuthFlow.hpp"
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

namespace core::auth {

namespace {

int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

} // namespace

ClaudeOAuthFlow::ClaudeOAuthFlow(std::shared_ptr<ui::AuthUI> ui)
    : ui_(std::move(ui)) {}

OAuthToken ClaudeOAuthFlow::login() {
    std::string auth_token;

    // 1. Try environment variable first (headless/automation support)
    const char* env_token = std::getenv("ANTHROPIC_AUTH_TOKEN");
    if (env_token && env_token[0] != '\0') {
        auth_token = env_token;
    } 
    // 2. Interactive flow if UI is available
    else if (ui_) {
        ui_->show_header("Claude Authentication Setup");
        
        ui_->show_instructions(
            "To use Claude with this CLI, you need to obtain your session key from the web browser.\n"
            "This allows you to use your existing Pro/Team plan without paying API fees."
        );

        ui_->show_url("https://claude.ai", "1. Open Claude in your browser and log in:");
        
        ui_->show_instructions(
            "2. Open Developer Tools (F12 or Right Click -> Inspect).\n"
            "3. Go to the 'Application' tab (Chrome/Edge) or 'Storage' tab (Firefox).\n"
            "4. Expand 'Cookies' and select 'https://claude.ai'.\n"
            "5. Find the cookie named 'sessionKey'.\n"
            "6. Copy its value (it starts with 'sk-ant-sid01-')."
        );

        auth_token = ui_->prompt_secret("Paste your sessionKey:");
        
        if (auth_token.empty()) {
             ui_->show_error("No token provided. Login aborted.");
             throw std::runtime_error("Login aborted by user.");
        }

        if (auth_token.find("sk-ant-sid01-") != 0) {
            ui_->show_error("Warning: The token does not start with 'sk-ant-sid01-'. It might be invalid.");
        }

        ui_->show_success("Token received!");
    }
    // 3. Fallback error
    else {
        throw std::runtime_error(
            "Missing ANTHROPIC_AUTH_TOKEN environment variable and no interactive UI available.\n"
            "Please export ANTHROPIC_AUTH_TOKEN or run in an interactive terminal."
        );
    }

    OAuthToken token;
    token.access_token = auth_token;
    token.token_type = "Bearer";
    // Anthropic session keys are long-lived but do expire. 
    // Set a reasonable default TTL (30 days) to prompt re-login eventually.
    token.expires_at = now_unix_seconds() + (60 * 60 * 24 * 30); 
    
    return token;
}

OAuthToken ClaudeOAuthFlow::refresh(std::string_view /*refresh_token*/) {
    // Session keys are refreshed automatically by the browser/server, 
    // but we can't do that easily here without simulating the full web client.
    // For now, simple re-login is the strategy.
    if (ui_) {
        ui_->show_error("Your Claude session has expired. Please log in again.");
    }
    throw std::runtime_error(
        "Claude token refresh is not supported. Please re-run `filo --login claude`."
    );
}

} // namespace core::auth
