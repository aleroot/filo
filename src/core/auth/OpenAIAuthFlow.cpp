#include "OpenAIAuthFlow.hpp"
#include <chrono>
#include <cstdlib>
#include <stdexcept>

namespace core::auth {

namespace {

int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

} // namespace

OpenAIAuthFlow::OpenAIAuthFlow(std::shared_ptr<ui::AuthUI> ui)
    : ui_(std::move(ui)) {}

OAuthToken OpenAIAuthFlow::login() {
    std::string api_key;

    // 1. Environment variable first (headless / CI)
    if (const char* env = std::getenv("OPENAI_API_KEY"); env && env[0] != '\0') {
        api_key = env;
    }
    // 2. Interactive prompt
    else if (ui_) {
        ui_->show_header("OpenAI API Key Setup");

        ui_->show_instructions(
            "To use OpenAI models, you need an API key from your OpenAI account.\n"
            "API keys start with 'sk-' and can be created at platform.openai.com/api-keys."
        );

        ui_->show_url("https://platform.openai.com/api-keys",
                      "1. Create or copy an API key from:");

        api_key = ui_->prompt_secret("Paste your OpenAI API key:");

        if (api_key.empty()) {
            ui_->show_error("No key provided. Login aborted.");
            throw std::runtime_error("Login aborted by user.");
        }

        if (!api_key.starts_with("sk-")) {
            ui_->show_error("Warning: key does not start with 'sk-'. It may be invalid.");
        }

        ui_->show_success("API key saved.");
    }
    // 3. No key and no UI
    else {
        throw std::runtime_error(
            "Missing OPENAI_API_KEY environment variable and no interactive UI available.\n"
            "Please export OPENAI_API_KEY or run `filo --login openai` in a terminal."
        );
    }

    OAuthToken token;
    token.access_token = std::move(api_key);
    token.token_type   = "Bearer";
    // API keys don't expire on their own; set a 1-year TTL to eventually
    // prompt re-entry when the user rotates the key.
    token.expires_at   = now_unix_seconds() + (60LL * 60 * 24 * 365);
    return token;
}

OAuthToken OpenAIAuthFlow::refresh(std::string_view /*refresh_token*/) {
    if (ui_) {
        ui_->show_error("Your OpenAI API key needs to be re-entered.");
    }
    throw std::runtime_error(
        "OpenAI API key refresh is not supported. "
        "Please run `filo --login openai` to enter a new key."
    );
}

} // namespace core::auth
