#pragma once

#include "AuthUI.hpp"
#include <iostream>
#include <string>

namespace core::auth::ui {

class ConsoleAuthUI : public AuthUI {
public:
    void show_header(const std::string& title) override;
    void show_instructions(const std::string& text) override;
    void show_url(const std::string& url, const std::string& label) override;
    std::string prompt_secret(const std::string& prompt_label) override;
    void show_success(const std::string& message) override;
    void show_error(const std::string& message) override;
};

} // namespace core::auth::ui
