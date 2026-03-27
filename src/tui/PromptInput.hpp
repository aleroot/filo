#pragma once

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>

namespace tui {

ftxui::Component PromptInput(ftxui::StringRef content,
                             ftxui::StringRef placeholder,
                             ftxui::InputOption option = {});

} // namespace tui
