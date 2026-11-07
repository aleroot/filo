#include "core/cli/OptionValueGuard.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstddef>
#include <format>

namespace core::cli {
namespace {

/// Values CLI11 takes from the following arguments whatever they look like;
/// the same minimum App::_parse_arg gathers. Zero for a flag.
int required_values(const CLI::Option& option) {
    if (option.get_items_expected_max() == 0) return 0;
    return std::min(option.get_type_size_min(), option.get_items_expected_min());
}

/// The option @p argument would select if CLI11 parsed it as an option.
const CLI::Option* option_named_by(const CLI::App& app, const std::string& argument) {
    if (argument.starts_with("--")) {
        return app.get_option_no_throw(argument.substr(0, argument.find('=')));
    }
    if (argument.size() > 1 && argument.front() == '-') {
        return app.get_option_no_throw(argument.substr(0, 2));
    }
    return nullptr;
}

/// How to write a value for @p option so CLI11 cannot read it as an option.
std::string literal_value_form(const CLI::Option& option) {
    if (!option.get_lnames().empty()) return std::format("--{}=VALUE", option.get_lnames().front());
    return std::format("-{}VALUE", option.get_snames().front());
}

struct ValueOption {
    const CLI::Option* option = nullptr;
    std::string spelling;
};

/// The option in @p argument that takes its values from the next arguments,
/// if any. A value written inside the argument (`--name=value`, `-nVALUE`)
/// needs nothing more; combined short flags (`-yw`) are split as CLI11 does.
ValueOption option_awaiting_values(const CLI::App& app, const std::string& argument) {
    if (argument.starts_with("--")) {
        const auto equals = argument.find('=');
        // CLI11 treats an empty inline value as no value at all.
        if (equals != std::string::npos && equals + 1 < argument.size()) return {};
        std::string name = argument.substr(0, equals);
        const auto* option = app.get_option_no_throw(name);
        return {option, std::move(name)};
    }
    if (argument.size() < 2 || argument.front() != '-') return {};
    for (std::size_t position = 1; position < argument.size(); ++position) {
        std::string name{'-', argument[position]};
        const auto* option = app.get_option_no_throw(name);
        if (option == nullptr) return {};  // CLI11 reports the unknown option.
        if (option->get_items_expected_max() == 0) continue;  // A flag: the cluster goes on.
        if (position + 1 < argument.size()) return {};        // The rest is its value.
        return {option, std::move(name)};
    }
    return {};
}

} // namespace

std::optional<std::string> find_option_missing_value(
    const CLI::App& app, std::span<const std::string> args) {
    for (std::size_t index = 0; index < args.size(); ++index) {
        if (args[index] == "--") break;
        const auto [option, spelling] = option_awaiting_values(app, args[index]);
        if (option == nullptr) continue;
        const auto needed = static_cast<std::size_t>(required_values(*option));
        for (std::size_t offset = 1; offset <= needed && index + offset < args.size(); ++offset) {
            const std::string& value = args[index + offset];
            if (value == "--") {
                return std::format(
                    "{} needs a value, but the next argument is '--', the end of options",
                    spelling);
            }
            if (option_named_by(app, value) != nullptr) {
                return std::format(
                    "{} needs a value, but the next argument is the option {}. "
                    "To pass a value that starts with '-', write {}",
                    spelling, value, literal_value_form(*option));
            }
        }
        index += needed;
    }
    return std::nullopt;
}

} // namespace core::cli
