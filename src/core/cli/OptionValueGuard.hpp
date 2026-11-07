#pragma once

#include <optional>
#include <span>
#include <string>

namespace CLI {
class App;
} // namespace CLI

namespace core::cli {

/**
 * @brief Finds an option whose required value would be another option.
 *
 * CLI11 always gives an option its required values, even when the next
 * argument is another option: `-w --config work.json` uses "--config" as the
 * workspace and "work.json" as an unexpected extra argument, and so ignores
 * the overlay without saying so. This check runs on the command line before
 * CLI11 parses it, so a bad command line changes nothing.
 *
 * It reads the option table of @p app, so every value option is covered, and
 * it follows CLI11's argument splitting: `--name=value`, `-nVALUE`, combined
 * short flags such as `-yw`, and `--` as the end of options. A value is
 * rejected only when it is `--` or an option @p app defines; other values
 * that start with '-', such as `-` for stdin or a negative number, are
 * accepted. Writing `--name=value` always passes a value literally.
 *
 * @param app  The fully configured parser.
 * @param args The arguments after the program name.
 * @return A message naming the option and how to pass such a value, or
 *         nullopt when every required value is present.
 */
[[nodiscard]] std::optional<std::string> find_option_missing_value(
    const CLI::App& app, std::span<const std::string> args);

} // namespace core::cli
