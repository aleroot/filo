#include <catch2/catch_test_macros.hpp>

#include "core/cli/OptionValueGuard.hpp"

#include <CLI/CLI.hpp>

#include <optional>
#include <string>
#include <vector>

namespace {

/// A parser shaped like filo's: value options of each arity, a flag, and an
/// option with only a short name.
struct Parser {
    CLI::App app{"guard test"};
    std::vector<std::string> work_dirs;
    std::vector<std::string> auth;
    std::vector<std::string> only;
    std::string config;
    std::string prompt;
    std::string resume;
    std::string import_source;
    std::string short_only;
    int port = 0;
    bool yolo = false;

    Parser() {
        app.add_option("-w,--work-dir", work_dirs)->expected(1)->take_all();
        app.add_option("--config", config);
        app.add_option("-p,--prompt", prompt);
        app.add_option("-r,--resume", resume)->expected(0, 1);
        app.add_option("--auth", auth)->expected(1, 2);
        app.add_option("--import", import_source);
        app.add_option("--only", only)->delimiter(',');
        app.add_option("--port", port);
        app.add_option("-x", short_only);
        app.add_flag("-y,--yolo", yolo);
    }

    std::optional<std::string> check(const std::vector<std::string>& args) const {
        return core::cli::find_option_missing_value(app, args);
    }
};

} // namespace

TEST_CASE("A value option followed by another option is reported before parsing",
          "[cli][option-values]") {
    const Parser parser;

    const auto problem = parser.check({"-w", "project", "-w", "--config", "work.json"});
    REQUIRE(problem);
    CHECK(*problem
          == "-w needs a value, but the next argument is the option --config. "
             "To pass a value that starts with '-', write --work-dir=VALUE");

    // Every value option is covered, not only -w, whatever its arity or type.
    CHECK(parser.check({"--config", "--yolo"}));
    CHECK(parser.check({"-p", "-y"}));
    CHECK(parser.check({"--auth", "--import", "bundle.json"}));
    CHECK(parser.check({"--only", "--config", "x"}));
    CHECK(parser.check({"--port", "--yolo"}));
    CHECK(parser.check({"--import", "--help"}));  // CLI11's own options count too
    CHECK(parser.check({"--config", "--work-dir=project"}));
    CHECK(parser.check({"--config", "-yw"}));
    CHECK(*parser.check({"-x", "--yolo"}) == "-x needs a value, but the next argument is the option --yolo. "
                                             "To pass a value that starts with '-', write -xVALUE");
    CHECK(*parser.check({"--config", "--"})
          == "--config needs a value, but the next argument is '--', the end of options");
}

TEST_CASE("Option values are read with CLI11's argument splitting",
          "[cli][option-values]") {
    const Parser parser;

    SECTION("combined short flags hand the last option the next argument") {
        CHECK(*parser.check({"-yw", "--config"})
              == "-w needs a value, but the next argument is the option --config. "
                 "To pass a value that starts with '-', write --work-dir=VALUE");
        CHECK(parser.check({"-yw", "project", "--config", "work.json"}) == std::nullopt);
    }
    SECTION("a value written inside the argument needs nothing more") {
        CHECK(parser.check({"--work-dir=--config", "--yolo"}) == std::nullopt);
        CHECK(parser.check({"-w--config", "--yolo"}) == std::nullopt);
        CHECK(parser.check({"-ywproject", "--yolo"}) == std::nullopt);
        CHECK(parser.check({"--prompt=-y", "-y"}) == std::nullopt);
    }
    SECTION("an empty inline value is no value, as CLI11 reads it") {
        CHECK(parser.check({"--config=", "--yolo"}));
        CHECK(parser.check({"--config=", "work.json"}) == std::nullopt);
    }
    SECTION("an optional value never swallows an option, so it is left to CLI11") {
        CHECK(parser.check({"--resume", "--yolo"}) == std::nullopt);
        CHECK(parser.check({"-r", "-w", "project"}) == std::nullopt);
        CHECK(parser.check({"--auth", "claude", "--config", "work.json"}) == std::nullopt);
    }
    SECTION("arguments after -- are not options") {
        CHECK(parser.check({"--", "-w", "--config"}) == std::nullopt);
    }
}

TEST_CASE("Values that only look like options are accepted",
          "[cli][option-values]") {
    const Parser parser;

    CHECK(parser.check({"--import", "-"}) == std::nullopt);          // stdin
    CHECK(parser.check({"--port", "-1"}) == std::nullopt);           // a negative number
    CHECK(parser.check({"-w", "-zone"}) == std::nullopt);            // -z is not an option here
    CHECK(parser.check({"-p", "- list the files"}) == std::nullopt);
    CHECK(parser.check({"-p", "--not-an-option"}) == std::nullopt);
    // ...but one that starts with a defined short option is reported (-p).
    CHECK(parser.check({"-w", "-project"}));
    // A value is consumed before the scan moves on, so it is not read as an option.
    CHECK(parser.check({"-p", "-q", "--yolo"}) == std::nullopt);
    // A missing last value is CLI11's own error.
    CHECK(parser.check({"--config"}) == std::nullopt);
}

TEST_CASE("The guard agrees with what CLI11 would have parsed",
          "[cli][option-values][regression]") {
    // Without the guard, CLI11 takes the option as the value and never sees
    // --config as an option: the overlay is silently lost.
    Parser parser;
    std::vector<std::string> args{"-w", "--config", "work.json"};
    REQUIRE(parser.check(args));

    parser.app.allow_extras();
    std::vector<std::string> reversed(args.rbegin(), args.rend());
    parser.app.parse(reversed);
    REQUIRE_FALSE(parser.work_dirs.empty());
    CHECK(parser.work_dirs.front() == "--config");
    CHECK(parser.config.empty());
}
