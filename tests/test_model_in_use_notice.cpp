#include <catch2/catch_test_macros.hpp>

#include "core/commands/ModelInUseNotice.hpp"

using core::commands::append_model_in_use_source;
using core::commands::format_context_size_label;
using core::commands::format_model_in_use_notice;
using core::commands::is_model_switch_success;
using core::commands::model_switch_mark;
using core::commands::result_mark;

TEST_CASE("context size labels use M and k", "[model][notice]") {
    CHECK(format_context_size_label(0).empty());
    CHECK(format_context_size_label(-1).empty());
    CHECK(format_context_size_label(1'000'000) == "1M context");
    CHECK(format_context_size_label(2'000'000) == "2M context");
    CHECK(format_context_size_label(1 << 20) == "1M context");
    CHECK(format_context_size_label(2 << 20) == "2M context");
    CHECK(format_context_size_label(200'000) == "200k context");
    CHECK(format_context_size_label(500000) == "500k context");
    CHECK(format_context_size_label(262'144) == "256k context");
    CHECK(format_context_size_label(16385) == "16385 context");
}

TEST_CASE("model in-use notice names the model, context, and effort", "[model][notice]") {
    CHECK(format_model_in_use_notice("Claude Opus 5.5", 1'000'000, "high")
          == "Using Claude Opus 5.5 · 1M context · high");
    CHECK(format_model_in_use_notice("Claude Opus 5.5", 1'000'000, "")
          == "Using Claude Opus 5.5 · 1M context");
    CHECK(format_model_in_use_notice("Claude Opus 5.5", 1'000'000, "auto")
          == "Using Claude Opus 5.5 · 1M context");
    CHECK(format_model_in_use_notice("Claude Opus 5.5", 1'000'000, "none")
          == "Using Claude Opus 5.5 · 1M context · off");
    CHECK(format_model_in_use_notice("Router · balanced", 0, "")
          == "Using Router · balanced");
    CHECK(format_model_in_use_notice("local · llama", 0, "max")
          == "Using local · llama · max");
}

TEST_CASE("model in-use source cites the defaults file and /model", "[model][notice]") {
    CHECK(append_model_in_use_source(
              "Using Claude Opus 5.5 · 1M context", true)
          == "Using Claude Opus 5.5 · 1M context · from model_defaults.json · /model");
    CHECK(append_model_in_use_source(
              "Using Claude Opus 5.5 · 1M context", false)
          == "Using Claude Opus 5.5 · 1M context · /model");

    const std::string with_hint = append_model_in_use_source(
        "Using Grok 4\n   Set XAI_API_KEY to start chatting.", true);
    CHECK(with_hint
          == "Using Grok 4 · from model_defaults.json · /model\n"
             "   Set XAI_API_KEY to start chatting.");

    CHECK(append_model_in_use_source("Switched to Manual mode", true)
          == "Switched to Manual mode");
}

TEST_CASE("model switch marks distinguish indications from failures", "[model][notice]") {
    CHECK(is_model_switch_success("Using Claude Opus 5.5"));
    CHECK(is_model_switch_success("Switched to grok-mini"));
    CHECK_FALSE(is_model_switch_success("Switching provider is not available"));
    CHECK_FALSE(is_model_switch_success("Failed to activate Manual mode"));

    CHECK(model_switch_mark("Using Claude Opus 5.5") == "ℹ  ");
    CHECK(model_switch_mark("Switched to grok-mini") == "✓  ");
    CHECK(model_switch_mark("Unknown model selector") == "✗  ");
}

TEST_CASE("result mark follows the caller's verdict", "[model][notice]") {
    CHECK(result_mark(true, "Set effort to high") == "✓  ");
    CHECK(result_mark(true, "Using Claude Opus 5.5") == "ℹ  ");
    CHECK(result_mark(false, "Using Claude Opus 5.5") == "✗  ");
    CHECK(result_mark(false, "Unknown model selector") == "✗  ");
}
