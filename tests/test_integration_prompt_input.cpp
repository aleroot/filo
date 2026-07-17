#include <catch2/catch_test_macros.hpp>

#include "tui/PromptInput.hpp"

#include <ftxui/component/component_options.hpp>
#include <ftxui/screen/screen.hpp>

#include <chrono>
#include <string>

using namespace ftxui;

namespace {

ftxui::Component make_input(std::string& content,
                            int& cursor,
                            bool multiline = false) {
    InputOption option;
    option.content = &content;
    option.cursor_position = &cursor;
    option.multiline = multiline;
    return tui::PromptInput(&content, "", option);
}

std::string make_big_document(int lines, const std::string& marker_line) {
    std::string content;
    content.reserve(static_cast<size_t>(lines) * 40);
    for (int i = 0; i < lines; ++i) {
        if (i == lines - 1) {
            content += marker_line;
        } else {
            content += "padding line number ";
            content += std::to_string(i);
        }
        content += '\n';
    }
    return content;
}

} // namespace

TEST_CASE("PromptInput - large document render cost is viewport-bounded",
          "[integration][prompt_input][perf]") {
    auto render_n = [](int lines, int iterations) {
        std::string content = make_big_document(lines, "end");
        int cursor = static_cast<int>(content.size());
        auto component = make_input(content, cursor, true);
        using clock = std::chrono::steady_clock;
        const auto start = clock::now();
        for (int i = 0; i < iterations; ++i) {
            Screen screen(90, 24);
            Render(screen, component->Render());
        }
        const auto end = clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count()
            / iterations;
    };

    // Virtualized rendering is ~constant in document size; the old O(n) path
    // was ~10x slower per frame at 5000 vs 500 lines. A 4x budget cleanly
    // separates the two while tolerating integration-environment variance.
    const double small = render_n(500, 60);
    const double large = render_n(5000, 60);
    REQUIRE(large < 4.0 * small);
}
