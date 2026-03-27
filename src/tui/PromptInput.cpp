#include "PromptInput.hpp"
#include "StringUtils.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <ftxui/component/captured_mouse.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/mouse.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/box.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/string.hpp>

using namespace ftxui;

namespace tui {

namespace {

// ── UTF-8 glyph navigation ────────────────────────────────────────────────────

size_t glyph_next(const std::string& input, size_t iter) {
    if (iter >= input.size()) {
        return input.size();
    }

    ++iter;
    while (iter < input.size()
           && (static_cast<unsigned char>(input[iter]) & 0b1100'0000) == 0b1000'0000) {
        ++iter;
    }
    return iter;
}

size_t glyph_previous(const std::string& input, size_t iter) {
    if (iter == 0) {
        return 0;
    }

    --iter;
    while (iter > 0
           && (static_cast<unsigned char>(input[iter]) & 0b1100'0000) == 0b1000'0000) {
        --iter;
    }
    return iter;
}

size_t glyph_width(const std::string& input, size_t iter) {
    const size_t next = glyph_next(input, iter);
    if (next <= iter) {
        return 0;
    }
    return static_cast<size_t>(string_width(input.substr(iter, next - iter)));
}

bool is_word_character(const std::string& input, size_t iter) {
    if (iter >= input.size()) {
        return false;
    }
    const auto ch = static_cast<unsigned char>(input[iter]);
    return std::isalnum(ch) != 0 || ch == '_';
}

Element place_cursor_at_origin(Element child, Screen::Cursor::Shape shape) {
    class CursorAtOrigin final : public Node {
    public:
        CursorAtOrigin(Element child, Screen::Cursor::Shape shape)
            : Node(unpack(std::move(child))), shape_(shape) {}

        void ComputeRequirement() override {
            Node::ComputeRequirement();
            requirement_ = children_[0]->requirement();
        }

        void SetBox(Box box) override {
            Node::SetBox(box);
            children_[0]->SetBox(box);
        }

        void Render(Screen& screen) override {
            children_[0]->Render(screen);
            screen.SetCursor(Screen::Cursor{
                box_.x_min,
                box_.y_min,
                shape_,
            });
        }

    private:
        Screen::Cursor::Shape shape_;
    };

    return std::make_shared<CursorAtOrigin>(std::move(child), shape);
}

// ── PromptInputBase component ─────────────────────────────────────────────────

class PromptInputBase : public ComponentBase, public InputOption {
public:
    explicit PromptInputBase(InputOption option) : InputOption(std::move(option)) {}

private:
    // ── Rendering ─────────────────────────────────────────────────────────────

    /// Returns a password-masked version of the text, counting visual glyphs
    /// rather than raw bytes so that multi-byte characters each produce one '*'.
    Element MaskedText(const std::string& input) {
        std::string out;
        out.reserve(input.size());
        for (size_t i = 0; i < input.size(); i = glyph_next(input, i)) {
            out += '*';
        }
        return text(out);
    }

    Element Text(const std::string& input) {
        return password() ? MaskedText(input) : text(input);
    }

    /// Current cursor position expressed as (line index, char-index within line).
    struct CursorCoord {
        int line = 0;
        int char_index = 0;
    };

    CursorCoord cursor_coord(const std::vector<std::string>& lines) const {
        CursorCoord coord;
        int remaining = cursor_position();
        for (const auto& line : lines) {
            if (remaining <= static_cast<int>(line.size())) {
                coord.char_index = remaining;
                break;
            }
            remaining -= static_cast<int>(line.size()) + 1;
            ++coord.line;
        }
        return coord;
    }

    Element OnRender() override {
        const bool is_focused = Focused();
        const auto apply_focus = [&](Element child) {
            if (is_focused || hovered_) {
                return focusCursorBlock(std::move(child));
            }
            return nothing(std::move(child));
        };
        auto transform_func = transform ? transform : InputOption::Default().transform;

        if (content->empty()) {
            auto element = transform_func({
                text(placeholder()) | xflex | frame,
                hovered_,
                is_focused,
                true
            });

            if (is_focused) {
                element = place_cursor_at_origin(std::move(element), Screen::Cursor::Block);
            }

            return std::move(element) | reflect(box_);
        }

        const auto lines = split_lines(*content);
        cursor_position() = std::clamp(cursor_position(), 0, static_cast<int>(content->size()));
        const auto [cursor_line, cursor_char_index] = cursor_coord(lines);

        Elements elements;
        elements.reserve(lines.size());

        if (lines.empty()) {
            elements.push_back(apply_focus(text("")));
        }

        for (size_t i = 0; i < lines.size(); ++i) {
            const auto& line = lines[i];
            if (static_cast<int>(i) != cursor_line) {
                elements.push_back(Text(line));
                continue;
            }

            if (cursor_char_index >= static_cast<int>(line.size())) {
                elements.push_back(hbox(Elements{
                    Text(line),
                    apply_focus(text(" ")) | reflect(cursor_box_),
                }) | xflex);
                continue;
            }

            const int glyph_start = cursor_char_index;
            const int glyph_end = static_cast<int>(glyph_next(line, static_cast<size_t>(glyph_start)));
            const std::string before = line.substr(0, glyph_start);
            const std::string at     = line.substr(glyph_start, glyph_end - glyph_start);
            const std::string after  = line.substr(glyph_end);

            elements.push_back(hbox(Elements{
                Text(before),
                apply_focus(Text(at)) | reflect(cursor_box_),
                Text(after),
            }) | xflex);
        }

        auto element = vbox(std::move(elements)) | frame;
        return transform_func({
                   std::move(element),
                   hovered_,
                   is_focused,
                   false
               }) | xflex | reflect(box_);
    }

    // ── Text editing handlers ─────────────────────────────────────────────────

    bool HandleBackspace() {
        if (cursor_position() == 0) {
            return false;
        }
        const size_t start = glyph_previous(*content, static_cast<size_t>(cursor_position()));
        const size_t end = static_cast<size_t>(cursor_position());
        content->erase(start, end - start);
        cursor_position() = static_cast<int>(start);
        return true;
    }

    bool HandleDelete() {
        if (cursor_position() == static_cast<int>(content->size())) {
            return false;
        }
        const size_t start = static_cast<size_t>(cursor_position());
        const size_t end = glyph_next(*content, start);
        content->erase(start, end - start);
        return true;
    }

    // Delete from cursor to end of current line (Ctrl+K).
    bool HandleDeleteToEnd() {
        const int content_size = static_cast<int>(content->size());
        if (cursor_position() == content_size) {
            return false;
        }
        int end = cursor_position();
        while (end < content_size && (*content)[static_cast<size_t>(end)] != '\n') {
            end = static_cast<int>(glyph_next(*content, static_cast<size_t>(end)));
        }
        if (end == cursor_position()) {
            // Cursor is on a newline — delete it to join the lines.
            end = static_cast<int>(glyph_next(*content, static_cast<size_t>(end)));
        }
        content->erase(static_cast<size_t>(cursor_position()),
                       static_cast<size_t>(end - cursor_position()));
        on_change();
        return true;
    }

    // Delete from cursor to start of current line (Ctrl+U).
    bool HandleDeleteToStart() {
        if (cursor_position() == 0) {
            return false;
        }
        int start = cursor_position();
        while (start > 0) {
            const size_t prev = glyph_previous(*content, static_cast<size_t>(start));
            if ((*content)[prev] == '\n') {
                break;
            }
            start = static_cast<int>(prev);
        }
        content->erase(static_cast<size_t>(start),
                       static_cast<size_t>(cursor_position() - start));
        cursor_position() = start;
        on_change();
        return true;
    }

    // Delete the word immediately to the left of the cursor (Ctrl+W, Alt+Backspace).
    bool HandleDeleteWordLeft() {
        if (cursor_position() == 0) {
            return false;
        }
        const int end = cursor_position();
        // Skip non-word characters first, then the preceding word.
        while (cursor_position() > 0) {
            const size_t prev = glyph_previous(*content, static_cast<size_t>(cursor_position()));
            if (is_word_character(*content, prev)) {
                break;
            }
            cursor_position() = static_cast<int>(prev);
        }
        while (cursor_position() > 0) {
            const size_t prev = glyph_previous(*content, static_cast<size_t>(cursor_position()));
            if (!is_word_character(*content, prev)) {
                break;
            }
            cursor_position() = static_cast<int>(prev);
        }
        content->erase(static_cast<size_t>(cursor_position()),
                       static_cast<size_t>(end - cursor_position()));
        on_change();
        return true;
    }

    // Delete the word immediately to the right of the cursor (Alt+D, Alt+Delete, Ctrl+Delete).
    bool HandleDeleteWordRight() {
        const int content_size = static_cast<int>(content->size());
        if (cursor_position() == content_size) {
            return false;
        }
        const int start = cursor_position();
        int pos = start;
        // Skip non-word characters first, then the following word.
        while (pos < content_size && !is_word_character(*content, static_cast<size_t>(pos))) {
            pos = static_cast<int>(glyph_next(*content, static_cast<size_t>(pos)));
        }
        while (pos < content_size && is_word_character(*content, static_cast<size_t>(pos))) {
            pos = static_cast<int>(glyph_next(*content, static_cast<size_t>(pos)));
        }
        content->erase(static_cast<size_t>(start), static_cast<size_t>(pos - start));
        on_change();
        return true;
    }

    // ── Cursor movement handlers ──────────────────────────────────────────────

    bool HandleArrowLeft() {
        if (cursor_position() == 0) {
            return false;
        }
        cursor_position() = static_cast<int>(glyph_previous(*content, static_cast<size_t>(cursor_position())));
        return true;
    }

    bool HandleArrowRight() {
        if (cursor_position() == static_cast<int>(content->size())) {
            return false;
        }
        cursor_position() = static_cast<int>(glyph_next(*content, static_cast<size_t>(cursor_position())));
        return true;
    }

    int CursorColumn() const {
        size_t iter = static_cast<size_t>(cursor_position());
        int width = 0;
        while (true) {
            if (iter == 0) {
                break;
            }
            iter = glyph_previous(*content, iter);
            if ((*content)[iter] == '\n') {
                break;
            }
            width += static_cast<int>(glyph_width(*content, iter));
        }
        return width;
    }

    void MoveCursorColumn(int columns) {
        while (columns > 0) {
            if (cursor_position() == static_cast<int>(content().size())
                || content()[static_cast<size_t>(cursor_position())] == '\n') {
                return;
            }

            columns -= static_cast<int>(glyph_width(content(), static_cast<size_t>(cursor_position())));
            cursor_position() = static_cast<int>(glyph_next(content(), static_cast<size_t>(cursor_position())));
        }
    }

    bool HandleArrowUp() {
        if (cursor_position() == 0) {
            return false;
        }

        const int columns = CursorColumn();
        while (true) {
            if (cursor_position() == 0) {
                return true;
            }
            const size_t previous = glyph_previous(*content, static_cast<size_t>(cursor_position()));
            if ((*content)[previous] == '\n') {
                break;
            }
            cursor_position() = static_cast<int>(previous);
        }
        cursor_position() = static_cast<int>(glyph_previous(*content, static_cast<size_t>(cursor_position())));
        while (true) {
            if (cursor_position() == 0) {
                break;
            }
            const size_t previous = glyph_previous(*content, static_cast<size_t>(cursor_position()));
            if ((*content)[previous] == '\n') {
                break;
            }
            cursor_position() = static_cast<int>(previous);
        }

        MoveCursorColumn(columns);
        return true;
    }

    bool HandleArrowDown() {
        if (cursor_position() == static_cast<int>(content->size())) {
            return false;
        }

        const int columns = CursorColumn();
        while (true) {
            if ((*content)[static_cast<size_t>(cursor_position())] == '\n') {
                break;
            }
            cursor_position() = static_cast<int>(glyph_next(*content, static_cast<size_t>(cursor_position())));
            if (cursor_position() == static_cast<int>(content().size())) {
                return true;
            }
        }
        cursor_position() = static_cast<int>(glyph_next(*content, static_cast<size_t>(cursor_position())));

        MoveCursorColumn(columns);
        return true;
    }

    bool HandleHome() {
        cursor_position() = 0;
        return true;
    }

    bool HandleEnd() {
        cursor_position() = static_cast<int>(content->size());
        return true;
    }

    // Move one word to the left (Ctrl+Left, Alt+Left, Alt+B).
    bool HandleWordLeft() {
        if (cursor_position() == 0) {
            return false;
        }

        while (cursor_position()) {
            const size_t previous = glyph_previous(*content, static_cast<size_t>(cursor_position()));
            if (is_word_character(*content, previous)) {
                break;
            }
            cursor_position() = static_cast<int>(previous);
        }
        while (cursor_position()) {
            const size_t previous = glyph_previous(*content, static_cast<size_t>(cursor_position()));
            if (!is_word_character(*content, previous)) {
                break;
            }
            cursor_position() = static_cast<int>(previous);
        }
        return true;
    }

    // Move one word to the right (Ctrl+Right, Alt+Right, Alt+F).
    bool HandleWordRight() {
        if (cursor_position() == static_cast<int>(content().size())) {
            return false;
        }

        while (cursor_position() < static_cast<int>(content().size())) {
            cursor_position() = static_cast<int>(glyph_next(content(), static_cast<size_t>(cursor_position())));
            if (is_word_character(content(), static_cast<size_t>(cursor_position()))) {
                break;
            }
        }
        while (cursor_position() < static_cast<int>(content().size())) {
            const size_t next = glyph_next(content(), static_cast<size_t>(cursor_position()));
            if (!is_word_character(content(), static_cast<size_t>(cursor_position()))) {
                break;
            }
            cursor_position() = static_cast<int>(next);
        }
        return true;
    }

    // ── Submission and newline ────────────────────────────────────────────────

    bool ShouldTreatReturnAsPasteNewline() const {
        if (last_character_event_ == Clock::time_point::min()) {
            return false;
        }
        return (Clock::now() - last_character_event_) <= kFastReturnThreshold;
    }

    bool HandleReturn() {
        if (paste_mode_ || ShouldTreatReturnAsPasteNewline()) {
            return HandleCharacter("\n");
        }
        if (multiline()) {
            HandleCharacter("\n");
        }
        on_enter();
        return true;
    }

    // Insert a newline without submitting (Shift+Enter, Ctrl+Enter, Alt+Enter).
    bool HandleInsertNewline() {
        return HandleCharacter("\n");
    }

    bool HandleCharacter(const std::string& character) {
        content->insert(static_cast<size_t>(cursor_position()), character);
        cursor_position() += static_cast<int>(character.size());
        last_character_event_ = Clock::now();
        on_change();
        return true;
    }

    // ── Mouse ─────────────────────────────────────────────────────────────────

    bool HandleMouse(Event event) {
        hovered_ = box_.Contain(event.mouse().x, event.mouse().y) && CaptureMouse(event);
        if (!hovered_) {
            return false;
        }

        if (event.mouse().button != Mouse::Left || event.mouse().motion != Mouse::Pressed) {
            return false;
        }

        TakeFocus();

        if (content->empty()) {
            cursor_position() = 0;
            return true;
        }

        const auto lines = split_lines(*content);
        const auto [cur_line, cur_char_index] = cursor_coord(lines);
        const int cursor_column = string_width(
            lines[static_cast<size_t>(cur_line)].substr(0, static_cast<size_t>(cur_char_index)));

        int new_cursor_column = cursor_column + event.mouse().x - cursor_box_.x_min;
        int new_cursor_line   = cur_line + event.mouse().y - cursor_box_.y_min;
        new_cursor_line = std::max(std::min(new_cursor_line, static_cast<int>(lines.size())), 0);

        const std::string empty_string;
        const std::string& line =
            new_cursor_line < static_cast<int>(lines.size())
                ? lines[static_cast<size_t>(new_cursor_line)]
                : empty_string;
        new_cursor_column = std::clamp(new_cursor_column, 0, string_width(line));

        if (new_cursor_column == cursor_column && new_cursor_line == cur_line) {
            return false;
        }

        cursor_position() = 0;
        for (int i = 0; i < new_cursor_line; ++i) {
            cursor_position() += static_cast<int>(lines[static_cast<size_t>(i)].size()) + 1;
        }
        while (new_cursor_column > 0) {
            new_cursor_column -= static_cast<int>(glyph_width(*content, static_cast<size_t>(cursor_position())));
            cursor_position() = static_cast<int>(glyph_next(*content, static_cast<size_t>(cursor_position())));
        }

        on_change();
        return true;
    }

    // ── Event dispatch ────────────────────────────────────────────────────────

    bool OnEvent(Event event) override {
        cursor_position() = std::clamp(cursor_position(), 0, static_cast<int>(content->size()));

        // Bracketed paste sequences (ESC[200~ ... ESC[201~).
        if (event == Event::Special("\x1B[200~")) {
            paste_mode_ = true;
            return true;
        }
        if (event == Event::Special("\x1B[201~")) {
            paste_mode_ = false;
            return true;
        }
        if (paste_mode_) {
            if (event == Event::Return)      return HandleCharacter("\n");
            if (event == Event::Tab)         return HandleCharacter("\t");
            if (event.is_character())        return HandleCharacter(event.character());
            return true;
        }

        if (event == Event::Return)          return HandleReturn();
        if (event.is_character())            return HandleCharacter(event.character());
        if (event.is_mouse())                return HandleMouse(event);
        if (event == Event::Special({4}))    return HandleDelete();      // Ctrl+D
        if (event == Event::Backspace)       return HandleBackspace();
        if (event == Event::Delete)          return HandleDelete();
        if (event == Event::ArrowLeft)       return HandleArrowLeft();
        if (event == Event::ArrowRight)      return HandleArrowRight();
        if (event == Event::ArrowUp)         return HandleArrowUp();
        if (event == Event::ArrowDown)       return HandleArrowDown();
        if (event == Event::Home)            return HandleHome();
        if (event == Event::End)             return HandleEnd();
        if (event == Event::ArrowLeftCtrl)   return HandleWordLeft();
        if (event == Event::ArrowRightCtrl)  return HandleWordRight();

        // ── Cursor movement (Gemini CLI compat) ──────────────────────────────
        // Ctrl+A → start of line
        if (event == Event::Special({1}))    return HandleHome();
        // Ctrl+E → end of line
        if (event == Event::Special({5}))    return HandleEnd();
        // Ctrl+F → move right one character
        if (event == Event::Special({6}))    return HandleArrowRight();
        // Alt+Left / Alt+B → word left
        if (event == Event::Special("\x1B[1;3D") ||
            event == Event::Special("\x1B" "b"))  return HandleWordLeft();
        // Alt+Right / Alt+F → word right
        if (event == Event::Special("\x1B[1;3C") ||
            event == Event::Special("\x1B" "f"))  return HandleWordRight();

        // ── Editing (Gemini CLI compat) ──────────────────────────────────────
        // Ctrl+K → delete to end of line
        if (event == Event::Special({11}))   return HandleDeleteToEnd();
        // Ctrl+U → delete to start of line
        if (event == Event::Special({21}))   return HandleDeleteToStart();
        // Ctrl+W / Alt+Backspace → delete word left
        if (event == Event::Special({23}) ||
            event == Event::Special("\x1B\x7f"))  return HandleDeleteWordLeft();
        // Alt+D / Alt+Delete / Ctrl+Delete → delete word right
        if (event == Event::Special("\x1B" "d") ||
            event == Event::Special("\x1B[3;3~") ||
            event == Event::Special("\x1B[3;5~")) return HandleDeleteWordRight();

        // ── Newline insertion without submission (Gemini CLI compat) ─────────
        // Shift+Enter / Ctrl+Enter / Alt+Enter
        if (event == Event::Special("\x1B[27;2;13~") ||
            event == Event::Special("\x1B[27;5;13~") ||
            event == Event::Special("\x1B[27;3;13~") ||
            event == Event::Special("\x1B\n"))         return HandleInsertNewline();

        return false;
    }

    bool Focusable() const final { return true; }

    bool hovered_ = false;
    Box  box_;
    Box  cursor_box_;
    bool paste_mode_ = false;
    using Clock = std::chrono::steady_clock;
    static constexpr auto kFastReturnThreshold = std::chrono::milliseconds(30);
    Clock::time_point last_character_event_ = Clock::time_point::min();
};

} // namespace

Component PromptInput(StringRef content,
                      StringRef placeholder,
                      InputOption option) {
    option.content = std::move(content);
    option.placeholder = std::move(placeholder);
    return Make<PromptInputBase>(std::move(option));
}

} // namespace tui
