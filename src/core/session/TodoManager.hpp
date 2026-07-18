#pragma once

#include "SessionData.hpp"

#include <expected>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace core::session {

struct TodoDraft {
    std::string id;
    std::string text;
    TodoStatus status = TodoStatus::Pending;
};

class TodoManager final {
public:
    using ClockFn = std::string (*)();
    static constexpr std::size_t kMaxItems = 100;
    static constexpr std::size_t kMaxIdChars = 128;
    static constexpr std::size_t kMaxTextChars = 2048;

    explicit TodoManager(ClockFn clock);

    void restore(std::vector<SessionTodoItem> todos);
    [[nodiscard]] std::vector<SessionTodoItem> current() const;

    [[nodiscard]] std::expected<std::vector<SessionTodoItem>, std::string>
    replace(std::vector<TodoDraft> drafts);
    [[nodiscard]] std::expected<SessionTodoItem, std::string> add(std::string_view text);
    [[nodiscard]] std::expected<SessionTodoItem, std::string>
    set_status(std::string_view selector, TodoStatus status);
    [[nodiscard]] std::expected<SessionTodoItem, std::string> remove(std::string_view selector);
    [[nodiscard]] std::size_t clear_completed();
    void clear();

    [[nodiscard]] std::string prompt_context() const;

private:
    [[nodiscard]] static std::string normalize_text(std::string_view text);
    [[nodiscard]] std::string now() const;

    ClockFn clock_;
    mutable std::mutex mutex_;
    std::vector<SessionTodoItem> todos_;
};

} // namespace core::session
