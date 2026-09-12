#include "ThreadModalHost.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace tui {

void PermissionEventResult::resolve() {
    if (promise && approved.has_value()) {
        promise->set_value(*approved);
        promise.reset();
    }
}

PermissionSlotGuard::PermissionSlotGuard(ThreadModalHost& host,
                                         std::string session_id)
    : host_(&host)
    , session_id_(std::move(session_id)) {}

PermissionSlotGuard::~PermissionSlotGuard() {
    if (host_ != nullptr) {
        host_->release_permission_slot(session_id_);
    }
}

PermissionSlotGuard::PermissionSlotGuard(PermissionSlotGuard&& other) noexcept
    : host_(other.host_)
    , session_id_(std::move(other.session_id_)) {
    other.host_ = nullptr;
}

PermissionSlotGuard& PermissionSlotGuard::operator=(
    PermissionSlotGuard&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (host_ != nullptr) {
        host_->release_permission_slot(session_id_);
    }
    host_ = other.host_;
    session_id_ = std::move(other.session_id_);
    other.host_ = nullptr;
    return *this;
}

ThreadModalHost::~ThreadModalHost() = default;

ThreadModalHost::SessionSlot& ThreadModalHost::slot_locked(
    const std::string& session_id) {
    return slots_[session_id];
}

ThreadModalHost::SessionSlot* ThreadModalHost::find_slot_locked(
    std::string_view session_id) {
    auto it = slots_.find(session_id);
    if (it == slots_.end()) {
        return nullptr;
    }
    return &it->second;
}

const ThreadModalHost::SessionSlot* ThreadModalHost::find_slot_locked(
    std::string_view session_id) const {
    auto it = slots_.find(session_id);
    if (it == slots_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::shared_ptr<QuestionDialogController>
ThreadModalHost::busy_question_locked(std::string_view session_id) const {
    const auto* slot = find_slot_locked(session_id);
    if (slot == nullptr || !slot->question_busy) {
        return {};
    }
    return slot->question;
}

void ThreadModalHost::mark_question_idle(std::string_view session_id) {
    std::lock_guard lock(mutex_);
    if (auto* slot = find_slot_locked(session_id)) {
        slot->question_busy = false;
    }
    cv_.notify_all();
}

void ThreadModalHost::post_question(core::tools::QuestionRequest request) {
    const std::string session_id = request.session_id;
    std::shared_ptr<QuestionDialogController> controller;
    {
        std::unique_lock lock(mutex_);
        auto& slot = slot_locked(session_id);
        ++slot.question_waiters;
        cv_.wait(lock, [&]() {
            if (shutdown_) {
                return true;
            }
            const auto* waiting = find_slot_locked(session_id);
            return waiting == nullptr || !waiting->question_busy;
        });
        auto* ready = find_slot_locked(session_id);
        if (ready != nullptr) {
            --ready->question_waiters;
        }
        if (shutdown_ || ready == nullptr) {
            auto cancelled = std::move(request.promise);
            lock.unlock();
            if (cancelled) {
                cancelled->set_value(std::nullopt);
            }
            return;
        }
        ready->question_busy = true;
        if (!ready->question) {
            ready->question = std::make_shared<QuestionDialogController>();
        }
        controller = ready->question;
    }
    auto displaced = controller->open(std::move(request));
    if (displaced) {
        displaced->set_value(std::nullopt);
    }
}

bool ThreadModalHost::question_visible(std::string_view visible_session_id) const {
    std::lock_guard lock(mutex_);
    const auto* slot = find_slot_locked(visible_session_id);
    return slot != nullptr && slot->question_busy && slot->question != nullptr;
}

ftxui::Element ThreadModalHost::render_question(
    std::string_view visible_session_id) {
    std::shared_ptr<QuestionDialogController> controller;
    {
        std::lock_guard lock(mutex_);
        controller = busy_question_locked(visible_session_id);
    }
    if (!controller) {
        return ftxui::emptyElement();
    }
    return controller->render();
}

QuestionDialogEventResult ThreadModalHost::handle_question_event(
    std::string_view visible_session_id,
    const ftxui::Event& event,
    bool is_interrupt) {
    std::shared_ptr<QuestionDialogController> controller;
    {
        std::lock_guard lock(mutex_);
        controller = busy_question_locked(visible_session_id);
    }
    if (!controller) {
        return {};
    }
    auto result = controller->handle_event(event, is_interrupt);
    if (result.has_resolution()) {
        mark_question_idle(visible_session_id);
    }
    return result;
}

std::vector<QuestionDialogEventResult> ThreadModalHost::interrupt_all_questions() {
    std::vector<std::pair<std::string, std::shared_ptr<QuestionDialogController>>>
        controllers;
    {
        std::lock_guard lock(mutex_);
        controllers.reserve(slots_.size());
        for (auto& [id, slot] : slots_) {
            if (slot.question_busy && slot.question) {
                controllers.emplace_back(id, slot.question);
            }
        }
    }
    std::vector<QuestionDialogEventResult> results;
    results.reserve(controllers.size());
    for (auto& [id, controller] : controllers) {
        auto dismissed = controller->force_interrupt();
        mark_question_idle(id);
        if (dismissed.has_resolution()) {
            results.push_back(std::move(dismissed));
        }
    }
    return results;
}

QuestionDialogEventResult ThreadModalHost::interrupt_question(
    std::string_view session_id) {
    std::shared_ptr<QuestionDialogController> controller;
    {
        std::lock_guard lock(mutex_);
        controller = busy_question_locked(session_id);
    }
    if (!controller) {
        return {};
    }
    auto result = controller->force_interrupt();
    mark_question_idle(session_id);
    return result;
}

PermissionSlotGuard ThreadModalHost::acquire_permission_slot(
    std::string session_id) {
    std::unique_lock lock(mutex_);
    auto& slot = slot_locked(session_id);
    ++slot.permission_waiters;
    cv_.wait(lock, [&]() {
        if (shutdown_) {
            return true;
        }
        const auto* waiting = find_slot_locked(session_id);
        return waiting == nullptr || !waiting->permission_busy;
    });
    auto* ready = find_slot_locked(session_id);
    if (ready != nullptr) {
        --ready->permission_waiters;
    }
    if (shutdown_ || ready == nullptr) {
        return {};
    }
    ready->permission_busy = true;
    return PermissionSlotGuard(*this, std::move(session_id));
}

void ThreadModalHost::release_permission_slot(std::string_view session_id) {
    std::lock_guard lock(mutex_);
    if (auto* slot = find_slot_locked(session_id)) {
        slot->permission_busy = false;
        slot->permission.reset();
    }
    cv_.notify_all();
}

void ThreadModalHost::post_permission(PermissionPrompt prompt) {
    std::shared_ptr<std::promise<bool>> cancelled;
    {
        std::lock_guard lock(mutex_);
        if (shutdown_) {
            cancelled = std::move(prompt.promise);
        } else {
            auto& slot = slot_locked(prompt.session_id);
            slot.permission = std::move(prompt);
        }
    }
    if (cancelled) {
        cancelled->set_value(false);
    }
}

bool ThreadModalHost::permission_visible(
    std::string_view visible_session_id) const {
    std::lock_guard lock(mutex_);
    const auto* slot = find_slot_locked(visible_session_id);
    return slot != nullptr && slot->permission.has_value();
}

std::optional<PermissionView> ThreadModalHost::permission_view(
    std::string_view visible_session_id) const {
    std::lock_guard lock(mutex_);
    const auto* slot = find_slot_locked(visible_session_id);
    if (slot == nullptr || !slot->permission.has_value()) {
        return std::nullopt;
    }
    const auto& prompt = *slot->permission;
    return PermissionView{
        .session_id = prompt.session_id,
        .tool_name = prompt.tool_name,
        .args_preview = prompt.args_preview,
        .diff_preview = prompt.diff_preview,
        .selected = prompt.selected,
        .allow_label = prompt.allow_label,
    };
}

PermissionEventResult ThreadModalHost::handle_permission_event(
    std::string_view visible_session_id,
    const ftxui::Event& event,
    bool is_interrupt,
    bool yolo_shortcut) {
    std::lock_guard lock(mutex_);
    auto* slot = find_slot_locked(visible_session_id);
    if (slot == nullptr || !slot->permission.has_value()) {
        return {};
    }
    auto& prompt = *slot->permission;

    PermissionEventResult result;
    result.handled = true;
    result.origin_session_id = prompt.session_id;

    auto take = [&](std::optional<bool> approved,
                    bool always_allow,
                    bool enable_yolo,
                    bool stop_agent) {
        result.approved = approved;
        result.always_allow = always_allow;
        result.enable_yolo = enable_yolo;
        result.stop_agent = stop_agent;
        result.remember_rule = prompt.remember_rule;
        result.allow_label = prompt.allow_label;
        result.promise = std::move(prompt.promise);
        result.restore_main_input_focus = true;
        slot->permission.reset();
    };

    if (event == ftxui::Event::ArrowUp) {
        prompt.selected = (prompt.selected + 3) % 4;
        return result;
    }
    if (event == ftxui::Event::ArrowDown) {
        prompt.selected = (prompt.selected + 1) % 4;
        return result;
    }
    if (event == ftxui::Event::Character('1')
        || event == ftxui::Event::Character('y')
        || event == ftxui::Event::Character('Y')) {
        take(true, false, false, false);
        return result;
    }
    if (event == ftxui::Event::Character('2')
        || event == ftxui::Event::Character('a')
        || event == ftxui::Event::Character('A')) {
        take(true, true, false, false);
        return result;
    }
    if (event == ftxui::Event::Character('3') || yolo_shortcut) {
        take(true, false, true, false);
        return result;
    }
    if (event == ftxui::Event::Return) {
        const int selected = prompt.selected;
        take(selected != 3, selected == 1, selected == 2, false);
        return result;
    }
    if (is_interrupt) {
        take(false, false, false, true);
        return result;
    }
    if (event == ftxui::Event::Character('4')
        || event == ftxui::Event::Character('n')
        || event == ftxui::Event::Character('N')
        || event == ftxui::Event::Escape) {
        take(false, false, false, false);
        return result;
    }
    return result;
}

bool ThreadModalHost::waiting(std::string_view session_id) const {
    std::lock_guard lock(mutex_);
    const auto* slot = find_slot_locked(session_id);
    if (slot == nullptr) {
        return false;
    }
    return slot->question_busy || slot->permission.has_value();
}

std::unordered_set<std::string> ThreadModalHost::waiting_session_ids() const {
    std::lock_guard lock(mutex_);
    std::unordered_set<std::string> ids;
    for (const auto& [id, slot] : slots_) {
        if (slot.question_busy || slot.permission.has_value()) {
            ids.insert(id);
        }
    }
    return ids;
}

std::size_t ThreadModalHost::blocked_question_waiters(
    std::string_view session_id) const {
    std::lock_guard lock(mutex_);
    const auto* slot = find_slot_locked(session_id);
    return slot == nullptr ? 0 : slot->question_waiters;
}

std::size_t ThreadModalHost::blocked_permission_waiters(
    std::string_view session_id) const {
    std::lock_guard lock(mutex_);
    const auto* slot = find_slot_locked(session_id);
    return slot == nullptr ? 0 : slot->permission_waiters;
}

QuestionDialogEventResult ThreadModalHost::erase_session(
    std::string_view session_id) {
    std::shared_ptr<QuestionDialogController> controller;
    std::shared_ptr<std::promise<bool>> permission_promise;
    {
        std::lock_guard lock(mutex_);
        auto it = slots_.find(session_id);
        if (it == slots_.end()) {
            return {};
        }
        if (it->second.question_busy) {
            controller = it->second.question;
        }
        if (it->second.permission && it->second.permission->promise) {
            permission_promise = std::move(it->second.permission->promise);
        }
        slots_.erase(it);
        cv_.notify_all();
    }
    QuestionDialogEventResult question;
    if (controller) {
        question = controller->force_interrupt();
    }
    if (permission_promise) {
        permission_promise->set_value(false);
    }
    return question;
}

void ThreadModalHost::request_shutdown() {
    std::vector<std::shared_ptr<QuestionDialogController>> questions;
    std::vector<std::shared_ptr<std::promise<bool>>> permission_promises;
    {
        std::lock_guard lock(mutex_);
        shutdown_ = true;
        for (auto& [_, slot] : slots_) {
            if (slot.question_busy && slot.question) {
                questions.push_back(slot.question);
            }
            slot.question_busy = false;
            if (slot.permission && slot.permission->promise) {
                permission_promises.push_back(std::move(slot.permission->promise));
            }
            slot.permission.reset();
            slot.permission_busy = false;
        }
        cv_.notify_all();
    }
    for (const auto& controller : questions) {
        auto dismissed = controller->force_interrupt();
        dismissed.resolve();
    }
    for (auto& promise : permission_promises) {
        promise->set_value(false);
    }
}

bool ThreadModalHost::shutting_down() const {
    std::lock_guard lock(mutex_);
    return shutdown_;
}

} // namespace tui
