#include "SelectionClipboardCopier.hpp"

#include "core/commands/CommandExecutor.hpp"

#include <chrono>
#include <utility>

namespace tui {
namespace {

constexpr auto kSelectionClipboardDebounce = std::chrono::milliseconds(75);

} // namespace

SelectionClipboardCopier::SelectionClipboardCopier()
    : worker_([this]() { run(); }) {}

SelectionClipboardCopier::~SelectionClipboardCopier() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void SelectionClipboardCopier::enqueue(std::string selection) {
    if (selection.empty()) {
        return;
    }
    {
        std::lock_guard lock(mutex_);
        pending_ = std::move(selection);
        ++generation_;
    }
    cv_.notify_one();
}

void SelectionClipboardCopier::run() {
    std::unique_lock lock(mutex_);
    while (true) {
        cv_.wait(lock, [this]() {
            return stopping_ || pending_.has_value();
        });
        if (stopping_) {
            break;
        }

        const std::size_t observed_generation = generation_;
        const bool changed =
            cv_.wait_for(lock, kSelectionClipboardDebounce, [this, observed_generation]() {
                return stopping_ || generation_ != observed_generation;
            });
        if (stopping_) {
            break;
        }
        if (changed) {
            continue;
        }

        std::string selection = std::move(*pending_);
        pending_.reset();
        lock.unlock();
        core::commands::copy_text_to_clipboard(selection);
        lock.lock();
    }
}

} // namespace tui
