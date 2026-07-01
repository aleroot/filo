#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace tui {

class SelectionClipboardCopier {
public:
    SelectionClipboardCopier();
    ~SelectionClipboardCopier();

    SelectionClipboardCopier(const SelectionClipboardCopier&) = delete;
    SelectionClipboardCopier& operator=(const SelectionClipboardCopier&) = delete;

    void enqueue(std::string selection);

private:
    void run();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<std::string> pending_;
    std::size_t generation_ = 0;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace tui
