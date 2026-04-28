#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>

namespace exec::http {

[[nodiscard]] inline std::string sse_data_frame(std::string_view payload) {
    std::string frame;
    frame.reserve(payload.size() + 8);
    frame += "data: ";
    frame += payload;
    frame += "\n\n";
    return frame;
}

class SseFrameQueue {
public:
    enum class WaitResult {
        frames,
        closed,
        timeout,
    };

    void push(std::string frame) {
        {
            std::lock_guard lock(mutex_);
            if (closed_) return;
            frames_.push_back(std::move(frame));
        }
        cv_.notify_all();
    }

    void push_data(std::string_view payload) {
        push(sse_data_frame(payload));
    }

    void close() {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    [[nodiscard]] bool is_closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    [[nodiscard]] WaitResult wait_pop_until(
        std::chrono::steady_clock::time_point deadline,
        std::deque<std::string>& out) {
        std::unique_lock lock(mutex_);
        if (!cv_.wait_until(lock, deadline, [&] {
                return !frames_.empty() || closed_;
            })) {
            return WaitResult::timeout;
        }

        out.swap(frames_);
        return out.empty() && closed_ ? WaitResult::closed : WaitResult::frames;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> frames_;
    bool closed_ = false;
};

} // namespace exec::http
