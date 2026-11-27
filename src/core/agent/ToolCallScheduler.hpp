#pragma once

#include "ToolAccess.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace core::agent {

template <typename Result>
struct ScheduledToolTask {
    ToolAccessSet accesses;
    std::vector<std::size_t> after = {};
    std::function<Result()> run;
};

template <typename Result>
class ToolCallScheduler {
public:
    /// Runs tasks subject to their access/dependency constraints. `on_complete`
    /// is called on the scheduling thread as soon as each result is available,
    /// rather than after the complete batch has finished. The returned vector
    /// remains ordered by the submitted tasks for model-history replay.
    [[nodiscard]] std::vector<Result> run(
        std::vector<ScheduledToolTask<Result>> tasks,
        std::function<void(std::size_t, const Result&)> on_complete = {}) {
        std::vector<std::optional<Result>> results(tasks.size());
        std::vector<std::future<Result>> futures(tasks.size());
        std::vector<bool> started(tasks.size(), false);
        std::vector<bool> finished(tasks.size(), false);
        std::vector<std::size_t> active;
        active.reserve(tasks.size());
        // Each task reports here as it ends, so the scheduling thread sleeps
        // until one does instead of polling.
        const auto completions = std::make_shared<Completions>();
        const auto launch = [&](std::size_t i) {
            started[i] = true;
            active.push_back(i);
            futures[i] = std::async(std::launch::async,
                                    [task = std::move(tasks[i].run), completions, i]() mutable {
                const Reporter reporter{*completions, i};
                return task();
            });
        };

        std::size_t completed = 0;
        while (completed < tasks.size()) {
            bool launched = false;
            for (std::size_t i = 0; i < tasks.size(); ++i) {
                if (started[i]) {
                    continue;
                }
                if (is_blocked(i, tasks, active, started, finished)) {
                    continue;
                }
                launch(i);
                launched = true;
            }

            if (!launched && active.empty()) {
                for (std::size_t i = 0; i < tasks.size(); ++i) {
                    if (!started[i]) {
                        launch(i);
                        break;
                    }
                }
            }

            const std::size_t done_index = completions->wait_for_one();
            std::erase(active, done_index);
            results[done_index] = futures[done_index].get();
            finished[done_index] = true;
            if (on_complete) {
                on_complete(done_index, *results[done_index]);
            }
            ++completed;
        }

        std::vector<Result> ordered;
        ordered.reserve(results.size());
        for (auto& result : results) {
            ordered.push_back(std::move(*result));
        }
        return ordered;
    }

private:
    /// The indices of tasks that ended, in the order they did.
    class Completions {
    public:
        void report(std::size_t index) {
            {
                std::lock_guard lock(mutex_);
                done_.push_back(index);
            }
            changed_.notify_one();
        }

        [[nodiscard]] std::size_t wait_for_one() {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, [this] { return !done_.empty(); });
            const std::size_t index = done_.front();
            done_.pop_front();
            return index;
        }

    private:
        std::mutex mutex_;
        std::condition_variable changed_;
        std::deque<std::size_t> done_;
    };

    /// Reports its task as ended however the task leaves, a throw included
    /// (the future then rethrows it on get()).
    struct Reporter {
        Completions& completions;
        std::size_t index;
        ~Reporter() { completions.report(index); }
    };

    [[nodiscard]] static bool is_blocked(std::size_t index,
                                         const std::vector<ScheduledToolTask<Result>>& tasks,
                                         const std::vector<std::size_t>& active,
                                         const std::vector<bool>& started,
                                         const std::vector<bool>& finished) {
        for (const std::size_t active_index : active) {
            if (tool_access_sets_conflict(tasks[index].accesses, tasks[active_index].accesses)) {
                return true;
            }
        }
        for (const std::size_t dependency : tasks[index].after) {
            if (dependency < finished.size() && !finished[dependency]) {
                return true;
            }
        }
        for (std::size_t i = 0; i < index; ++i) {
            if (!started[i] || !finished[i]) {
                if (tool_access_sets_conflict(tasks[index].accesses, tasks[i].accesses)) {
                    return true;
                }
            }
        }
        return false;
    }
};

} // namespace core::agent
