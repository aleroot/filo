#pragma once

#include "ToolAccess.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <functional>
#include <optional>
#include <thread>
#include <vector>

namespace core::agent {

template <typename Result>
struct ScheduledToolTask {
    ToolAccessSet accesses;
    std::vector<std::size_t> after;
    std::function<Result()> run;
};

template <typename Result>
class ToolCallScheduler {
public:
    [[nodiscard]] std::vector<Result> run(std::vector<ScheduledToolTask<Result>> tasks) {
        std::vector<std::optional<Result>> results(tasks.size());
        std::vector<std::future<Result>> futures(tasks.size());
        std::vector<bool> started(tasks.size(), false);
        std::vector<bool> finished(tasks.size(), false);
        std::vector<std::size_t> active;
        active.reserve(tasks.size());

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
                started[i] = true;
                active.push_back(i);
                futures[i] = std::async(std::launch::async, [task = std::move(tasks[i].run)]() mutable {
                    return task();
                });
                launched = true;
            }

            if (!launched && active.empty()) {
                for (std::size_t i = 0; i < tasks.size(); ++i) {
                    if (!started[i]) {
                        started[i] = true;
                        active.push_back(i);
                        futures[i] = std::async(std::launch::async, [task = std::move(tasks[i].run)]() mutable {
                            return task();
                        });
                        break;
                    }
                }
            }

            const std::size_t done_index = wait_for_one(active, futures);
            std::erase(active, done_index);
            results[done_index] = futures[done_index].get();
            finished[done_index] = true;
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

    [[nodiscard]] static std::size_t wait_for_one(const std::vector<std::size_t>& active,
                                                  std::vector<std::future<Result>>& futures) {
        using namespace std::chrono_literals;
        while (true) {
            for (const std::size_t index : active) {
                if (futures[index].wait_for(0ms) == std::future_status::ready) {
                    return index;
                }
            }
            std::this_thread::sleep_for(1ms);
        }
    }
};

} // namespace core::agent
