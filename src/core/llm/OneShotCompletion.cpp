#include "OneShotCompletion.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>

namespace core::llm {

namespace {

[[nodiscard]] std::string trim_copy(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

[[nodiscard]] bool
cancelled(const std::function<bool()> &cancellation_requested) {
  return cancellation_requested && cancellation_requested();
}

} // namespace

std::expected<std::string, std::string>
complete_once(const std::shared_ptr<LLMProvider> &provider,
              std::string_view model, std::string_view prompt,
              std::function<bool()> cancellation_requested) {
  if (!provider) {
    return std::unexpected("no active provider");
  }
  if (cancelled(cancellation_requested)) {
    return std::unexpected(std::string(kCancelledMessage));
  }

  auto execution_provider = provider;
  if (auto isolated = provider->fork_for_parallel_request()) {
    execution_provider = std::move(isolated);
  }

  ChatRequest request;
  request.model = std::string(model);
  request.messages.push_back({"user", std::string(prompt), "", "", {}});

  std::mutex output_mutex;
  std::string output;
  std::atomic_bool completed{false};
  std::atomic_bool failed{false};
  std::atomic_bool stop_watch{false};
  std::thread watcher;
  if (cancellation_requested) {
    watcher = std::thread([&] {
      while (!stop_watch.load(std::memory_order_acquire)) {
        if (cancelled(cancellation_requested)) {
          execution_provider->cancel();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }

  const auto stop_watcher = [&] {
    stop_watch.store(true, std::memory_order_release);
    if (watcher.joinable()) {
      watcher.join();
    }
  };

  try {
    execution_provider->stream_response(request, [&](const StreamChunk &chunk) {
      if (!chunk.content.empty()) {
        std::lock_guard lock(output_mutex);
        output += chunk.content;
      }
      if (chunk.is_final) {
        failed.store(chunk.is_error, std::memory_order_release);
        completed.store(true, std::memory_order_release);
      }
    });
  } catch (const std::exception &error) {
    stop_watcher();
    if (cancelled(cancellation_requested)) {
      return std::unexpected(std::string(kCancelledMessage));
    }
    return std::unexpected(error.what());
  } catch (...) {
    stop_watcher();
    if (cancelled(cancellation_requested)) {
      return std::unexpected(std::string(kCancelledMessage));
    }
    return std::unexpected("unknown provider error");
  }
  stop_watcher();

  if (cancelled(cancellation_requested)) {
    return std::unexpected(std::string(kCancelledMessage));
  }
  if (failed.load(std::memory_order_acquire)) {
    return std::unexpected("provider returned an error");
  }
  if (!completed.load(std::memory_order_acquire)) {
    return std::unexpected("stream ended without a final response");
  }
  {
    std::lock_guard lock(output_mutex);
    output = trim_copy(std::move(output));
  }
  if (output.empty()) {
    return std::unexpected("empty model response");
  }
  return output;
}

} // namespace core::llm
