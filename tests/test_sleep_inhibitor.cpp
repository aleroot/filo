#include <catch2/catch_test_macros.hpp>

#include "core/agent/Agent.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/power/SleepInhibitor.hpp"
#include "core/tools/ToolManager.hpp"
#include "TestSessionContext.hpp"

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <unistd.h>
#endif

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace {

struct InhibitionState {
    int acquisitions = 0;
    int active_leases = 0;
    std::string reason;
};

class TestSleepInhibitionLease final : public core::power::SleepInhibitionLease {
public:
    explicit TestSleepInhibitionLease(std::shared_ptr<InhibitionState> state)
        : state_(std::move(state)) {
        ++state_->active_leases;
    }

    ~TestSleepInhibitionLease() override {
        --state_->active_leases;
    }

    [[nodiscard]] bool is_active() const noexcept override {
        return state_->active_leases > 0;
    }

private:
    std::shared_ptr<InhibitionState> state_;
};

class TestSleepInhibitor final : public core::power::SleepInhibitor {
public:
    explicit TestSleepInhibitor(std::shared_ptr<InhibitionState> state)
        : state_(std::move(state)) {}

    [[nodiscard]] std::shared_ptr<core::power::SleepInhibitionLease>
    inhibit(std::string_view reason) override {
        ++state_->acquisitions;
        state_->reason = reason;
        return std::make_shared<TestSleepInhibitionLease>(state_);
    }

private:
    std::shared_ptr<InhibitionState> state_;
};

class InhibitionObservingProvider final : public core::llm::LLMProvider {
public:
    explicit InhibitionObservingProvider(std::shared_ptr<InhibitionState> state)
        : state_(std::move(state)) {}

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        observed_active_lease_ = state_->active_leases == 1;
        callback(core::llm::StreamChunk{.content = "done", .is_final = true});
    }

    [[nodiscard]] bool observed_active_lease() const noexcept {
        return observed_active_lease_;
    }

private:
    std::shared_ptr<InhibitionState> state_;
    bool observed_active_lease_ = false;
};

class RetainingCancellationProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        retained_callback_ = callback;
        callback(core::llm::StreamChunk{.content = "partial"});
        callback(core::llm::StreamChunk::make_final());
    }

    void cancel() override {
        cancel_calls_.fetch_add(1, std::memory_order_relaxed);
    }

    void emit_late_terminal_chunk() {
        retained_callback_(core::llm::StreamChunk::make_final());
    }

    [[nodiscard]] int cancel_calls() const noexcept {
        return cancel_calls_.load(std::memory_order_relaxed);
    }

private:
    std::function<void(const core::llm::StreamChunk&)> retained_callback_;
    std::atomic_int cancel_calls_{0};
};

#if defined(__APPLE__)
[[nodiscard]] std::optional<std::size_t>
native_assertion_count(std::string_view reason) {
    CFDictionaryRef assertions_by_process = nullptr;
    if (IOPMCopyAssertionsByProcess(&assertions_by_process) != kIOReturnSuccess
        || assertions_by_process == nullptr) {
        return std::nullopt;
    }

    const int process_id = static_cast<int>(::getpid());
    const CFNumberRef process_key = CFNumberCreate(
        kCFAllocatorDefault, kCFNumberIntType, &process_id);
    const CFStringRef reason_string = CFStringCreateWithBytes(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(reason.data()),
        static_cast<CFIndex>(reason.size()),
        kCFStringEncodingUTF8,
        false);
    if (process_key == nullptr || reason_string == nullptr) {
        if (process_key != nullptr) CFRelease(process_key);
        if (reason_string != nullptr) CFRelease(reason_string);
        CFRelease(assertions_by_process);
        return std::nullopt;
    }

    std::size_t count = 0;
    const auto assertions = static_cast<CFArrayRef>(
        CFDictionaryGetValue(assertions_by_process, process_key));
    if (assertions != nullptr && CFGetTypeID(assertions) == CFArrayGetTypeID()) {
        const CFIndex assertion_count = CFArrayGetCount(assertions);
        for (CFIndex index = 0; index < assertion_count; ++index) {
            const auto assertion = static_cast<CFDictionaryRef>(
                CFArrayGetValueAtIndex(assertions, index));
            if (assertion == nullptr
                || CFGetTypeID(assertion) != CFDictionaryGetTypeID()) {
                continue;
            }
            const auto details = static_cast<CFStringRef>(
                CFDictionaryGetValue(assertion, kIOPMAssertionDetailsKey));
            if (details != nullptr
                && CFGetTypeID(details) == CFStringGetTypeID()
                && CFEqual(details, reason_string)) {
                ++count;
            }
        }
    }

    CFRelease(reason_string);
    CFRelease(process_key);
    CFRelease(assertions_by_process);
    return count;
}
#endif

} // namespace

TEST_CASE("Agent inhibits idle system sleep for the complete turn", "[agent][power]") {
    auto state = std::make_shared<InhibitionState>();
    auto provider = std::make_shared<InhibitionObservingProvider>(state);
    auto inhibitor = std::make_shared<TestSleepInhibitor>(state);
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context(),
        core::agent::ToolResultStore::default_root(),
        inhibitor);

    bool lease_was_active_at_completion = false;
    agent->send_message(
        "Keep this turn awake",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&] { lease_was_active_at_completion = state->active_leases == 1; });

    CHECK(state->acquisitions == 1);
    CHECK(state->reason == "Filo is working on an agent turn");
    CHECK(provider->observed_active_lease());
    CHECK(lease_was_active_at_completion);
    CHECK(state->active_leases == 0);
}

TEST_CASE("Native sleep inhibitor can be acquired without throwing", "[power]") {
    const auto inhibitor = core::power::make_sleep_inhibitor();
    REQUIRE(inhibitor);
#if defined(__APPLE__)
    constexpr std::string_view reason = "Filo native assertion lifecycle test";
    const auto assertions_before = native_assertion_count(reason);
    REQUIRE(assertions_before.has_value());
    {
        const auto lease = inhibitor->inhibit(reason);
        REQUIRE(lease);
        CHECK(lease->is_active());
        CHECK(native_assertion_count(reason) == *assertions_before + 1);
    }
    CHECK(native_assertion_count(reason) == assertions_before);
#else
    const auto lease = inhibitor->inhibit("Filo sleep-inhibition test");
    if (lease) {
        CHECK(lease->is_active());
    }
#endif
}

TEST_CASE("Agent releases sleep inhibition deterministically after cancellation",
          "[agent][power][stop]") {
    auto state = std::make_shared<InhibitionState>();
    auto provider = std::make_shared<RetainingCancellationProvider>();
    auto inhibitor = std::make_shared<TestSleepInhibitor>(state);
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context(),
        core::agent::ToolResultStore::default_root(),
        inhibitor);

    int completion_calls = 0;
    bool lease_was_active_at_completion = false;
    agent->send_message(
        "Stop and release the assertion",
        [&](const std::string&) {
            if (!agent->is_stop_requested()) {
                agent->request_stop();
            }
        },
        [](const std::string&, const std::string&) {},
        [&] {
            ++completion_calls;
            lease_was_active_at_completion = state->active_leases == 1;
        });

    REQUIRE(provider->cancel_calls() == 1);
    CHECK(completion_calls == 1);
    CHECK(lease_was_active_at_completion);
    CHECK(state->active_leases == 0);

    // A provider may retain and accidentally invoke its callback after the
    // terminal event. It must not retain the power assertion or finish twice.
    provider->emit_late_terminal_chunk();
    CHECK(completion_calls == 1);
    CHECK(state->active_leases == 0);
}
