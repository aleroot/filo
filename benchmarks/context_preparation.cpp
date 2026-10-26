#include "core/agent/SemanticHistoryEditor.hpp"
#include "core/session/SessionEfficiencyController.hpp"
#include "core/session/SessionStore.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <new>
#include <string>
#include <string_view>
#include <vector>

namespace allocation_profile {
thread_local bool enabled = false;
thread_local std::uint64_t calls = 0;
thread_local std::uint64_t bytes = 0;
}

// Requested heap bytes, not allocator overhead or resident memory. Track only
// this thread and only the measured operation, excluding corpus construction.
void* operator new(std::size_t size) {
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
        if (allocation_profile::enabled) {
            ++allocation_profile::calls;
            allocation_profile::bytes += size;
        }
        return pointer;
    }
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace {
using core::llm::Message;
using core::session::SessionEfficiencyController;
using core::session::SessionEfficiencyObservation;

std::uint64_t sink = 0;

std::uint64_t hash(std::string_view value, std::uint64_t initial = 14695981039346656037ULL) {
    for (const unsigned char byte : value) {
        initial ^= byte;
        initial *= 1099511628211ULL;
    }
    return initial;
}

std::uint64_t fingerprint(const core::agent::SemanticHistoryEditResult& result) {
    std::uint64_t value = hash(std::to_string(result.superseded_results));
    value = hash(std::to_string(result.characters_removed), value);
    for (const auto& message : result.messages)
        value = hash(core::session::SessionStore::message_json(message), value);
    return value;
}

std::vector<Message> corpus(std::string_view kind) {
    std::vector<Message> history{{.role = "user", .content = "Inspect the current implementation."}};
    const std::size_t count = kind == "short" ? 2 : 128;
    const std::size_t payload = kind == "short" ? 128 : 16 * 1024;
    for (std::size_t i = 0; i < count; ++i) {
        const bool mixed = kind == "mixed";
        const bool untracked = kind == "untracked" || (mixed && i % 3 == 0);
        const std::string id = std::format("call-{}", i);
        const std::size_t file = kind == "unique" ? i : i % 8;
        core::llm::ToolCall call;
        call.id = id;
        call.function.name = untracked ? "run_terminal_command" : "read";
        call.function.arguments = untracked ? R"({"command":"make test"})"
            : std::format(R"({{"path":"src/file-{}.cpp"}})", file);
        history.push_back({.role = "assistant", .tool_calls = {std::move(call)}});
        Message result{.role = "tool", .name = untracked ? "run_terminal_command" : "read", .tool_call_id = id};
        const std::size_t size = mixed && i % 11 == 0 ? 32 : payload;
        const std::string field = mixed && i % 13 == 0 ? "error" : "content";
        result.content = std::format(R"({{"{}":"{}"}})", field, std::string(size, static_cast<char>('a' + i % 26)));
        if (mixed && i % 17 == 0)
            result.continuation_items.push_back({.provider = "test", .kind = "signed", .payload = "opaque-state"});
        if (mixed && i % 19 == 0) result.content = "not a JSON object";
        history.push_back(std::move(result));
    }
    return history;
}

struct Measurement {
    double ns_per_operation;
    double allocations_per_operation;
    double bytes_per_operation;
};

template<class Function>
Measurement measure(int iterations, Function operation) {
    std::vector<double> times;
    std::uint64_t calls = 0, bytes = 0;
    for (int trial = 0; trial < 7; ++trial) {
        allocation_profile::calls = allocation_profile::bytes = 0;
        allocation_profile::enabled = true;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) operation();
        const auto end = std::chrono::steady_clock::now();
        allocation_profile::enabled = false;
        calls += allocation_profile::calls;
        bytes += allocation_profile::bytes;
        times.push_back(std::chrono::duration<double, std::nano>(end - start).count() / iterations);
    }
    std::ranges::sort(times);
    return {times[times.size() / 2], static_cast<double>(calls) / (7 * iterations),
            static_cast<double>(bytes) / (7 * iterations)};
}

SessionEfficiencyObservation observation(std::size_t index) {
    return {.prompt_tokens = index % 37 == 0 ? -1 : static_cast<int32_t>((index * 7919) % 90'000),
            .completion_tokens = static_cast<int32_t>(index % 4000),
            .estimated_history_tokens = 1000 + index * 53 % 180'000,
            .max_context_tokens = index % 19 == 0 ? 0 : 200'000,
            .provider_is_local = index % 11 < 4,
            .provider_supports_prompt_caching = index % 7 < 3};
}

std::uint64_t decision_fingerprint() {
    SessionEfficiencyController controller;
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 10'000; ++i) {
        if (i == 499 || i == 2001 || i == 8191) controller.reset();
        controller.record_turn(observation(i));
        const auto decision = controller.current_decision();
        value = hash(std::format("{}:{}:{}:{}:{}:{}", static_cast<int>(decision.action),
            decision.turn_count, std::bit_cast<std::uint64_t>(decision.waste_factor),
            std::bit_cast<std::uint64_t>(decision.context_utilization),
            decision.estimated_history_tokens, decision.reason), value);
    }
    return value;
}

void print(std::string_view name, Measurement value, std::uint64_t checksum,
           std::size_t input_bytes = 0, std::size_t edited_results = 0) {
    std::cout << std::format(R"({{"name":"{}","median_ns":{:.1f},"allocations":{:.2f},"allocated_bytes":{:.1f},"fingerprint":"{:016x}","input_payload_bytes":{},"edited_results":{}}})",
        name, value.ns_per_operation, value.allocations_per_operation,
        value.bytes_per_operation, checksum, input_bytes, edited_results) << '\n';
}
}

int main() {
    for (const std::string_view kind : {"short", "unique", "superseded", "untracked", "mixed"}) {
        const auto history = corpus(kind);
        // This is the real Agent path: it already owns a request-local copy.
        // Include that first copy equally, and offer the owned vector to edit.
        const auto expected = core::agent::SemanticHistoryEditor::edit(history);
        const std::uint64_t checksum = fingerprint(expected);
        auto owned = history;
        const auto moved = core::agent::SemanticHistoryEditor::edit(std::move(owned));
        if (fingerprint(moved) != checksum) return 1;
        std::size_t payload_bytes = 0;
        for (const auto& message : history) payload_bytes += message.content.size();
        const auto measured = measure(kind == "short" ? 2000 : 100, [&] {
            auto request_history = history;
            const auto result = core::agent::SemanticHistoryEditor::edit(std::move(request_history));
            sink += result.messages.size() + result.characters_removed + result.superseded_results;
        });
        print(kind, measured, checksum, payload_bytes, expected.superseded_results);
    }
    constexpr std::size_t samples = 200'000;
    std::size_t index = 0;
    SessionEfficiencyController controller;
    const auto measured = measure(1, [&] {
        controller.reset();
        // Fresh object for each trial also measures the old vector's complete
        // growth cost instead of letting retained capacity hide it after reset.
        SessionEfficiencyController fresh;
        for (std::size_t i = 0; i < samples; ++i) fresh.record_turn(observation(index++));
        controller = std::move(fresh);
    });
    print("efficiency_200k_samples", measured, decision_fingerprint());
    std::cout << std::format(R"({{"sink":{}}})", sink) << '\n';
}
