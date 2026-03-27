#include "PolicyConfig.hpp"

#include <algorithm>
#include <cctype>

namespace core::llm::routing {

namespace {

[[nodiscard]] std::string normalise(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        if (ch == '-' || ch == '_' || ch == ' ') {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

} // namespace

std::string_view to_string(Strategy strategy) noexcept {
    switch (strategy) {
        case Strategy::Smart: return "smart";
        case Strategy::Fallback: return "fallback";
        case Strategy::LoadBalance: return "load_balance";
        case Strategy::Latency: return "latency";
    }
    return "fallback";
}

std::optional<Strategy> strategy_from_string(std::string_view value) noexcept {
    const std::string norm = normalise(value);
    if (norm == "smart") return Strategy::Smart;
    if (norm == "fallback" || norm == "fallbackchain") return Strategy::Fallback;
    if (norm == "loadbalance" || norm == "weighted") return Strategy::LoadBalance;
    if (norm == "latency" || norm == "latencybased") return Strategy::Latency;
    return std::nullopt;
}

} // namespace core::llm::routing
