#pragma once

#include "Plan.hpp"
#include "TurnRunner.hpp"
#include "Types.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace core::review {

class Engine {
public:
    struct Options {
        std::unique_ptr<TurnRunner> runner;
        std::function<bool()> cancellation_requested;
        std::function<void(const Progress&)> on_progress;
        GrouperConfig grouper{};
        std::size_t max_parallel_groups = kMaxParallelReviewGroups;
    };

    explicit Engine(Options options);

    [[nodiscard]] CampaignResult run(const CampaignInput& input) const;

private:
    Options options_;
};

} // namespace core::review
