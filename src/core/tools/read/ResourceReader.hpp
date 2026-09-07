#pragma once
#include "ReadTypes.hpp"
#include "../WebAccess.hpp"
#include "../../agent/ToolResultStore.hpp"

namespace core::tools::read {
class ResourceReader {
public:
    explicit ResourceReader(core::agent::ToolResultStore store = core::agent::ToolResultStore{}, web::WebAccess* web = nullptr)
        : store_(std::move(store)), web_(web) {}
    [[nodiscard]] std::expected<Resource, std::string> read(
        const std::string& path, const Options& options, const ToolInvocationContext& invocation) const;
private:
    core::agent::ToolResultStore store_;
    web::WebAccess* web_;
};
} // namespace core::tools::read
