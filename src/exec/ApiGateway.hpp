#pragma once

#include "../core/llm/ProviderDescriptor.hpp"

#include <httplib.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace core::config {
struct AppConfig;
}

namespace core::llm {
class ProviderManager;
}

namespace exec::gateway {

struct ProviderCatalog {
    core::llm::ProviderDescriptorSet providers;
    std::unordered_map<std::string, std::string> provider_default_models;
};

class ApiGateway final {
public:
    ApiGateway(const core::config::AppConfig& config, core::llm::ProviderManager& provider_manager, ProviderCatalog provider_catalog);
    void register_routes(httplib::Server& server) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace exec::gateway
