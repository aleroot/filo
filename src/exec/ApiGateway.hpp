#pragma once

#include <httplib.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace core::config {
struct AppConfig;
}

namespace core::llm {
class ProviderManager;
}

namespace exec::gateway {

struct ProviderCatalog {
    std::unordered_set<std::string> provider_names;
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
