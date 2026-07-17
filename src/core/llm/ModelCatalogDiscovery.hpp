#pragma once

#include "ModelCatalogProvider.hpp"
#include "protocols/ApiProtocol.hpp"
#include "core/auth/ICredentialSource.hpp"
#include "core/config/ConfigManager.hpp"

#include <chrono>
#include <memory>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stop_token>
#include <thread>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::llm {

class LLMProvider;

struct ModelCatalogDiscoveryOptions {
    int max_pages = 8;
    int timeout_ms = 2500;
};

/**
 * @brief Optional provider capability for refreshing a remote model catalog.
 *
 * Callers depend on this capability instead of knowing which concrete provider
 * owns the transport. Providers without live catalog discovery simply do not
 * implement it.
 */
class ModelCatalogDiscoverable {
public:
    virtual ~ModelCatalogDiscoverable() = default;

    virtual void discover_models(
        const ModelCatalogDiscoveryOptions& options = {}) const = 0;
};

/**
 * @brief Request catalog discovery when the provider exposes that capability.
 *
 * This is intentionally a no-op for providers without model discovery.
 */
void request_model_catalog_discovery(
    const std::shared_ptr<LLMProvider>& provider,
    const ModelCatalogDiscoveryOptions& options = {});

struct ModelCatalogDiscoveryResult {
    bool attempted = false;
    bool permanent_skip = false;
    int retry_after_seconds = 0;
    int fetched = 0;
    int inserted = 0;
    int updated = 0;
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

enum class ModelCatalogDiscoveryState {
    NeverAttempted,
    Refreshing,
    Succeeded,
    TransientFailure,
    PermanentSkip,
};

struct ProviderModelCatalogSnapshot {
    ModelCatalogDiscoveryState state = ModelCatalogDiscoveryState::NeverAttempted;
    bool refresh_in_progress = false;
    bool checked = false;
    bool attempted = false;
    int consecutive_failures = 0;
    int fetched = 0;
    int inserted = 0;
    int updated = 0;
    std::string error;
    std::chrono::steady_clock::time_point last_attempt{};
    std::chrono::steady_clock::time_point next_retry_at{};
    std::vector<ModelInfo> models;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
    [[nodiscard]] bool refresh_due(
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) const noexcept;
};

class ModelCatalogAvailability {
public:
    static ModelCatalogAvailability& instance();

    [[nodiscard]] bool try_mark_refreshing(std::string_view provider_name);
    void record_result(std::string_view provider_name,
                       const ModelCatalogDiscoveryResult& result,
                       std::vector<ModelInfo> models);

    [[nodiscard]] ProviderModelCatalogSnapshot snapshot(std::string_view provider_name) const;
    [[nodiscard]] std::unordered_map<std::string, ProviderModelCatalogSnapshot> snapshots() const;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, ProviderModelCatalogSnapshot> providers_;
};

[[nodiscard]] ModelCatalogDiscoveryResult discover_and_register_models(
    std::string_view provider_name,
    config::ApiType api_type,
    std::string_view base_url,
    const std::shared_ptr<core::auth::ICredentialSource>& cred_source,
    const protocols::ApiProtocolBase& protocol,
    const ModelCatalogDiscoveryOptions& options = {});

void discover_and_register_models_in_background(
    std::string provider_name,
    config::ApiType api_type,
    std::string base_url,
    std::shared_ptr<core::auth::ICredentialSource> cred_source,
    std::unique_ptr<protocols::ApiProtocolBase> protocol,
    ModelCatalogDiscoveryOptions options = {});

class ModelCatalogDiscoveryService {
public:
    static ModelCatalogDiscoveryService& instance();

    void request_refresh(
        std::string provider_name,
        config::ApiType api_type,
        std::string base_url,
        std::shared_ptr<core::auth::ICredentialSource> cred_source,
        std::unique_ptr<protocols::ApiProtocolBase> protocol,
        ModelCatalogDiscoveryOptions options = {});

private:
    struct Task {
        std::string provider_name;
        config::ApiType api_type = config::ApiType::Unknown;
        std::string base_url;
        std::shared_ptr<core::auth::ICredentialSource> cred_source;
        std::unique_ptr<protocols::ApiProtocolBase> protocol;
        ModelCatalogDiscoveryOptions options;
    };

    ModelCatalogDiscoveryService();
    ~ModelCatalogDiscoveryService() = default;

    ModelCatalogDiscoveryService(const ModelCatalogDiscoveryService&) = delete;
    ModelCatalogDiscoveryService& operator=(const ModelCatalogDiscoveryService&) = delete;

    void run(std::stop_token stop_token);

    std::mutex mutex_;
    std::condition_variable_any cv_;
    std::deque<Task> queue_;
    std::jthread worker_;
};

} // namespace core::llm
