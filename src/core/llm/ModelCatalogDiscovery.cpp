#include "ModelCatalogDiscovery.hpp"

#include "ModelRegistry.hpp"
#include "OpenAIEndpointUtils.hpp"
#include "core/logging/Logger.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/utils/UriUtils.hpp"

#include <cpr/cpr.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <format>
#include <random>
#include <unordered_map>

namespace core::llm {
namespace {

constexpr auto kCatalogSuccessRefreshTtl = std::chrono::minutes{15};
constexpr auto kCatalogRetryBaseDelay = std::chrono::seconds{30};
constexpr auto kCatalogRetryMaxDelay = std::chrono::minutes{30};

[[nodiscard]] bool is_remote_without_auth(std::string_view base_url,
                                          const core::auth::AuthInfo& auth) {
    return !core::utils::uri::is_loopback_http_url(base_url)
        && auth.headers.empty()
        && auth.query_params.empty();
}

[[nodiscard]] std::string append_query_params(
    std::string url,
    const std::unordered_map<std::string, std::string>& query_params) {
    bool first = url.find('?') == std::string::npos;
    for (const auto& [key, value] : query_params) {
        url += first ? '?' : '&';
        url += core::utils::uri::percent_encode_uri_query_component(key);
        url += '=';
        url += core::utils::uri::percent_encode_uri_query_component(value);
        first = false;
    }
    return url;
}

[[nodiscard]] std::string build_url(std::string_view base_url,
                                    std::string_view path) {
    std::string base = core::utils::str::trim_trailing_slashes(base_url);
    if (path.empty()) return base;
    if (path.front() == '/') {
        base += path;
    } else {
        base += '/';
        base += path;
    }
    return base;
}

[[nodiscard]] bool should_skip_provider(config::ApiType api_type,
                                        std::string_view base_url,
                                        std::string_view protocol_name) {
    return api_type == config::ApiType::Unknown
        || api_type == config::ApiType::Ollama
        || api_type == config::ApiType::LlamaCppLocal
        || protocol_name == "gemini_code_assist"
        || (api_type == config::ApiType::OpenAI
            && openai_endpoint::is_azure_openai_base_url(base_url));
}

[[nodiscard]] int parse_retry_after_seconds(const cpr::Header& headers) {
    for (const auto& [key, value] : headers) {
        if (!core::utils::ascii::iequals(key, "retry-after")) continue;
        try {
            const int seconds = std::stoi(value);
            return std::max(0, seconds);
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

[[nodiscard]] std::chrono::steady_clock::duration retry_delay_for(
    int consecutive_failures,
    int retry_after_seconds) {
    if (retry_after_seconds > 0) {
        return std::chrono::seconds{retry_after_seconds};
    }

    const int exponent = std::clamp(consecutive_failures - 1, 0, 10);
    auto capped = kCatalogRetryBaseDelay * (1 << exponent);
    if (capped > kCatalogRetryMaxDelay) {
        capped = kCatalogRetryMaxDelay;
    }

    const auto max_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capped).count();
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<long long> jitter_ms(1000, std::max<long long>(1000, max_ms));
    return std::chrono::milliseconds{jitter_ms(rng)};
}

} // namespace

bool ProviderModelCatalogSnapshot::refresh_due(
    std::chrono::steady_clock::time_point now) const noexcept {
    return !refresh_in_progress
        && state != ModelCatalogDiscoveryState::PermanentSkip
        && now >= next_retry_at;
}

ModelCatalogAvailability& ModelCatalogAvailability::instance() {
    static ModelCatalogAvailability availability;
    return availability;
}

bool ModelCatalogAvailability::try_mark_refreshing(std::string_view provider_name) {
    if (provider_name.empty()) return false;

    std::unique_lock lock(mutex_);
    auto& snapshot = providers_[std::string(provider_name)];
    const auto now = std::chrono::steady_clock::now();
    if (!snapshot.refresh_due(now)) {
        return false;
    }

    snapshot.state = ModelCatalogDiscoveryState::Refreshing;
    snapshot.refresh_in_progress = true;
    snapshot.last_attempt = now;
    return true;
}

void ModelCatalogAvailability::record_result(
    std::string_view provider_name,
    const ModelCatalogDiscoveryResult& result,
    std::vector<ModelInfo> models) {
    if (provider_name.empty()) return;

    std::string key(provider_name);
    std::unique_lock lock(mutex_);
    const auto existing = providers_.find(key);
    const auto now = std::chrono::steady_clock::now();
    const int previous_failures =
        existing != providers_.end() ? existing->second.consecutive_failures : 0;

    ProviderModelCatalogSnapshot snapshot;
    snapshot.refresh_in_progress = false;
    snapshot.checked = true;
    snapshot.attempted = result.attempted;
    snapshot.fetched = result.fetched;
    snapshot.inserted = result.inserted;
    snapshot.updated = result.updated;
    snapshot.error = result.error;
    snapshot.last_attempt =
        existing != providers_.end() ? existing->second.last_attempt : now;

    if (result.permanent_skip) {
        snapshot.state = ModelCatalogDiscoveryState::PermanentSkip;
        snapshot.consecutive_failures = 0;
        snapshot.next_retry_at = std::chrono::steady_clock::time_point::max();
    } else if (result.ok() && result.attempted) {
        snapshot.state = ModelCatalogDiscoveryState::Succeeded;
        snapshot.consecutive_failures = 0;
        snapshot.next_retry_at = now + kCatalogSuccessRefreshTtl;
    } else {
        snapshot.state = ModelCatalogDiscoveryState::TransientFailure;
        snapshot.consecutive_failures = previous_failures + 1;
        snapshot.next_retry_at = now + retry_delay_for(
            snapshot.consecutive_failures,
            result.retry_after_seconds);
    }

    if (((snapshot.state == ModelCatalogDiscoveryState::TransientFailure)
            || result.permanent_skip)
        && models.empty()
        && existing != providers_.end()) {
        snapshot.models = existing->second.models;
    } else {
        snapshot.models = std::move(models);
    }

    providers_[std::move(key)] = std::move(snapshot);
}

ProviderModelCatalogSnapshot ModelCatalogAvailability::snapshot(
    std::string_view provider_name) const {
    std::shared_lock lock(mutex_);
    if (auto it = providers_.find(std::string(provider_name)); it != providers_.end()) {
        return it->second;
    }
    return {};
}

std::unordered_map<std::string, ProviderModelCatalogSnapshot>
ModelCatalogAvailability::snapshots() const {
    std::shared_lock lock(mutex_);
    return providers_;
}

ModelCatalogDiscoveryResult discover_and_register_models(
    std::string_view provider_name,
    config::ApiType api_type,
    std::string_view base_url,
    const std::shared_ptr<core::auth::ICredentialSource>& cred_source,
    const protocols::ApiProtocolBase& protocol,
    const ModelCatalogDiscoveryOptions& options) {
    ModelCatalogDiscoveryResult discovery;

    const std::string provider_key(provider_name);
    if (should_skip_provider(api_type, base_url, protocol.name())) {
        discovery.permanent_skip = true;
        return discovery;
    }

    core::auth::AuthInfo auth;
    if (cred_source) {
        auth = cred_source->get_auth();
    }
    const bool include_session_only_models = provider_name.starts_with("grok")
        && auth.properties.contains("oauth");
    auto catalog_provider = make_model_catalog_provider(
        api_type, provider_name, include_session_only_models);
    if (!catalog_provider) {
        discovery.permanent_skip = true;
        return discovery;
    }
    if (is_remote_without_auth(base_url, auth)) {
        discovery.error = "missing credentials for remote model discovery";
        return discovery;
    }

    discovery.attempted = true;

    cpr::Header headers = protocol.build_headers(auth);
    headers["Accept"] = "application/json";

    std::vector<ModelInfo> available_models;
    std::string page_token;
    const int max_pages = std::max(1, options.max_pages);
    for (int page = 0; page < max_pages; ++page) {
        const std::string path = catalog_provider->model_list_path(page_token);
        const std::string url = append_query_params(build_url(base_url, path), auth.query_params);

        cpr::Response response = cpr::Get(
            cpr::Url{url},
            headers,
            cpr::Timeout{std::max(1, options.timeout_ms)});

        if (response.error) {
            discovery.error = response.error.message;
            break;
        }
        if (response.status_code < 200 || response.status_code >= 300) {
            discovery.retry_after_seconds = parse_retry_after_seconds(response.header);
            discovery.error = std::format(
                "model discovery for provider '{}' failed with HTTP {}",
                provider_name,
                response.status_code);
            break;
        }

        ModelCatalogResult parsed = catalog_provider->parse_models_response(response.text);
        if (!parsed.ok()) {
            discovery.error = parsed.error;
            break;
        }

        discovery.fetched += static_cast<int>(parsed.models.size());
        for (auto& model : parsed.models) {
            model.provider = provider_key;
            available_models.push_back(model);
            if (ModelRegistry::instance().merge_model(std::move(model))) {
                ++discovery.inserted;
            } else {
                ++discovery.updated;
            }
        }

        if (parsed.next_page_token.empty() || parsed.next_page_token == page_token) {
            break;
        }
        page_token = std::move(parsed.next_page_token);
    }

    ModelCatalogAvailability::instance().record_result(
        provider_key,
        discovery,
        std::move(available_models));

    if (!discovery.ok()) {
        core::logging::debug(
            "Model discovery skipped for provider '{}': {}",
            provider_name,
            discovery.error);
    } else if (discovery.fetched > 0) {
        core::logging::debug(
            "Model discovery for provider '{}': fetched {}, inserted {}, updated {}",
            provider_name,
            discovery.fetched,
            discovery.inserted,
            discovery.updated);
    }

    return discovery;
}

void discover_and_register_models_in_background(
    std::string provider_name,
    config::ApiType api_type,
    std::string base_url,
    std::shared_ptr<core::auth::ICredentialSource> cred_source,
    std::unique_ptr<protocols::ApiProtocolBase> protocol,
    ModelCatalogDiscoveryOptions options) {
    ModelCatalogDiscoveryService::instance().request_refresh(
        std::move(provider_name),
        api_type,
        std::move(base_url),
        std::move(cred_source),
        std::move(protocol),
        options);
}

ModelCatalogDiscoveryService& ModelCatalogDiscoveryService::instance() {
    static ModelCatalogDiscoveryService service;
    return service;
}

ModelCatalogDiscoveryService::ModelCatalogDiscoveryService()
    : worker_([this](std::stop_token stop_token) { run(stop_token); }) {}

void ModelCatalogDiscoveryService::request_refresh(
    std::string provider_name,
    config::ApiType api_type,
    std::string base_url,
    std::shared_ptr<core::auth::ICredentialSource> cred_source,
    std::unique_ptr<protocols::ApiProtocolBase> protocol,
    ModelCatalogDiscoveryOptions options) {
    if (provider_name.empty() || !protocol) {
        return;
    }

    if (!ModelCatalogAvailability::instance().try_mark_refreshing(provider_name)) {
        return;
    }

    {
        std::lock_guard lock(mutex_);
        queue_.push_back(Task{
            .provider_name = std::move(provider_name),
            .api_type = api_type,
            .base_url = std::move(base_url),
            .cred_source = std::move(cred_source),
            .protocol = std::move(protocol),
            .options = options,
        });
    }
    cv_.notify_one();
}

void ModelCatalogDiscoveryService::run(std::stop_token stop_token) {
    while (!stop_token.stop_requested()) {
        Task task;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, stop_token, [this] {
                return !queue_.empty();
            });
            if (stop_token.stop_requested()) {
                break;
            }
            if (queue_.empty()) {
                continue;
            }

            task = std::move(queue_.front());
            queue_.pop_front();
        }

        try {
            if (stop_token.stop_requested()) {
                ModelCatalogAvailability::instance().record_result(
                    task.provider_name,
                    {},
                    {});
                continue;
            }

            const auto result = discover_and_register_models(
                task.provider_name,
                task.api_type,
                task.base_url,
                task.cred_source,
                *task.protocol,
                task.options);

            if (!result.attempted) {
                ModelCatalogAvailability::instance().record_result(
                    task.provider_name,
                    result,
                    {});
            }
        } catch (const std::exception& ex) {
            ModelCatalogDiscoveryResult result;
            result.attempted = true;
            result.error = ex.what();
            ModelCatalogAvailability::instance().record_result(task.provider_name, result, {});
            core::logging::debug(
                "Model discovery failed for provider '{}': {}",
                task.provider_name,
                ex.what());
        } catch (...) {
            ModelCatalogDiscoveryResult result;
            result.attempted = true;
            result.error = "unknown model discovery failure";
            ModelCatalogAvailability::instance().record_result(task.provider_name, result, {});
            core::logging::debug(
                "Model discovery failed for provider '{}': unknown error",
                task.provider_name);
        }
    }
}

} // namespace core::llm
