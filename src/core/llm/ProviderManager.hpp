#pragma once

#include "LLMProvider.hpp"
#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <vector>

namespace core::llm {

class ProviderManager {
public:
    static ProviderManager& get_instance() {
        static ProviderManager instance;
        return instance;
    }

    void register_provider(const std::string& name, std::shared_ptr<LLMProvider> provider) {
        std::lock_guard lock(mutex_);
        providers_[name] = std::move(provider);
    }

    std::shared_ptr<LLMProvider> get_provider(const std::string& name) {
        std::lock_guard lock(mutex_);
        auto it = providers_.find(name);
        if (it == providers_.end()) {
            throw std::runtime_error("Provider not found: " + name);
        }
        return it->second;
    }

    [[nodiscard]] bool has_provider(const std::string& name) const {
        std::lock_guard lock(mutex_);
        return providers_.contains(name);
    }

    /// Registered provider names, sorted.
    [[nodiscard]] std::vector<std::string> names() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> out;
        out.reserve(providers_.size());
        for (const auto& [name, provider] : providers_) out.push_back(name);
        std::ranges::sort(out);
        return out;
    }

private:
    ProviderManager() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<LLMProvider>> providers_;
};

} // namespace core::llm
