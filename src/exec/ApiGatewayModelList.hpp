#pragma once

#include <map>
#include <string>
#include <utility>

namespace exec::gateway {

struct GatewayModelListEntry {
    std::string id;
    std::string provider;
    bool discovered = false;
};

using GatewayModelListKey = std::pair<std::string, std::string>;
using GatewayModelList = std::map<GatewayModelListKey, GatewayModelListEntry>;

inline void upsert_gateway_model_list_entry(GatewayModelList& models,
                                            std::string id,
                                            std::string provider,
                                            bool discovered) {
    if (id.empty() || provider.empty()) {
        return;
    }

    GatewayModelListKey key{id, provider};
    auto [it, inserted] = models.emplace(
        std::move(key),
        GatewayModelListEntry{
            .id = std::move(id),
            .provider = std::move(provider),
            .discovered = discovered,
        });
    if (!inserted && discovered) {
        it->second.discovered = true;
    }
}

} // namespace exec::gateway
