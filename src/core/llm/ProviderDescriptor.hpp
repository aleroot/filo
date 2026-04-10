#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_set>

namespace core::llm {

struct ProviderDescriptor {
    std::string name;
    bool is_local = false;
};

struct ProviderDescriptorHash {
    using is_transparent = void;

    [[nodiscard]] std::size_t operator()(const ProviderDescriptor& descriptor) const noexcept {
        return std::hash<std::string_view>{}(descriptor.name);
    }

#if defined(__cpp_lib_generic_unordered_lookup) && __cpp_lib_generic_unordered_lookup >= 201811L
    [[nodiscard]] std::size_t operator()(std::string_view provider_name) const noexcept {
        return std::hash<std::string_view>{}(provider_name);
    }
#endif
};

struct ProviderDescriptorEqual {
    using is_transparent = void;

    [[nodiscard]] bool operator()(const ProviderDescriptor& lhs,
                                  const ProviderDescriptor& rhs) const noexcept {
        return lhs.name == rhs.name;
    }

#if defined(__cpp_lib_generic_unordered_lookup) && __cpp_lib_generic_unordered_lookup >= 201811L
    [[nodiscard]] bool operator()(const ProviderDescriptor& lhs,
                                  std::string_view rhs) const noexcept {
        return lhs.name == rhs;
    }

    [[nodiscard]] bool operator()(std::string_view lhs,
                                  const ProviderDescriptor& rhs) const noexcept {
        return lhs == rhs.name;
    }
#endif
};

using ProviderDescriptorSet =
    std::unordered_set<ProviderDescriptor, ProviderDescriptorHash, ProviderDescriptorEqual>;

[[nodiscard]] inline ProviderDescriptorSet::const_iterator find_provider(
    const ProviderDescriptorSet& providers,
    std::string_view provider_name) {
#if defined(__cpp_lib_generic_unordered_lookup) && __cpp_lib_generic_unordered_lookup >= 201811L
    return providers.find(provider_name);
#else
    return providers.find(ProviderDescriptor{std::string(provider_name), false});
#endif
}

[[nodiscard]] inline bool contains_provider(
    const ProviderDescriptorSet& providers,
    std::string_view provider_name) {
    return find_provider(providers, provider_name) != providers.end();
}

[[nodiscard]] inline bool is_local_provider(
    const ProviderDescriptorSet& providers,
    std::string_view provider_name) {
    const auto it = find_provider(providers, provider_name);
    return it != providers.end() && it->is_local;
}

} // namespace core::llm
