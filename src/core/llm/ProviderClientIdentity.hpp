#pragma once

#include <filesystem>
#include <memory>
#include <string>

namespace core::llm {

class IProviderClientIdentitySource {
public:
    virtual ~IProviderClientIdentitySource() = default;

    [[nodiscard]] virtual std::string installation_id() const = 0;
};

[[nodiscard]] std::shared_ptr<IProviderClientIdentitySource>
make_codex_client_identity_source(std::filesystem::path config_dir);

} // namespace core::llm
