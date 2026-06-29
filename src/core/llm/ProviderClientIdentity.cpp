#include "ProviderClientIdentity.hpp"

#include "../utils/StringUtils.hpp"
#include "../utils/Uuid.hpp"

#include <fstream>
#include <mutex>
#include <utility>

namespace core::llm {

namespace {

[[nodiscard]] std::string load_or_create_installation_id(
    const std::filesystem::path& config_dir,
    std::string_view filename) {
    const std::string generated = core::utils::random_uuid_v4();
    if (config_dir.empty()) return generated;

    try {
        const auto path = config_dir / filename;

        {
            std::ifstream in(path);
            std::string value;
            if (in && std::getline(in, value)) {
                value = core::utils::str::trim_ascii_copy(value);
                if (!value.empty()) return value;
            }
        }

        std::error_code ec;
        std::filesystem::create_directories(config_dir, ec);
        std::ofstream out(path, std::ios::trunc);
        if (out) {
            out << generated << '\n';
        }
    } catch (...) {
        // Identity persistence should never make the transport unusable.
    }

    return generated;
}

class FileBackedProviderClientIdentitySource final
    : public IProviderClientIdentitySource {
public:
    FileBackedProviderClientIdentitySource(std::filesystem::path config_dir,
                                           std::string filename)
        : config_dir_(std::move(config_dir))
        , filename_(std::move(filename)) {}

    [[nodiscard]] std::string installation_id() const override {
        std::lock_guard lock(mutex_);
        if (installation_id_.empty()) {
            installation_id_ = load_or_create_installation_id(config_dir_, filename_);
        }
        return installation_id_;
    }

private:
    std::filesystem::path config_dir_;
    std::string filename_;
    mutable std::mutex mutex_;
    mutable std::string installation_id_;
};

} // namespace

std::shared_ptr<IProviderClientIdentitySource>
make_codex_client_identity_source(std::filesystem::path config_dir) {
    return std::make_shared<FileBackedProviderClientIdentitySource>(
        std::move(config_dir),
        "codex_installation_id");
}

} // namespace core::llm
