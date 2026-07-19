#pragma once

#include "ConfigManager.hpp"
#include "core/utils/InterprocessFile.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace core::config {

enum class ModelPersistenceStatus {
    Saved,
    SessionOnly,
    Failed,
};

struct ModelPersistenceResult {
    ModelPersistenceStatus status = ModelPersistenceStatus::Saved;
    std::string detail;
};

// Owns the process-lifetime lease governing model-default persistence. The
// earliest interactive process saves defaults; followers remain session-only.
class ModelDefaultsPersistence {
public:
    explicit ModelDefaultsPersistence(ConfigManager& manager);

    [[nodiscard]] ModelPersistenceResult persist(
        std::string_view provider,
        std::string_view selection,
        std::string_view model = {});

    [[nodiscard]] ModelPersistenceResult persist_local(
        std::string_view model_path,
        std::string_view model_label);

private:
    [[nodiscard]] ModelPersistenceResult ensure_ownership();

    ConfigManager& manager_;
    std::filesystem::path owner_path_;
    std::optional<core::utils::InterprocessFileLock> lease_;
};

} // namespace core::config
