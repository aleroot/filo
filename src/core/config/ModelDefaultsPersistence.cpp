#include "ModelDefaultsPersistence.hpp"

#include <format>

namespace core::config {

ModelDefaultsPersistence::ModelDefaultsPersistence(ConfigManager& manager)
    : manager_(manager),
      owner_path_(std::filesystem::path(manager.get_config_dir())
                  / "model_defaults.owner.lock") {
    lease_ = core::utils::InterprocessFileLock::try_acquire(owner_path_);
}

ModelPersistenceResult ModelDefaultsPersistence::ensure_ownership() {
    if (lease_) return {};

    std::string error;
    lease_ = core::utils::InterprocessFileLock::try_acquire(owner_path_, &error);
    if (lease_) return {};
    if (!error.empty()) {
        return {
            .status = ModelPersistenceStatus::Failed,
            .detail = std::format(
                "Could not establish model-default ownership: {}", error),
        };
    }
    return {
        .status = ModelPersistenceStatus::SessionOnly,
        .detail = "Model changed for this session only; the first running Filo instance owns saved model defaults.",
    };
}

ModelPersistenceResult ModelDefaultsPersistence::persist(
    std::string_view provider,
    std::string_view selection,
    std::string_view model) {
    if (auto ownership = ensure_ownership();
        ownership.status != ModelPersistenceStatus::Saved) {
        return ownership;
    }

    std::string error;
    if (!manager_.persist_model_defaults(provider, selection, model, &error)) {
        return {
            .status = ModelPersistenceStatus::Failed,
            .detail = std::format("Could not persist model defaults: {}", error),
        };
    }
    return {};
}

ModelPersistenceResult ModelDefaultsPersistence::persist_local(
    std::string_view model_path,
    std::string_view model_label) {
    if (auto ownership = ensure_ownership();
        ownership.status != ModelPersistenceStatus::Saved) {
        return ownership;
    }

    std::string error;
    if (!manager_.persist_local_provider(model_path, model_label, &error)) {
        return {
            .status = ModelPersistenceStatus::Failed,
            .detail = std::format("Could not persist local model: {}", error),
        };
    }
    return {};
}

} // namespace core::config
