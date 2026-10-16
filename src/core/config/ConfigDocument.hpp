#pragma once

#include <vector>
#include <filesystem>

#include "core/utils/JsonValue.hpp"

namespace core::config {

class ConfigSource;

struct ConfigDocument {
    std::vector<core::utils::JsonValue> layers;
    core::utils::JsonValue user_settings;
    core::utils::JsonValue workspace_settings;
    std::vector<core::utils::JsonValue> model_defaults;

    /// Read only the supplied layers. The existing loader owns default,
    /// workspace, and profile resolution; omitted fields remain untouched.
    [[nodiscard]] static ConfigDocument load(const ConfigSource& source, const std::filesystem::path& workspace);

    static void validate(const core::utils::JsonValue& document);
};

} // namespace core::config
