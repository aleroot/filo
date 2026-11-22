#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "core/utils/JsonValue.hpp"

namespace core::config {

/// Read-only supplied configuration, with a separate private directory for
/// mutable state. The ordinary user configuration remains a fallback layer.
class ConfigSource {
public:
    using Documents = std::map<std::string, core::utils::JsonValue>;

    [[nodiscard]] static std::shared_ptr<const ConfigSource> open(
        const std::filesystem::path& input,
        const std::filesystem::path& fallback_directory);

    [[nodiscard]] const std::filesystem::path& input_path() const { return input_; }
    [[nodiscard]] const std::filesystem::path& state_directory() const { return state_; }
    [[nodiscard]] const std::filesystem::path& fallback_directory() const { return fallback_; }
    [[nodiscard]] const Documents& documents() const { return documents_; }
    [[nodiscard]] std::filesystem::path workspace_state_directory(const std::filesystem::path& workspace) const;

    /// Strict reads: absence is an omitted layer; malformed input is an error.
    [[nodiscard]] static std::optional<core::utils::JsonValue> read_document(
        const std::filesystem::path& path);

    [[nodiscard]] std::filesystem::path resolve_file(std::string_view name) const;
    /// Where the login `id` lives: this config's state when it supplies the
    /// login or signed in or out on its own, else the fallback store, used in
    /// place. Refresh tokens are single-use, so an inherited login is never
    /// copied: its refreshes rotate the fallback's token. Explicit logins and
    /// logout still stay in this config's state; opening the config folds any
    /// state copy of a fallback login back into one store (newest wins).
    [[nodiscard]] std::filesystem::path credential_directory(std::string_view id) const;
    void claim_credential(std::string_view id) const;
    void verify_available() const;

private:
    ConfigSource(std::filesystem::path input, std::filesystem::path state,
                 std::filesystem::path fallback, Documents documents,
                 std::vector<std::string> credential_files);
    std::filesystem::path input_, state_, fallback_;
    Documents documents_;
    std::vector<std::string> credential_files_;
    bool directory_input_;
};

} // namespace core::config
