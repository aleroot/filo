#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

namespace core::workspace {

[[nodiscard]] inline std::string lower_path_name(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

[[nodiscard]] inline bool is_sensitive_agent_path(const std::filesystem::path& path) {
    const auto filename = lower_path_name(path.filename().string());
    if (filename == ".env" || filename.starts_with(".env.")) return true;

    constexpr std::array<std::string_view, 12> sensitive_names{
        ".npmrc", ".pypirc", ".netrc", ".mcp.json", "id_rsa", "id_ed25519",
        "id_ecdsa", "id_dsa", "credentials.json", "service-account.json",
        "secrets.json", "secrets.yaml"};
    if (std::ranges::find(sensitive_names, filename) != sensitive_names.end()) {
        return true;
    }

    constexpr std::array<std::string_view, 7> sensitive_extensions{
        ".pem", ".key", ".p12", ".pfx", ".kdbx", ".jks", ".keystore"};
    if (std::ranges::find(sensitive_extensions,
                          lower_path_name(path.extension().string()))
        != sensitive_extensions.end()) {
        return true;
    }

    constexpr std::array<std::string_view, 5> sensitive_directories{
        ".ssh", ".gnupg", ".aws", ".azure", ".kube"};
    for (const auto& component : path) {
        const auto name = lower_path_name(component.string());
        if (std::ranges::find(sensitive_directories, name)
            != sensitive_directories.end()) {
            return true;
        }
    }

    return false;
}

inline constexpr std::string_view kSensitivePathReason =
    "Path is protected by Filo's secure-data policy.";

} // namespace core::workspace
