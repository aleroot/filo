#pragma once

#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::tools {

class TempFileAccessRegistry {
public:
    static TempFileAccessRegistry& instance();

    [[nodiscard]] static std::string session_key(std::string_view session_id);

    void grant_read(std::string_view session_id, const std::filesystem::path& path);
    [[nodiscard]] bool can_read(std::string_view session_id, const std::filesystem::path& path);
    void clear_session(std::string_view session_id);

private:
    struct Grant {
        std::filesystem::path path;
        std::chrono::steady_clock::time_point expires_at;
    };

    TempFileAccessRegistry() = default;

    void prune_locked(const std::string& key, std::chrono::steady_clock::time_point now);

    std::mutex mutex_;
    std::unordered_map<std::string, std::vector<Grant>> grants_;
};

} // namespace core::tools
