#pragma once

#include <expected>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace core::changes {

struct RecoveryImage {
    std::string hash; // Empty means absent; empty files have a SHA-256 hash.
    unsigned mode = 0;
    bool operator==(const RecoveryImage&) const = default;
};

struct RecoverySnapshot {
    RecoveryImage image;
    std::string bytes;
};

struct RecoveryEntry {
    RecoveryImage before;
    RecoveryImage after;
    bool after_known = false;
};

enum class RecoveryPhase { Recording, Complete, Interrupted, Reverted };

struct RecoveryManifest {
    std::filesystem::path workspace;
    RecoveryPhase phase = RecoveryPhase::Recording;
    bool partial = false;
    bool inventory_complete = false;
    std::vector<std::filesystem::path> roots;
    std::set<std::filesystem::path> uncaptured;
    std::vector<int> groups;
    std::map<std::filesystem::path, RecoveryEntry> entries;
};

// Filesystem persistence only. No tool execution, writer coordination or diff
// presentation; loading always validates the durable schema and workspace.
class RecoveryStore final {
public:
    static constexpr std::size_t kMaxFileBytes = 8 * 1024 * 1024;
    static constexpr std::size_t kMaxFiles = 4096;
    static constexpr std::size_t kRetainedTurns = 8;

    explicit RecoveryStore(std::filesystem::path directory);
    [[nodiscard]] static std::filesystem::path session_root(
        const std::filesystem::path& journal, std::string_view session_id);
    [[nodiscard]] static std::filesystem::path session_root_in(
        const std::filesystem::path& sessions_directory, std::string_view session_id);
    [[nodiscard]] static std::filesystem::path new_directory(const std::filesystem::path& root);
    [[nodiscard]] bool is_internal(const std::filesystem::path& path) const;
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }
    [[nodiscard]] static std::expected<RecoverySnapshot, std::string> read(const std::filesystem::path& path);
    void save(const RecoveryManifest& manifest) const;
    [[nodiscard]] std::expected<RecoveryManifest, std::string> load(const std::filesystem::path& workspace) const;
    void put_original(const RecoverySnapshot& snapshot) const;
    [[nodiscard]] std::expected<std::string, std::string> original(const RecoveryImage& image) const;
    void acknowledge() const;
    void retain(const std::filesystem::path& workspace) const;
private:
    std::filesystem::path directory_;
    std::vector<std::filesystem::path> private_roots_;
};

} // namespace core::changes
