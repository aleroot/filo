#include "FileTokenStore.hpp"
#include "core/logging/Logger.hpp"
#include "core/utils/Uuid.hpp"
#include <simdjson.h>
#include <fstream>
#include <filesystem>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include "../utils/JsonUtils.hpp"

namespace core::auth {

namespace {

class PosixTokenStoreLock final : public ITokenStoreLock {
public:
    explicit PosixTokenStoreLock(const std::string& path) {
        fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd_ < 0 || ::flock(fd_, LOCK_EX) != 0) {
            if (fd_ >= 0) ::close(fd_);
            throw std::runtime_error("Cannot lock OAuth credential store: " + path);
        }
    }

    ~PosixTokenStoreLock() override {
        if (fd_ >= 0) {
            (void)::flock(fd_, LOCK_UN);
            (void)::close(fd_);
        }
    }

private:
    int fd_ = -1;
};

void sync_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot reopen OAuth token file: " + path);
    const int result = ::fsync(fd);
    (void)::close(fd);
    if (result != 0) throw std::runtime_error("Cannot sync OAuth token file: " + path);
}

} // namespace

FileTokenStore::FileTokenStore(std::string config_dir)
    : config_dir_(std::move(config_dir)) {}

std::string FileTokenStore::token_path(std::string_view provider_id) const {
    return config_dir_ + "/oauth_" + std::string(provider_id) + ".json";
}

std::optional<OAuthToken> FileTokenStore::load(std::string_view provider_id) {
    auto path = token_path(provider_id);
    if (!std::filesystem::exists(path)) return std::nullopt;

    try {
        simdjson::padded_string json = simdjson::padded_string::load(path);
        simdjson::dom::parser parser;
        simdjson::dom::element doc = parser.parse(json);

        OAuthToken token;
        std::string_view sv;
        int64_t v;

        if (doc["access_token"].get_string().get(sv) == simdjson::SUCCESS)
            token.access_token = std::string(sv);
        if (doc["refresh_token"].get_string().get(sv) == simdjson::SUCCESS)
            token.refresh_token = std::string(sv);
        if (doc["token_type"].get_string().get(sv) == simdjson::SUCCESS)
            token.token_type = std::string(sv);
        if (doc["expires_at"].get_int64().get(v) == simdjson::SUCCESS)
            token.expires_at = v;
        if (doc["device_id"].get_string().get(sv) == simdjson::SUCCESS)
            token.device_id = std::string(sv);
        if (doc["account_id"].get_string().get(sv) == simdjson::SUCCESS)
            token.account_id = std::string(sv);
        if (doc["organization_id"].get_string().get(sv) == simdjson::SUCCESS) {
            token.organization_id = std::string(sv);
        } else if (doc["organization_uuid"].get_string().get(sv) == simdjson::SUCCESS) {
            // Backward compatibility with tokens persisted before the rename.
            token.organization_id = std::string(sv);
        }
        if (doc["project_id"].get_string().get(sv) == simdjson::SUCCESS) {
            token.project_id = std::string(sv);
        } else if (doc["project"].get_string().get(sv) == simdjson::SUCCESS) {
            token.project_id = std::string(sv);
        } else if (doc["cloudaicompanion_project"].get_string().get(sv) == simdjson::SUCCESS) {
            token.project_id = std::string(sv);
        }
        if (doc["issuer"].get_string().get(sv) == simdjson::SUCCESS)
            token.issuer = std::string(sv);
        if (doc["client_id"].get_string().get(sv) == simdjson::SUCCESS)
            token.client_id = std::string(sv);
        if (doc["user_id"].get_string().get(sv) == simdjson::SUCCESS)
            token.user_id = std::string(sv);
        if (doc["email"].get_string().get(sv) == simdjson::SUCCESS)
            token.email = std::string(sv);
        if (doc["principal_type"].get_string().get(sv) == simdjson::SUCCESS)
            token.principal_type = std::string(sv);
        if (doc["principal_id"].get_string().get(sv) == simdjson::SUCCESS)
            token.principal_id = std::string(sv);
        if (doc["team_id"].get_string().get(sv) == simdjson::SUCCESS)
            token.team_id = std::string(sv);
        simdjson::dom::array scopes;
        if (doc["scopes"].get(scopes) == simdjson::SUCCESS) {
            for (auto entry : scopes) {
                if (entry.get_string().get(sv) == simdjson::SUCCESS) {
                    token.scopes.emplace_back(sv);
                }
            }
        }

        if (token.access_token.empty()) return std::nullopt;
        return token;
    } catch (const std::exception& error) {
        const auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string backup = path + ".corrupt." + std::to_string(timestamp);
        std::error_code ec;
        std::filesystem::rename(path, backup, ec);
        core::logging::warn(
            "Ignoring malformed OAuth credential file '{}': {}{}",
            path,
            error.what(),
            ec ? std::string{} : std::string(" (backed up to '") + backup + "')");
        return std::nullopt;
    }
}

void FileTokenStore::save(std::string_view provider_id, const OAuthToken& token) {
    std::filesystem::create_directories(config_dir_);

    auto path = token_path(provider_id);
    auto tmp  = path + ".tmp." + core::utils::random_uuid_v4();

    // Create the file with its final private mode before writing any secret.
    // O_EXCL also protects against accidentally following a pre-created path.
    const int tmp_fd = ::open(tmp.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (tmp_fd < 0) {
        throw std::runtime_error("Cannot create OAuth token file: " + tmp);
    }
    if (::close(tmp_fd) != 0) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw std::runtime_error("Cannot close OAuth token file: " + tmp);
    }

    try {
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out) throw std::runtime_error("Cannot write OAuth token file: " + tmp);

            out << "{\n"
                << "  \"access_token\":  \"" << core::utils::escape_json_string(token.access_token)  << "\",\n"
                << "  \"refresh_token\": \"" << core::utils::escape_json_string(token.refresh_token) << "\",\n"
                << "  \"token_type\":    \"" << core::utils::escape_json_string(token.token_type)    << "\",\n"
                << "  \"expires_at\":    "   << token.expires_at    << ",\n"
                << "  \"device_id\":     \"" << core::utils::escape_json_string(token.device_id)     << "\",\n"
                << "  \"account_id\":    \"" << core::utils::escape_json_string(token.account_id)    << "\",\n"
                << "  \"organization_id\": \"" << core::utils::escape_json_string(token.organization_id) << "\",\n"
                << "  \"project_id\":    \"" << core::utils::escape_json_string(token.project_id)    << "\",\n"
                << "  \"issuer\":        \"" << core::utils::escape_json_string(token.issuer)        << "\",\n"
                << "  \"client_id\":     \"" << core::utils::escape_json_string(token.client_id)     << "\",\n"
                << "  \"user_id\":       \"" << core::utils::escape_json_string(token.user_id)       << "\",\n"
                << "  \"email\":         \"" << core::utils::escape_json_string(token.email)         << "\",\n"
                << "  \"principal_type\": \"" << core::utils::escape_json_string(token.principal_type) << "\",\n"
                << "  \"principal_id\":  \"" << core::utils::escape_json_string(token.principal_id)  << "\",\n"
                << "  \"team_id\":       \"" << core::utils::escape_json_string(token.team_id)       << "\",\n"
                << "  \"scopes\":        [";

            for (std::size_t i = 0; i < token.scopes.size(); ++i) {
                if (i > 0) out << ", ";
                out << "\"" << core::utils::escape_json_string(token.scopes[i]) << "\"";
            }

            out << "]\n"
                << "}\n";
            if (!out) throw std::runtime_error("Cannot write OAuth token file: " + tmp);
        }

        sync_file(tmp);
        std::filesystem::rename(tmp, path); // atomic on POSIX
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw;
    }
}

void FileTokenStore::clear(std::string_view provider_id) {
    std::error_code ec;
    std::filesystem::remove(token_path(provider_id), ec);
    if (ec) {
        throw std::runtime_error(
            "Cannot remove OAuth token file: " + token_path(provider_id)
            + ": " + ec.message());
    }
}

std::unique_ptr<ITokenStoreLock>
FileTokenStore::acquire_refresh_lock(std::string_view provider_id) {
    std::filesystem::create_directories(config_dir_);
    const std::string path = token_path(provider_id) + ".lock";
    return std::make_unique<PosixTokenStoreLock>(path);
}

} // namespace core::auth
