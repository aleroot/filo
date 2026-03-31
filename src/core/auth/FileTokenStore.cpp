#include "FileTokenStore.hpp"
#include <simdjson.h>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include "../utils/JsonUtils.hpp"

namespace core::auth {

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
    } catch (...) {
        return std::nullopt;
    }
}

void FileTokenStore::save(std::string_view provider_id, const OAuthToken& token) {
    std::filesystem::create_directories(config_dir_);

    auto path = token_path(provider_id);
    auto tmp  = path + ".tmp";

    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot write OAuth token file: " + tmp);

        out << "{\n"
            << "  \"access_token\":  \"" << core::utils::escape_json_string(token.access_token)  << "\",\n"
            << "  \"refresh_token\": \"" << core::utils::escape_json_string(token.refresh_token) << "\",\n"
            << "  \"token_type\":    \"" << core::utils::escape_json_string(token.token_type)    << "\",\n"
            << "  \"expires_at\":    "   << token.expires_at    << ",\n"
            << "  \"device_id\":     \"" << core::utils::escape_json_string(token.device_id)     << "\",\n"
            << "  \"scopes\":        [";

        for (std::size_t i = 0; i < token.scopes.size(); ++i) {
            if (i > 0) out << ", ";
            out << "\"" << core::utils::escape_json_string(token.scopes[i]) << "\"";
        }

        out << "]\n"
            << "}\n";
    }

    std::filesystem::rename(tmp, path); // atomic on POSIX
    chmod(path.c_str(), 0600);
}

void FileTokenStore::clear(std::string_view provider_id) {
    std::error_code ec;
    std::filesystem::remove(token_path(provider_id), ec);
    // ignore ec — if file is already absent, that's fine
}

} // namespace core::auth
