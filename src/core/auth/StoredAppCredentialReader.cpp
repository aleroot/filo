#include "StoredAppCredentialReader.hpp"

#include "../utils/PathUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace core::auth {

namespace {

[[nodiscard]] bool is_keyword(std::string_view source, std::string_view keyword) {
    return core::utils::str::to_lower_ascii_copy(source) == keyword;
}

/// The whole content of a regular file; nullopt when it is not one or is unreadable.
[[nodiscard]] std::optional<std::string> read_regular_file(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return std::nullopt;
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

[[nodiscard]] std::string join(const std::vector<std::string>& parts, std::string_view separator) {
    std::string joined;
    for (const auto& part : parts) {
        if (!joined.empty()) joined += separator;
        joined += part;
    }
    return joined;
}

} // namespace

StoredAppCredentialReader::StoredAppCredentialReader(std::shared_ptr<const IKeyring> keyring)
    : keyring_(std::move(keyring)) {
    if (!keyring_) throw std::logic_error("StoredAppCredentialReader needs a keyring");
}

AppCredential StoredAppCredentialReader::read(std::string_view source) const {
    namespace fs = std::filesystem;
    using core::utils::path::abbreviate_user_path;

    const std::string app(app_name());
    const std::string file(file_name());
    // Every location that held nothing, in the order tried, for the final error.
    std::vector<std::string> misses;
    std::vector<KeyringEntry> tried_entries;

    const auto from_keyring = [&](const fs::path& directory) -> std::optional<AppCredential> {
        const auto entry = keyring_entry(directory);
        if (!entry.has_value() || keyring_->description().empty()) return std::nullopt;
        if (std::ranges::find(tried_entries, *entry) != tried_entries.end()) return std::nullopt;
        tried_entries.push_back(*entry);

        const std::string where = std::string(keyring_->description()) + " (" +
                                  describe_keyring_entry(*entry) + ")";
        const auto secret = keyring_->read(*entry);
        if (!secret.has_value()) {
            misses.push_back(where + " could not be read: " + secret.error());
        } else if (!secret->has_value() || (*secret)->empty()) {
            misses.push_back(where + " holds no such item");
        } else {
            return parse(**secret, where);
        }
        return std::nullopt;
    };
    const auto from_file = [&](const fs::path& path) -> std::optional<AppCredential> {
        const auto payload = read_regular_file(path);
        if (payload.has_value() && !payload->empty()) {
            return parse(*payload, abbreviate_user_path(path));
        }
        misses.push_back(abbreviate_user_path(path) +
                         (payload.has_value() ? " is empty" : " does not exist"));
        return std::nullopt;
    };
    const auto nothing_found = [&](std::string_view advice) {
        return std::runtime_error("Found no " + app + " credential: " + join(misses, "; ") +
                                  ". " + std::string(advice));
    };

    if (is_keyword(source, "keyring") || is_keyword(source, "keychain")) {
        if (!keyring_entry(default_directory()).has_value()) {
            throw std::runtime_error("The " + app + " app keeps no credential in a keyring. "
                                     "Point --from at its " + file + " file, or use \"auto\".");
        }
        if (keyring_->description().empty()) {
            throw std::runtime_error("Filo cannot read a keyring on this platform. Point --from "
                                     "at the " + app + " app's " + file + " file instead.");
        }
        if (auto imported = from_keyring(default_directory())) return std::move(*imported);
        throw nothing_found("Sign in to the " + app + " app once, or point --from at its " +
                            file + " file.");
    }

    if (!source.empty() && !is_keyword(source, "auto")) {
        const fs::path path = core::utils::path::expand_user_path(source);
        std::error_code ec;
        if (fs::is_directory(path, ec)) {
            if (fs::is_regular_file(path / file, ec)) {
                if (auto imported = from_file(path / file)) return std::move(*imported);
            }
            // No file: an app that stores into the keyring, or an app bundle.
            if (auto imported = from_keyring(path)) return std::move(*imported);
        } else if (fs::exists(path, ec)) {
            if (auto imported = from_file(path)) return std::move(*imported);
            throw std::runtime_error("Cannot import the " + app + " credential: " +
                                     join(misses, "; ") + ".");
        } else {
            throw std::runtime_error(
                "Cannot import the " + app + " credential: " + abbreviate_user_path(path) +
                " does not exist. Expected the app's " + file +
                " file, its directory, \"keyring\" or \"auto\".");
        }
    }

    const fs::path directory = default_directory();
    if (auto imported = from_keyring(directory)) return std::move(*imported);
    if (auto imported = from_file(directory / file)) return std::move(*imported);
    throw nothing_found("Sign in to the " + app +
                        " app once, or drop --from and complete a browser login instead.");
}

} // namespace core::auth
