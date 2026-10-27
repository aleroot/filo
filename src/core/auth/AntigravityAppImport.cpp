#include "AntigravityAppImport.hpp"

#include "IdTokenClaims.hpp"
#include "../utils/Base64.hpp"
#include "../utils/PathUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <simdjson.h>

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>

namespace core::auth::antigravity {

namespace {

constexpr std::string_view kAppTokenFileName = "jetski-standalone-oauth-token";
constexpr std::string_view kKeyringPrefix = "go-keyring-base64:";
constexpr std::string_view kKeychainDescription =
    "the macOS login keychain (service=gemini, account=antigravity)";

[[nodiscard]] bool keyword_is(std::string_view value, std::string_view keyword) {
    return core::utils::str::to_lower_ascii_copy(value) == keyword;
}

/// Read a whole file; empty when missing or unreadable.
[[nodiscard]] std::string read_whole_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/// The credential the Antigravity app keeps live in the macOS login keychain.
/// Reading it needs no authorization prompt, and unlike the app's file it stays
/// current while the app runs. Empty when absent, unreadable, or not on macOS.
[[nodiscard]] std::string read_keychain_item() {
#ifdef __APPLE__
    constexpr const char* command =
        "/usr/bin/security find-generic-password -s gemini -a antigravity -w 2>/dev/null";
    std::FILE* pipe = ::popen(command, "r");
    if (pipe == nullptr) return {};
    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    if (::pclose(pipe) != 0) return {};
    while (!output.empty() &&
           (output.back() == '\n' || output.back() == '\r' || output.back() == ' ')) {
        output.pop_back();
    }
    return output;
#else
    return {};
#endif
}

[[nodiscard]] std::int64_t to_unix_utc(std::tm& calendar) {
#ifdef _WIN32
    return static_cast<std::int64_t>(::_mkgmtime(&calendar));
#else
    return static_cast<std::int64_t>(::timegm(&calendar));
#endif
}

/// Unix seconds for an RFC 3339 timestamp such as the app's
/// `2026-10-27T09:55:12.753027+04:00`; 0 when it cannot be parsed.
[[nodiscard]] std::int64_t parse_rfc3339(std::string_view text) {
    std::tm calendar;
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
    const std::string copy(text);
    if (std::sscanf(copy.c_str(), "%d-%d-%dT%d:%d:%d",
                    &year, &month, &day, &hour, &minute, &second) != 6) {
        return 0;
    }
    calendar.tm_year = year - 1900;
    calendar.tm_mon  = month - 1;
    calendar.tm_mday = day;
    calendar.tm_hour = hour;
    calendar.tm_min  = minute;
    calendar.tm_sec  = second;
    const std::int64_t utc = to_unix_utc(calendar);
    if (utc <= 0) return 0;

    // The zone follows an optional fractional part: `Z`, or `±HH:MM` / `±HHMM`.
    const auto time_part = text.find('T');
    if (time_part == std::string_view::npos) return utc;
    if (text.back() == 'Z' || text.back() == 'z') return utc;
    const auto sign = text.find_last_of("+-");
    if (sign == std::string_view::npos || sign <= time_part) return utc;
    int zone_hours = 0;
    int zone_minutes = 0;
    if (std::sscanf(std::string(text.substr(sign + 1)).c_str(), "%d:%d",
                    &zone_hours, &zone_minutes) < 1) {
        return utc;
    }
    const std::int64_t offset = zone_hours * 3600 + zone_minutes * 60;
    // A head-of-UTC zone means the wall clock ran ahead, so UTC is earlier.
    return text[sign] == '+' ? utc - offset : utc + offset;
}

[[nodiscard]] std::string json_string(simdjson::dom::element element, std::string_view key) {
    std::string_view value;
    if (element[key].get(value) == simdjson::SUCCESS) return std::string(value);
    return {};
}

} // namespace

std::string app_token_file(std::string_view home_dir) {
    return (std::filesystem::path(home_dir) / ".gemini" / std::string(kAppTokenFileName)).string();
}

AppImport parse_app_token(std::string_view payload, std::string_view source) {
    std::string json(payload);
    if (json.starts_with(kKeyringPrefix)) {
        const auto decoded =
            core::utils::Base64::decode(std::string_view(json).substr(kKeyringPrefix.size()));
        if (!decoded.has_value()) {
            throw std::runtime_error(
                "The Antigravity credential in " + std::string(source) +
                " is not valid base64 after its \"go-keyring-base64:\" prefix.");
        }
        json = *decoded;
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS || !doc.is_object()) {
        throw std::runtime_error(
            "The Antigravity credential in " + std::string(source) +
            " is not a JSON object, so it cannot be imported.");
    }

    // The app nests the OAuth token under "token"; accept a flat document too so
    // a hand-written file works as well.
    simdjson::dom::element token = doc;
    simdjson::dom::element nested;
    if (doc["token"].get(nested) == simdjson::SUCCESS && nested.is_object()) {
        token = nested;
    }

    AppImport imported;
    imported.access_token  = json_string(token, "access_token");
    imported.refresh_token = json_string(token, "refresh_token");
    if (imported.access_token.empty()) {
        throw std::runtime_error(
            "The Antigravity credential in " + std::string(source) +
            " carries no access_token. Sign in to the Antigravity app once, or drop "
            "--from and complete a browser login instead.");
    }
    if (const std::string expiry = json_string(token, "expiry"); !expiry.empty()) {
        imported.expires_at = parse_rfc3339(expiry);
    }
    if (const std::string id_token = json_string(doc, "id_token"); !id_token.empty()) {
        const IdTokenClaims claims = parse_id_token_claims(id_token);
        imported.client_id = claims.audience;
        imported.email     = claims.email;
    }
    imported.source = std::string(source);
    return imported;
}

AppImport import_from_app(std::string_view source) {
    const auto from_keychain = []() -> std::optional<AppImport> {
        if (const std::string payload = read_keychain_item(); !payload.empty()) {
            return parse_app_token(payload, kKeychainDescription);
        }
        return std::nullopt;
    };
    const auto from_file = [](const std::filesystem::path& path) -> std::optional<AppImport> {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) return std::nullopt;
        const std::string payload = read_whole_file(path);
        if (payload.empty()) return std::nullopt;
        return parse_app_token(payload, core::utils::path::abbreviate_user_path(path));
    };
    const auto detected_file = [] {
        return core::utils::path::home_directory() / ".gemini" / std::string(kAppTokenFileName);
    };

    if (keyword_is(source, "keychain")) {
        if (auto imported = from_keychain()) return *imported;
        throw std::runtime_error(
            "Found no Antigravity credential in " + std::string(kKeychainDescription) +
            ". Sign in to the Antigravity app once, or point --from at its " +
            std::string(kAppTokenFileName) + " file.");
    }

    const bool auto_detect = source.empty() || keyword_is(source, "auto");
    if (!auto_detect) {
        const std::filesystem::path path = core::utils::path::expand_user_path(source);
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec)) {
            // ~/.gemini holds the file directly. An Antigravity.app bundle holds no user
            // credential at all, so a bundle path falls through to auto-detection.
            if (auto imported = from_file(path / std::string(kAppTokenFileName))) return *imported;
        } else if (std::filesystem::exists(path, ec)) {
            if (auto imported = from_file(path)) return *imported;
        } else {
            throw std::runtime_error(
                "Cannot import an Antigravity credential: " +
                core::utils::path::abbreviate_user_path(path) + " does not exist. Expected the "
                "app's " + std::string(kAppTokenFileName) + " file, its directory, or "
                "\"keychain\".");
        }
    }

    // The app refreshes the keychain item while it runs but writes its file only
    // at launch, so the keychain holds the newer access token.
    if (auto imported = from_keychain()) return *imported;
    if (auto imported = from_file(detected_file())) return *imported;

    throw std::runtime_error(
        "Found no Antigravity credential: " + std::string(kKeychainDescription) +
        " holds none and " +
        core::utils::path::abbreviate_user_path(detected_file()) +
        " does not exist. Sign in to the Antigravity app once, or drop --from and "
        "complete a browser login instead.");
}

} // namespace core::auth::antigravity
