#include "AntigravityAppImport.hpp"

#include "IdTokenClaims.hpp"
#include "../utils/Base64.hpp"
#include "../utils/PathUtils.hpp"

#include <simdjson.h>

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace core::auth::antigravity {

namespace {

constexpr std::string_view kAppTokenFileName = "jetski-standalone-oauth-token";
constexpr std::string_view kKeyringPrefix = "go-keyring-base64:";

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

std::filesystem::path AppCredentialReader::default_directory() const {
    return core::utils::path::home_directory() / ".gemini";
}

std::string_view AppCredentialReader::file_name() const noexcept {
    return kAppTokenFileName;
}

std::optional<KeyringEntry> AppCredentialReader::keyring_entry(
    const std::filesystem::path& /*directory*/) const {
    // One item per user, whatever the data directory.
    return KeyringEntry{.service = "gemini", .account = "antigravity"};
}

AppCredential AppCredentialReader::parse(std::string_view payload,
                                         std::string_view origin) const {
    return parse_app_token(payload, origin);
}

AppCredential parse_app_token(std::string_view payload, std::string_view origin) {
    std::string json(payload);
    if (json.starts_with(kKeyringPrefix)) {
        const auto decoded =
            core::utils::Base64::decode(std::string_view(json).substr(kKeyringPrefix.size()));
        if (!decoded.has_value()) {
            throw std::runtime_error(
                "The Antigravity credential in " + std::string(origin) +
                " is not valid base64 after its \"go-keyring-base64:\" prefix.");
        }
        json = *decoded;
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS || !doc.is_object()) {
        throw std::runtime_error(
            "The Antigravity credential in " + std::string(origin) +
            " is not a JSON object, so it cannot be imported.");
    }

    // The app nests the OAuth token under "token"; accept a flat document too so
    // a hand-written file works as well.
    simdjson::dom::element token = doc;
    simdjson::dom::element nested;
    if (doc["token"].get(nested) == simdjson::SUCCESS && nested.is_object()) {
        token = nested;
    }

    AppCredential imported;
    imported.origin = std::string(origin);
    imported.token.access_token  = json_string(token, "access_token");
    imported.token.refresh_token = json_string(token, "refresh_token");
    if (imported.token.access_token.empty()) {
        throw std::runtime_error(
            "The Antigravity credential in " + std::string(origin) +
            " carries no access_token. Sign in to the Antigravity app once, or drop "
            "--from and complete a browser login instead.");
    }
    if (const std::string expiry = json_string(token, "expiry"); !expiry.empty()) {
        imported.token.expires_at = parse_rfc3339(expiry);
    }
    if (const std::string id_token = json_string(doc, "id_token"); !id_token.empty()) {
        const IdTokenClaims claims = parse_id_token_claims(id_token);
        imported.token.client_id = claims.audience;
        imported.token.user_id   = claims.subject;
        imported.token.email     = claims.email;
    }
    return imported;
}

} // namespace core::auth::antigravity
