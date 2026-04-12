#include "GoogleCodeAssist.hpp"
#include "../logging/Logger.hpp"
#include "../utils/JsonUtils.hpp"
#include <cpr/cpr.h>
#include <simdjson.h>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace core::auth::google_code_assist {

namespace {

constexpr std::string_view kDefaultEndpoint = "https://cloudcode-pa.googleapis.com";
constexpr std::string_view kDocsUrl = "https://goo.gle/gemini-cli-auth-docs#workspace-gca";
constexpr std::string_view kApiVersion = "v1internal";

[[nodiscard]] std::string trim_copy(std::string_view in) {
    std::size_t begin = 0;
    while (begin < in.size()
        && std::isspace(static_cast<unsigned char>(in[begin]))) {
        ++begin;
    }

    std::size_t end = in.size();
    while (end > begin
        && std::isspace(static_cast<unsigned char>(in[end - 1]))) {
        --end;
    }

    return std::string(in.substr(begin, end - begin));
}

[[nodiscard]] std::string client_metadata_json(std::string_view project_id) {
    std::string json =
        R"({"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI")";
    if (!project_id.empty()) {
        json += R"(,"duetProject":")";
        json += core::utils::escape_json_string(project_id);
        json += '"';
    }
    json += '}';
    return json;
}

[[nodiscard]] std::string load_code_assist_payload(std::string_view project_id) {
    std::string json = "{";
    if (!project_id.empty()) {
        json += R"("cloudaicompanionProject":")";
        json += core::utils::escape_json_string(project_id);
        json += R"(",)";
    }
    json += R"("metadata":)";
    json += client_metadata_json(project_id);
    json += '}';
    return json;
}

[[nodiscard]] std::string onboard_user_payload(std::string_view tier_id,
                                               std::string_view project_id) {
    std::string json = R"({"tierId":")";
    json += core::utils::escape_json_string(tier_id);
    json += '"';
    if (!project_id.empty()) {
        json += R"(,"cloudaicompanionProject":")";
        json += core::utils::escape_json_string(project_id);
        json += '"';
    }
    json += R"(,"metadata":)";
    json += client_metadata_json(project_id);
    json += '}';
    return json;
}

[[nodiscard]] cpr::Response post_json(std::string_view url,
                                      std::string_view access_token,
                                      std::string body) {
    return cpr::Post(
        cpr::Url{std::string(url)},
        cpr::Header{
            {"Authorization", "Bearer " + std::string(access_token)},
            {"Content-Type", "application/json"},
        },
        cpr::Body{std::move(body)}
    );
}

[[nodiscard]] TierInfo parse_tier_info(simdjson::dom::element element) {
    TierInfo info;
    std::string_view sv;
    bool bool_value = false;

    if (element["id"].get_string().get(sv) == simdjson::SUCCESS) {
        info.id = std::string(sv);
    }
    if (element["userDefinedCloudaicompanionProject"].get_bool().get(bool_value)
        == simdjson::SUCCESS) {
        info.user_defined_project = bool_value;
    }
    if (element["isDefault"].get_bool().get(bool_value) == simdjson::SUCCESS) {
        info.is_default = bool_value;
    }
    return info;
}

[[nodiscard]] std::string format_endpoint_error(std::string_view method,
                                                const cpr::Response& response) {
    return std::string(method) + " failed (" + std::to_string(response.status_code)
        + "): " + response.text;
}

} // namespace

std::string code_assist_endpoint() {
    if (const char* raw = std::getenv("CODE_ASSIST_ENDPOINT");
        raw && raw[0] != '\0') {
        return raw;
    }
    return std::string(kDefaultEndpoint);
}

std::optional<std::string> configured_project_override() {
    if (const char* raw = std::getenv("GOOGLE_CLOUD_PROJECT");
        raw && raw[0] != '\0') {
        const std::string trimmed = trim_copy(raw);
        if (!trimmed.empty()) return trimmed;
    }
    return std::nullopt;
}

LoadCodeAssistResponseData parse_load_code_assist_response(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::dom::element doc = parser.parse(padded);

    LoadCodeAssistResponseData result;
    std::string_view sv;

    if (doc["cloudaicompanionProject"].get_string().get(sv) == simdjson::SUCCESS) {
        result.project_id = std::string(sv);
    }

    simdjson::dom::element current_tier;
    if (doc["currentTier"].get(current_tier) == simdjson::SUCCESS) {
        result.current_tier = parse_tier_info(current_tier);
    }

    simdjson::dom::array allowed_tiers;
    if (doc["allowedTiers"].get(allowed_tiers) == simdjson::SUCCESS) {
        for (auto entry : allowed_tiers) {
            result.allowed_tiers.push_back(parse_tier_info(entry));
        }
    }

    return result;
}

OnboardUserOperation parse_onboard_user_response(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::dom::element doc = parser.parse(padded);

    OnboardUserOperation result;
    bool done = false;
    std::string_view sv;

    if (doc["done"].get_bool().get(done) == simdjson::SUCCESS) {
        result.done = done;
    }
    if (doc["response"]["cloudaicompanionProject"]["id"].get_string().get(sv)
        == simdjson::SUCCESS) {
        result.project_id = std::string(sv);
    }

    return result;
}

TierInfo select_onboard_tier(const LoadCodeAssistResponseData& response) {
    if (response.current_tier.has_value() && !response.current_tier->id.empty()) {
        return *response.current_tier;
    }

    for (const auto& tier : response.allowed_tiers) {
        if (tier.is_default) return tier;
    }

    return TierInfo{
        .id = "legacy-tier",
        .user_defined_project = true,
        .is_default = false,
    };
}

std::string setup_user(std::string_view access_token,
                       std::shared_ptr<ui::AuthUI> ui) {
    std::string project_id;
    if (const auto configured = configured_project_override(); configured.has_value()) {
        project_id = *configured;
    }

    if (ui) {
        ui->show_instructions("Completing Gemini Code Assist setup for your Google account.");
    }

    const std::string endpoint = code_assist_endpoint();
    const std::string load_url = endpoint + "/" + std::string(kApiVersion) + ":loadCodeAssist";
    cpr::Response load_response = post_json(
        load_url,
        access_token,
        load_code_assist_payload(project_id));
    if (load_response.status_code != 200) {
        throw std::runtime_error(format_endpoint_error("loadCodeAssist", load_response));
    }

    const LoadCodeAssistResponseData load_data =
        parse_load_code_assist_response(load_response.text);
    if (project_id.empty() && !load_data.project_id.empty()) {
        project_id = load_data.project_id;
    }

    const TierInfo tier = select_onboard_tier(load_data);
    if (tier.user_defined_project && project_id.empty()) {
        throw std::runtime_error(
            "This Google account requires setting GOOGLE_CLOUD_PROJECT. See "
            + std::string(kDocsUrl));
    }

    const std::string onboard_url = endpoint + "/" + std::string(kApiVersion) + ":onboardUser";
    for (int attempt = 0; attempt < 60; ++attempt) {
        cpr::Response onboard_response = post_json(
            onboard_url,
            access_token,
            onboard_user_payload(tier.id, project_id));
        if (onboard_response.status_code != 200) {
            throw std::runtime_error(format_endpoint_error("onboardUser", onboard_response));
        }

        const OnboardUserOperation operation =
            parse_onboard_user_response(onboard_response.text);
        if (operation.done) {
            if (!operation.project_id.empty()) {
                return operation.project_id;
            }
            return project_id;
        }

        core::logging::debug(
            "Google Code Assist onboarding still pending for tier '{}'; retrying.",
            tier.id);
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    throw std::runtime_error("Timed out waiting for Google Code Assist onboarding to finish");
}

} // namespace core::auth::google_code_assist
