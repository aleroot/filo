#include "QwenSubscription.hpp"

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/utils/UriUtils.hpp"

#include <simdjson.h>

#include <cmath>

namespace core::llm::qwen_subscription {
namespace {

using core::utils::uri::percent_encode_uri_query_component;

/// Qwen Code's attribution for terminal clients (oauth-config.ts).
constexpr std::string_view kTrackingQuery = "entry=qwen_code&paywallScene=cli";

[[nodiscard]] std::string non_blank(simdjson::dom::object object, std::string_view key) {
    std::string_view value;
    if (object[key].get(value) != simdjson::SUCCESS) return {};
    return core::utils::str::trim_ascii_view(value).empty() ? std::string{} : std::string(value);
}

/// Gateway request ids are echoed to the user, so accept only their plain shape.
[[nodiscard]] bool is_safe_request_id(std::string_view id) {
    if (id.empty() || id.size() > 128) return false;
    for (const char ch : id) {
        const bool safe = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
            || (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-';
        if (!safe) return false;
    }
    return true;
}

[[nodiscard]] std::string reset_phrase(std::int64_t reset_at, std::int64_t now) {
    if (reset_at <= now) return {};
    const std::int64_t remaining = reset_at - now;
    const std::int64_t total_hours = (remaining + 3599) / 3600;
    const std::int64_t days = remaining >= 86400 ? total_hours / 24 : 0;
    const std::int64_t hours = days > 0 ? total_hours % 24 : total_hours;
    const auto unit = [](std::int64_t count, std::string_view name) {
        return std::to_string(count) + ' ' + std::string(name) + (count == 1 ? "" : "s");
    };
    std::string time = days > 0 ? unit(days, "day") + " and " + unit(hours, "hour")
                                : unit(hours, "hour");
    return " Your credits are expected to reset in " + time + '.';
}

} // namespace

std::string authorization_url(std::string_view user_code) {
    return std::string(kWebOrigin) + "/authorize?user_code="
        + percent_encode_uri_query_component(user_code) + "&client=cli&"
        + std::string(kTrackingQuery);
}

std::string plans_url() {
    return std::string(kWebOrigin) + "/?memberPlan=qwen_code&" + std::string(kTrackingQuery);
}

bool membership_qualifies(std::string_view member_type) {
    const auto plan = core::utils::str::trim_ascii_view(member_type);
    return core::utils::ascii::iequals(plan, "plus") || core::utils::ascii::iequals(plan, "pro");
}

std::string display_plan(std::string_view member_type) {
    std::string plan = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(member_type));
    if (!plan.empty() && plan.front() >= 'a' && plan.front() <= 'z') {
        plan.front() = static_cast<char>(plan.front() - 'a' + 'A');
    }
    return plan;
}

std::optional<int> thinking_budget(std::string_view effort) {
    const std::string value = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(effort));
    if (value == "none" || value == "off" || value == "disabled") return std::nullopt;
    if (value == "minimal" || value == "low") return 4096;
    if (value == "high" || value == "xhigh" || value == "max" || value == "ultra") return 32768;
    // Qwen Code defaults every subscription model to medium.
    return 16384;
}

std::optional<std::string> gateway_error_message(int status,
                                                 std::string_view body,
                                                 std::int64_t now_unix_seconds) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::object root;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS) return std::nullopt;

    simdjson::dom::object detail;
    const bool has_detail = root["error"].get(detail) == simdjson::SUCCESS;
    if (has_detail && status == 402 && non_blank(detail, "code") == "insufficient_quota") {
        std::string member;
        std::int64_t reset_at = 0;
        if (simdjson::dom::object quota; detail["quota"].get(quota) == simdjson::SUCCESS) {
            member = non_blank(quota, "member_type");
            if (quota["reset_at"].get(reset_at) != simdjson::SUCCESS) {
                double fractional = 0;
                if (quota["reset_at"].get(fractional) == simdjson::SUCCESS
                    && std::isfinite(fractional) && fractional == std::floor(fractional)) {
                    reset_at = static_cast<std::int64_t>(fractional);
                }
            }
        }
        const std::string reset = reset_phrase(reset_at, now_unix_seconds);
        std::string message;
        if (member == "lite" || member == "plus") {
            message = "You've used up your weekly Qwen Subscription credits." + reset
                + " You can wait or run /upgrade to get more credits.";
        } else if (member == "pro") {
            message = "You've used up your weekly Qwen Subscription credits." + reset
                + " Please try again after the reset.";
        } else {
            message = "You've used up your Qwen Subscription credits." + reset
                + " Please try again after the reset.";
        }
        if (const std::string request_id = non_blank(detail, "request_id");
            is_safe_request_id(request_id)) {
            message += " (request_id: " + request_id + ')';
        }
        return message;
    }
    if (has_detail) {
        if (std::string message = non_blank(detail, "message"); !message.empty()) return message;
    }
    if (std::string description = non_blank(root, "error_description"); !description.empty()) {
        return description;
    }
    return std::nullopt;
}

} // namespace core::llm::qwen_subscription
