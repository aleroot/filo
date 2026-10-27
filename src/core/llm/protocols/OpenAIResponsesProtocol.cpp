#include "OpenAIResponsesProtocol.hpp"
#include "OpenAIUsage.hpp"
#include "SseUtils.hpp"
#include "OpenAIProtocol.hpp"
#include "../ModelCatalogProvider.hpp"
#include "../../auth/CodexClientIdentity.hpp"
#include "../Models.hpp"
#include "../ProviderClientIdentity.hpp"
#include "../transport/HttpHeaderUtils.hpp"
#include "../../logging/Logger.hpp"
#include "../../tools/StrictToolSchema.hpp"
#include "../../tools/ToolSchema.hpp"
#include "../StrictToolPolicy.hpp"
#include "../../utils/JsonUtils.hpp"
#include "core/utils/TimeUtils.hpp"
#include "core/version/Version.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <climits>
#include <cstdint>
#include <format>
#include <random>
#include <set>

namespace core::llm::protocols {

using core::llm::ToolSchemaWire;
using core::llm::configured_strict_tool_dialect;

namespace {

[[nodiscard]] std::string extract_assistant_output_text(simdjson::dom::object item);

[[nodiscard]] std::string generate_prompt_cache_key() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    const uint64_t hi = dist(rng);
    const uint64_t lo = dist(rng);
    return std::format("filo-{0:016x}{1:016x}", hi, lo);
}

void append_member_before_object_end(std::string& object_json, std::string_view member_json) {
    if (object_json.empty() || object_json.back() != '}') return;
    object_json.pop_back();
    object_json += ',';
    object_json += member_json;
    object_json += '}';
}

[[nodiscard]] std::string websocket_url_for_responses(std::string_view base_url) {
    std::string url(base_url);
    if (url.starts_with("https://")) {
        url.replace(0, std::string("https://").size(), "wss://");
    } else if (url.starts_with("http://")) {
        url.replace(0, std::string("http://").size(), "ws://");
    }
    return url + "/responses";
}

void append_tool_schema(
    std::string& payload,
    const std::vector<Tool>& tools,
    std::span<const std::string_view> hosted_tool_types = {},
    std::optional<core::tools::schema::StrictDialect> strict_dialect = std::nullopt) {
    payload += R"(,"tools":[)";
    bool first = true;
    for (const std::string_view type : hosted_tool_types) {
        if (!first) payload += ',';
        first = false;
        payload += R"({"type":")";
        payload += core::utils::escape_json_string(type);
        payload += R"("})";
    }
    for (std::size_t i = 0; i < tools.size(); ++i) {
        if (!first) payload += ',';
        first = false;
        const auto& def = tools[i].function;
        payload += R"({"type":"function","name":")";
        payload += core::utils::escape_json_string(def.name);
        payload += R"(","description":")";
        payload += core::utils::escape_json_string(def.description);
        // Strict mode compiles the schema into a decoding grammar, which is the
        // only place an out-of-shape argument can be prevented rather than
        // rejected. A contract that does not survive the projection stays on the
        // permissive path instead of being silently narrowed.
        const std::string canonical_schema =
            core::tools::schema::canonical_input_schema(def);
        const auto strict_schema = strict_dialect.has_value()
            ? core::tools::schema::strict_input_schema(canonical_schema, *strict_dialect)
            : std::nullopt;
        payload += strict_schema.has_value() ? R"(","strict":true,"parameters":)"
                                             : R"(","strict":false,"parameters":)";
        payload += strict_schema.value_or(canonical_schema);
        payload += "}";
    }
    payload += "]";
}

[[nodiscard]] std::string collect_instructions(const ChatRequest& req) {
    if (!req.prompt_plan.empty()) return req.prompt_plan.render();
    std::string instructions;
    bool first = true;
    for (const auto& msg : req.messages) {
        if (msg.role != "system" || msg.content.empty()) continue;
        if (!first) instructions += "\n\n";
        instructions += msg.content;
        first = false;
    }
    return instructions;
}

void append_message_item(std::string& payload,
                         const Message& msg,
                         std::string_view phase = {}) {
    payload += R"({"type":"message","role":")";
    payload += core::utils::escape_json_string(msg.role);
    payload += '"';
    if (msg.role == "assistant" && !phase.empty()) {
        payload += R"(,"phase":")";
        payload += core::utils::escape_json_string(phase);
        payload += '"';
    }
    payload += R"(,"content":[)";

    bool first_part = true;
    const auto append_part_sep = [&]() {
        if (!first_part) payload += ",";
        first_part = false;
    };

    const auto append_text_part = [&](std::string_view text) {
        append_part_sep();
        payload += R"({"type":")";
        payload += (msg.role == "assistant") ? "output_text" : "input_text";
        payload += R"(","text":")";
        payload += core::utils::escape_json_string(text);
        payload += R"("})";
    };

    if (!msg.content_parts.empty()) {
        for (const auto& part : msg.content_parts) {
            if (part.type == ContentPartType::Text) {
                if (!part.text.empty()) {
                    append_text_part(part.text);
                }
                continue;
            }

            if (const auto encoded = encode_image_part(part); encoded.has_value()) {
                append_part_sep();
                payload += R"({"type":"input_image","image_url":")";
                payload += core::utils::escape_json_string(encoded->data_url());
                payload += R"("})";
            } else {
                append_text_part(unavailable_media_attachment_text(part.type, media_reference(part)));
            }
        }
    } else if (!msg.content.empty()) {
        append_text_part(msg.content);
    }

    payload += R"(]})";
}

void append_function_call_item(std::string& payload,
                               const ToolCall& call,
                               std::size_t fallback_idx) {
    const std::string call_id = !call.id.empty()
        ? call.id
        : "call_" + std::to_string(fallback_idx);
    const std::string arguments = call.function.arguments.empty()
        ? "{}"
        : call.function.arguments;

    payload += R"({"type":"function_call","call_id":")";
    payload += core::utils::escape_json_string(call_id);
    payload += R"(","name":")";
    payload += core::utils::escape_json_string(call.function.name);
    payload += R"(","arguments":")";
    payload += core::utils::escape_json_string(arguments);
    payload += R"("})";
}

void append_function_call_output_item(std::string& payload,
                                      const Message& msg,
                                      std::size_t fallback_idx) {
    const std::string call_id = !msg.tool_call_id.empty()
        ? msg.tool_call_id
        : "call_" + std::to_string(fallback_idx);
    payload += R"({"type":"function_call_output","call_id":")";
    payload += core::utils::escape_json_string(call_id);
    payload += R"(","output":")";
    payload += core::utils::escape_json_string(msg.content);
    payload += R"("})";
}

// Native output snapshots preserve message boundaries, phases, and interleaved
// reasoning. Only replay them when they still describe the current history:
// tool normalization or a history edit must not resurrect stale output.
[[nodiscard]] std::optional<std::vector<std::string>> native_response_items(
    const Message& message) {
    std::vector<std::string> items;
    std::string text;
    std::size_t tool_index = 0;
    bool has_snapshot = false;
    for (const auto& continuation : message.continuation_items) {
        if (continuation.provider != "openai") continue;
        if (continuation.kind != "assistant_content" && continuation.kind != "reasoning") continue;
        simdjson::dom::parser parser;
        simdjson::padded_string padded(continuation.payload);
        simdjson::dom::object item;
        if (parser.parse(padded).get(item) != simdjson::SUCCESS) return std::nullopt;
        std::string_view type;
        if (item["type"].get(type) != simdjson::SUCCESS) return std::nullopt;
        if (type == "message") {
            has_snapshot = true;
            text += extract_assistant_output_text(item);
        } else if (type == "function_call") {
            has_snapshot = true;
            if (tool_index >= message.tool_calls.size()) return std::nullopt;
            const auto& call = message.tool_calls[tool_index++];
            std::string_view id, name, arguments;
            if (item["call_id"].get(id) != simdjson::SUCCESS || id != call.id
                || item["name"].get(name) != simdjson::SUCCESS || name != call.function.name
                || item["arguments"].get(arguments) != simdjson::SUCCESS) {
                return std::nullopt;
            }
            // Argument coercion may update a call after generation. Preserve
            // its surrounding message phases while replaying the current call.
            std::string serialized;
            append_function_call_item(serialized, call, tool_index);
            items.push_back(std::move(serialized));
            continue;
        } else if (type != "reasoning") {
            return std::nullopt;
        }
        items.push_back(continuation.payload);
    }
    if (has_snapshot && text == message.content && tool_index == message.tool_calls.size()) return items;
    return std::nullopt;
}

[[nodiscard]] std::vector<std::string> build_input_items(
    const std::vector<Message>& messages,
    bool include_system_messages = false) {
    std::vector<std::string> items;
    std::size_t fallback_idx = 0;

    for (const auto& msg : messages) {
        if (msg.role == "system" && !include_system_messages) continue;

        if (msg.role == "assistant") {
            if (auto native = native_response_items(msg)) {
                items.insert(items.end(), native->begin(), native->end());
                continue;
            }
            for (const auto& continuation : msg.continuation_items) {
                if ((continuation.provider.empty() || continuation.provider == "openai")
                    && continuation.kind != "assistant_content"
                    && has_valid_continuation_payload(continuation)) {
                    items.push_back(continuation.payload);
                }
            }
        }

        if (msg.role == "tool") {
            if (msg.content.empty() && msg.tool_call_id.empty()) continue;
            std::string item;
            append_function_call_output_item(item, msg, ++fallback_idx);
            items.push_back(std::move(item));
            continue;
        }

        if (!msg.content.empty() || !msg.content_parts.empty()) {
            std::string item;
            append_message_item(item, msg);
            items.push_back(std::move(item));
        }

        if (msg.role == "assistant" && !msg.tool_calls.empty()) {
            for (const auto& tc : msg.tool_calls) {
                if (tc.function.name.empty()) continue;
                std::string item;
                append_function_call_item(item, tc, ++fallback_idx);
                items.push_back(std::move(item));
            }
        }
    }

    return items;
}

// The ChatGPT backend rejects `instructions`. The official client sends the
// same stable prompt as a developer message at the front of `input`.
[[nodiscard]] std::vector<std::string> codex_input_items(const ChatRequest& request) {
    auto items = build_input_items(request.messages);
    const std::string instructions = collect_instructions(request);
    if (instructions.empty()) return items;

    Message developer;
    developer.role = "developer";
    developer.content = instructions;
    std::string item;
    append_message_item(item, developer);
    items.insert(items.begin(), std::move(item));
    return items;
}

void append_input_items(std::string& payload, const std::vector<std::string>& input_items) {
    payload += R"(,"input":[)";
    for (std::size_t i = 0; i < input_items.size(); ++i) {
        if (i > 0) payload += ',';
        payload += input_items[i];
    }
    payload += "]";
}

[[nodiscard]] bool has_prefix(const std::vector<std::string>& values,
                              const std::vector<std::string>& prefix) {
    return values.size() >= prefix.size()
        && std::equal(prefix.begin(), prefix.end(), values.begin());
}

[[nodiscard]] std::string serialize_assistant_output_item(simdjson::dom::object item) {
    std::string_view item_type;
    if (item["type"].get(item_type) != simdjson::SUCCESS) return {};

    if (item_type == "function_call") {
        ToolCall call;
        call.type = "function";

        std::string_view call_id;
        if (item["call_id"].get(call_id) == simdjson::SUCCESS) {
            call.id = std::string(call_id);
        } else {
            std::string_view id;
            if (item["id"].get(id) == simdjson::SUCCESS) {
                call.id = std::string(id);
            }
        }

        std::string_view name;
        if (item["name"].get(name) == simdjson::SUCCESS) {
            call.function.name = std::string(name);
        }
        if (call.function.name.empty()) return {};

        std::string_view arguments;
        if (item["arguments"].get(arguments) == simdjson::SUCCESS) {
            call.function.arguments = std::string(arguments);
        }

        std::string serialized;
        append_function_call_item(serialized, call, 0);
        return serialized;
    }

    if (item_type == "message") {
        std::string_view role;
        if (item["role"].get(role) != simdjson::SUCCESS || role != "assistant") {
            return {};
        }

        const std::string text = extract_assistant_output_text(item);
        if (text.empty()) return {};

        Message message;
        message.role = "assistant";
        message.content = text;

        std::string serialized;
        std::string_view phase;
        (void)item["phase"].get(phase);
        append_message_item(serialized, message, phase);
        return serialized;
    }

    if (item_type == "reasoning") return simdjson::to_string(item);

    return {};
}

[[nodiscard]] std::string lower_ascii(std::string_view value) {
    std::string lowered;
    lowered.reserve(value.size());
    for (const char ch : value) {
        lowered.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }
    return lowered;
}

[[nodiscard]] std::optional<int64_t> parse_i64_header(
    const cpr::Header& headers,
    std::string_view name) {
    const auto value = transport::find_header(headers, name);
    if (!value.has_value()) return std::nullopt;
    try {
        std::size_t consumed = 0;
        const int64_t parsed = std::stoll(*value, &consumed);
        if (consumed != value->size()) return std::nullopt;
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] std::optional<double> parse_f64_header(
    const cpr::Header& headers,
    std::string_view name) {
    const auto value = transport::find_header(headers, name);
    if (!value.has_value()) return std::nullopt;
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(*value, &consumed);
        if (consumed != value->size() || !std::isfinite(parsed)) return std::nullopt;
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] std::optional<bool> parse_bool_header(
    const cpr::Header& headers,
    std::string_view name) {
    const auto value = transport::find_header(headers, name);
    if (!value.has_value()) return std::nullopt;
    const std::string lowered = lower_ascii(*value);
    if (lowered == "true" || lowered == "1") return true;
    if (lowered == "false" || lowered == "0") return false;
    return std::nullopt;
}

[[nodiscard]] std::string codex_window_label(
    std::optional<int64_t> window_minutes,
    std::string_view limit_id,
    std::string_view position) {
    std::string duration;
    if (window_minutes.has_value() && *window_minutes > 0) {
        const int64_t minutes = *window_minutes;
        if (minutes % (24 * 60) == 0) {
            duration = std::to_string(minutes / (24 * 60)) + "d";
        } else if (minutes % 60 == 0) {
            duration = std::to_string(minutes / 60) + "h";
        } else {
            duration = std::to_string(minutes) + "m";
        }
    } else {
        duration = std::string(position);
    }

    if (limit_id.empty() || limit_id == "codex") return duration;
    return std::string(limit_id) + " " + duration;
}

[[nodiscard]] std::string normalize_codex_limit_id(std::string_view value) {
    std::string normalized = lower_ascii(value);
    std::ranges::replace(normalized, '-', '_');
    return normalized;
}

void append_codex_header_window(RateLimitInfo& info,
                                const cpr::Header& headers,
                                std::string_view limit_id,
                                std::string_view position) {
    std::string normalized(limit_id.empty() ? "codex" : limit_id);
    std::ranges::replace(normalized, '_', '-');
    const std::string prefix = "x-" + normalized + "-" + std::string(position);
    const auto used_percent = parse_f64_header(headers, prefix + "-used-percent");
    if (!used_percent.has_value()) return;

    const auto window_minutes = parse_i64_header(headers, prefix + "-window-minutes");
    const auto resets_at = parse_i64_header(headers, prefix + "-reset-at");
    if (*used_percent == 0.0
        && (!window_minutes.has_value() || *window_minutes == 0)
        && !resets_at.has_value()) {
        return;
    }

    info.usage_windows.push_back(UsageWindow{
        .label = codex_window_label(window_minutes, limit_id, position),
        .utilization = static_cast<float>(*used_percent / 100.0),
        .resets_at = resets_at.value_or(0),
    });
}

struct CodexRateLimitUpdate {
    std::vector<UsageWindow> usage_windows;
    std::string limit_id = "codex";
    bool updates_windows = false;
    bool replaces_all_windows = false;
    std::optional<std::string> tier_label;
    std::optional<SupplementalCredits> supplemental_credits;
    std::optional<std::string> notice;
    std::optional<std::string> limit_reached_reason;
};

[[nodiscard]] CodexRateLimitUpdate parse_codex_rate_limit_headers(
    const cpr::Header& headers) {
    CodexRateLimitUpdate update;
    std::set<std::string> limit_ids{"codex"};
    constexpr std::string_view suffix = "-primary-used-percent";
    for (const auto& [raw_name, unused] : headers) {
        (void)unused;
        const std::string name = lower_ascii(raw_name);
        if (!name.starts_with("x-") || !name.ends_with(suffix)) continue;
        std::string id = normalize_codex_limit_id(
            name.substr(2, name.size() - 2 - suffix.size()));
        if (!id.empty()) limit_ids.insert(std::move(id));
    }

    for (const auto& limit_id : limit_ids) {
        RateLimitInfo parsed;
        append_codex_header_window(parsed, headers, limit_id, "primary");
        append_codex_header_window(parsed, headers, limit_id, "secondary");
        if (!parsed.usage_windows.empty()) {
            update.updates_windows = true;
            update.replaces_all_windows = true;
            update.usage_windows.insert(
                update.usage_windows.end(),
                std::make_move_iterator(parsed.usage_windows.begin()),
                std::make_move_iterator(parsed.usage_windows.end()));
        }
    }

    const auto has_credits = parse_bool_header(headers, "x-codex-credits-has-credits");
    const auto unlimited = parse_bool_header(headers, "x-codex-credits-unlimited");
    if (has_credits.has_value() && unlimited.has_value()) {
        SupplementalCredits credits{
            .available = *has_credits,
            .unlimited = *unlimited,
        };
        if (const auto balance = transport::find_header(headers, "x-codex-credits-balance");
            balance.has_value()) {
            credits.balance = *balance;
        }
        update.supplemental_credits = std::move(credits);
    }
    if (const auto promo = transport::find_header(headers, "x-codex-promo-message");
        promo.has_value()) {
        update.notice = *promo;
    }
    if (const auto active = transport::find_header(headers, "x-codex-active-limit");
        active.has_value()) {
        update.limit_id = normalize_codex_limit_id(*active);
    }
    if (const auto reached = transport::find_header(
            headers, "x-codex-rate-limit-reached-type"); reached.has_value()) {
        update.limit_reached_reason = *reached;
    }
    return update;
}

[[nodiscard]] std::optional<CodexRateLimitUpdate> parse_codex_rate_limit_event(
    simdjson::dom::element doc) {
    std::string_view event_type;
    if (doc["type"].get_string().get(event_type) != simdjson::SUCCESS
        || event_type != "codex.rate_limits") {
        return std::nullopt;
    }

    CodexRateLimitUpdate update;
    std::string limit_id = "codex";
    std::string_view value;
    if (doc["metered_limit_name"].get_string().get(value) == simdjson::SUCCESS
        && !value.empty()) {
        limit_id = std::string(value);
    } else if (doc["limit_name"].get_string().get(value) == simdjson::SUCCESS
               && !value.empty()) {
        limit_id = std::string(value);
    }
    limit_id = normalize_codex_limit_id(limit_id);

    update.limit_id = limit_id;
    if (doc["plan_type"].get_string().get(value) == simdjson::SUCCESS) {
        update.tier_label = value.empty()
            ? std::string{}
            : "ChatGPT " + std::string(value);
    }

    simdjson::dom::object details;
    if (doc["rate_limits"].get_object().get(details) == simdjson::SUCCESS) {
        update.updates_windows = true;
        const auto parse_window = [&](std::string_view position) {
            simdjson::dom::object window;
            if (details[position].get_object().get(window) != simdjson::SUCCESS) return;
            double used_percent = 0.0;
            if (window["used_percent"].get_double().get(used_percent) != simdjson::SUCCESS
                || !std::isfinite(used_percent)) {
                return;
            }
            int64_t minutes = 0;
            int64_t resets_at = 0;
            const bool has_minutes = window["window_minutes"].get_int64().get(minutes)
                == simdjson::SUCCESS;
            core::utils::json::ignore_error(window["reset_at"].get_int64().get(resets_at));
            update.usage_windows.push_back(UsageWindow{
                .label = codex_window_label(
                    has_minutes ? std::optional<int64_t>{minutes} : std::nullopt,
                    limit_id,
                    position),
                .utilization = static_cast<float>(used_percent / 100.0),
                .resets_at = resets_at,
            });
        };
        parse_window("primary");
        parse_window("secondary");
    }

    simdjson::dom::object credits;
    if (doc["credits"].get_object().get(credits) == simdjson::SUCCESS) {
        SupplementalCredits parsed_credits;
        bool has_credits = false;
        bool unlimited = false;
        if (credits["has_credits"].get_bool().get(has_credits) == simdjson::SUCCESS
            && credits["unlimited"].get_bool().get(unlimited) == simdjson::SUCCESS) {
            parsed_credits.available = has_credits;
            parsed_credits.unlimited = unlimited;
        }
        if (credits["balance"].get_string().get(value) == simdjson::SUCCESS) {
            parsed_credits.balance = std::string(value);
        }
        update.supplemental_credits = std::move(parsed_credits);
    }
    return update;
}

void extract_usage_from_completed(simdjson::dom::element doc, ParseResult& result) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return;

    simdjson::dom::object usage_obj;
    if (response_obj["usage"].get(usage_obj) != simdjson::SUCCESS) return;

    (void)parse_openai_usage(usage_obj, result);
}

[[nodiscard]] std::string parse_response_id(simdjson::dom::element doc) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return {};

    std::string_view id;
    if (response_obj["id"].get(id) == simdjson::SUCCESS && !id.empty()) {
        return std::string(id);
    }
    return {};
}

[[nodiscard]] std::string extract_assistant_output_text(simdjson::dom::object item) {
    simdjson::dom::array content_items;
    if (item["content"].get(content_items) != simdjson::SUCCESS) return {};

    std::string text;
    for (simdjson::dom::element content_el : content_items) {
        simdjson::dom::object content_obj;
        if (content_el.get(content_obj) != simdjson::SUCCESS) continue;

        std::string_view content_type;
        if (content_obj["type"].get(content_type) != simdjson::SUCCESS
            || content_type != "output_text") {
            continue;
        }

        std::string_view text_part;
        if (content_obj["text"].get(text_part) == simdjson::SUCCESS) {
            text.append(text_part);
        }
    }
    return text;
}

[[nodiscard]] std::string parse_failed_message(simdjson::dom::element doc) {
    auto from_error_obj = [](simdjson::dom::object error_obj) -> std::string {
        std::string_view message;
        if (error_obj["message"].get(message) == simdjson::SUCCESS && !message.empty()) {
            return std::string(message);
        }
        return {};
    };

    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) == simdjson::SUCCESS) {
        simdjson::dom::object error_obj;
        if (response_obj["error"].get(error_obj) == simdjson::SUCCESS) {
            if (std::string msg = from_error_obj(error_obj); !msg.empty()) return msg;
        }

        std::string_view top_message;
        if (response_obj["message"].get(top_message) == simdjson::SUCCESS && !top_message.empty()) {
            return std::string(top_message);
        }
    }

    simdjson::dom::object error_obj;
    if (doc["error"].get(error_obj) == simdjson::SUCCESS) {
        if (std::string msg = from_error_obj(error_obj); !msg.empty()) return msg;
    }

    std::string_view top_message;
    if (doc["message"].get(top_message) == simdjson::SUCCESS && !top_message.empty()) {
        return std::string(top_message);
    }

    return "Responses API stream failed.";
}

/// Error class of a failure event, from the same shapes parse_failed_message
/// accepts: `response.error`, a nested `error`, or a flat `error` event whose
/// own `code` carries it. A flat event's `type` names the event, not the error.
[[nodiscard]] std::string_view parse_failed_code(simdjson::dom::element doc) {
    const auto field = [](simdjson::dom::object object, std::string_view key) {
        std::string_view value;
        if (object[key].get(value) != simdjson::SUCCESS) return std::string_view{};
        return value;
    };
    simdjson::dom::object error;
    if (doc["response"]["error"].get(error) == simdjson::SUCCESS
        || doc["error"].get(error) == simdjson::SUCCESS) {
        const auto code = field(error, "code");
        return code.empty() ? field(error, "type") : code;
    }
    simdjson::dom::object event;
    if (doc.get(event) != simdjson::SUCCESS) return {};
    return field(event, "code");
}

[[nodiscard]] std::string parse_incomplete_reason(simdjson::dom::element doc) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return {};

    simdjson::dom::object details_obj;
    if (response_obj["incomplete_details"].get(details_obj) != simdjson::SUCCESS) return {};

    std::string_view reason;
    if (details_obj["reason"].get(reason) == simdjson::SUCCESS && !reason.empty()) {
        return std::string(reason);
    }
    return {};
}

} // namespace

class CodexResponsesProtocol::RateLimitState {
public:
    void apply_http(RateLimitInfo transport_limits,
                    const CodexRateLimitUpdate& update) {
        std::lock_guard lock(mutex_);
        auto windows = std::move(snapshot_.usage_windows);
        auto subscription = std::move(snapshot_.subscription);
        snapshot_ = std::move(transport_limits);
        snapshot_.usage_windows = std::move(windows);
        snapshot_.subscription = std::move(subscription);
        apply_update(update);
    }

    void apply_event(const CodexRateLimitUpdate& update) {
        std::lock_guard lock(mutex_);
        apply_update(update);
    }

    [[nodiscard]] RateLimitInfo snapshot() const {
        std::lock_guard lock(mutex_);
        return snapshot_;
    }

private:
    void apply_update(const CodexRateLimitUpdate& update) {
        if (update.updates_windows) {
            if (update.replaces_all_windows) {
                snapshot_.usage_windows = update.usage_windows;
            } else {
                const std::string prefix = update.limit_id + " ";
                std::erase_if(snapshot_.usage_windows, [&](const UsageWindow& window) {
                    return update.limit_id == "codex"
                        ? window.label.find(' ') == std::string::npos
                        : window.label.starts_with(prefix);
                });
                snapshot_.usage_windows.insert(
                    snapshot_.usage_windows.end(),
                    update.usage_windows.begin(),
                    update.usage_windows.end());
            }
        }
        if (update.tier_label.has_value()) {
            snapshot_.subscription.tier_label = *update.tier_label;
        }
        if (update.supplemental_credits.has_value()) {
            snapshot_.subscription.supplemental_credits =
                *update.supplemental_credits;
        }
        if (update.notice.has_value()) {
            snapshot_.subscription.notice = *update.notice;
        }
        if (update.limit_reached_reason.has_value()) {
            snapshot_.subscription.limit_reached_reason =
                *update.limit_reached_reason;
        }
    }

    mutable std::mutex mutex_;
    RateLimitInfo snapshot_;
};

void OpenAIResponsesProtocol::prepare_request(ChatRequest& req) {
    std::scoped_lock lock(shared_state_->mutex);

    active_session_id_ = req.session_id;
    auto& session = shared_state_->sessions[active_session_id_];
    if (session.prompt_cache_key.empty()) {
        session.prompt_cache_key = req.session_id.empty()
            ? generate_prompt_cache_key()
            : req.session_id;
    }
    if (req.prompt_cache_key.empty()) {
        req.prompt_cache_key = session.prompt_cache_key;
    }
    if (conversation_context_strategy()
        == ConversationContextStrategy::ReplayInput) {
        req.previous_response_id.clear();
    } else if (req.previous_response_id.empty()
               && !session.previous_response_id.empty()) {
        req.previous_response_id = session.previous_response_id;
    }
}

void OpenAIResponsesProtocol::reset_state() {
    std::scoped_lock lock(shared_state_->mutex);
    shared_state_->sessions.clear();
    active_session_id_.clear();
    OpenAIResponsesProtocol::reset_attempt_state();
}

void OpenAIResponsesProtocol::reset_attempt_state() {
    in_progress_response_id_.clear();
    last_response_id_.clear();
    saw_text_delta_ = false;
    stream_retry_vetoed_ = false;
    last_rate_limit_ = {};
}

std::string OpenAIResponsesProtocol::serialize_with_input_items(
    const ChatRequest& req,
    const std::vector<std::string>& input_items,
    std::optional<std::string_view> previous_response_id_override) const {
    const SerializationOptions options;
    return serialize_with_input_items(
        req, input_items, previous_response_id_override, options);
}

std::string OpenAIResponsesProtocol::serialize_with_input_items(
    const ChatRequest& req,
    const std::vector<std::string>& input_items,
    std::optional<std::string_view> previous_response_id_override,
    const SerializationOptions& options) const {
    std::string payload;
    std::size_t input_bytes = 0;
    for (const auto& item : input_items) input_bytes += item.size();
    payload.reserve(estimated_request_bytes(req, /*include_messages=*/false) + input_bytes);

    payload += R"({"model":")";
    payload += core::utils::escape_json_string(req.model);
    payload += '"';
    const bool replay_input = conversation_context_strategy()
        == ConversationContextStrategy::ReplayInput;
    if (!replay_input || !replay_system_messages_in_input()) {
        payload += R"(,"instructions":")";
        payload += core::utils::escape_json_string(collect_instructions(req));
        payload += '"';
    }
    payload += R"(,"stream":)";
    payload += req.stream ? "true" : "false";
    if (options.include_store) {
        payload += R"(,"store":false)";
    }

    if (!replay_input || previous_response_id_override.has_value()) {
        const std::string_view previous_response_id = previous_response_id_override.has_value()
            ? *previous_response_id_override
            : std::string_view{req.previous_response_id};
        if (!previous_response_id.empty()) {
            payload += R"(,"previous_response_id":")";
            payload += core::utils::escape_json_string(previous_response_id);
            payload += '"';
        }
    }

    if (options.include_sampling_controls && req.temperature.has_value()) {
        payload += R"(,"temperature":)";
        payload += std::to_string(*req.temperature);
    }
    if (options.include_sampling_controls && req.max_tokens.has_value()) {
        payload += R"(,"max_output_tokens":)";
        payload += std::to_string(*req.max_tokens);
    }
    const auto* catalog = uses_provider_reasoning_catalog() && req.catalog_reasoning
        ? &*req.catalog_reasoning : nullptr;
    const auto reasoning = catalog && catalog->complete
        ? catalog->effort : reasoning_capabilities(req.model);
    std::string_view requested_effort = req.effort;
    if (requested_effort.empty() && catalog && catalog->catalog_default_effort) {
        requested_effort = *catalog->catalog_default_effort;
    }
    if (options.include_reasoning_summary
        || options.reasoning_effort_override.has_value()
        || reasoning.supports_effort()) {
        // Public OpenAI/Grok/DashScope send effort only. Codex opts into
        // `summary: auto` via SerializationOptions because the ChatGPT
        // backend 400s when that field is missing.
        std::string effort = options.reasoning_effort_override.has_value()
            ? std::string(*options.reasoning_effort_override)
            : reasoning.supports_effort()
                ? normalize_openai_reasoning_effort(requested_effort, reasoning)
                : std::string{};
        if (!options.reasoning_effort_override && catalog && !catalog->supported_efforts.empty()) {
            const auto& levels = catalog->supported_efforts;
            // The Codex catalog lists `ultra`, but it is a client-side
            // selection the API rejects; normalization resolved it above.
            if (requested_effort != "ultra"
                && std::ranges::find(levels, requested_effort) != levels.end()) {
                effort = requested_effort;
            } else if (!effort.empty() && std::ranges::find(levels, effort) == levels.end()) {
                // An authoritative catalog may expose only a subset of the
                // usual low/medium/high ladder. Its default is always safer
                // than sending a level the deployment does not accept.
                effort = catalog->catalog_default_effort.value_or("");
                if (std::ranges::find(levels, effort) == levels.end()) effort.clear();
            }
        }
        if (!effort.empty() || options.include_reasoning_summary) {
            payload += R"(,"reasoning":{)";
            bool first = true;
            if (!effort.empty()) {
                payload += R"("effort":")";
                payload += core::utils::escape_json_string(effort);
                payload += '"';
                first = false;
            }
            if (options.include_reasoning_summary) {
                if (!first) payload += ',';
                payload += R"("summary":"auto")";
            }
            payload += '}';
        }
    }

    append_input_items(payload, input_items);

    const bool has_tools = !req.tools.empty() || !options.hosted_tool_types.empty();
    payload += R"(,"tool_choice":"auto","parallel_tool_calls":)";
    payload += has_tools ? "true" : "false";

    if (has_tools) {
        append_tool_schema(
            payload,
            req.tools,
            options.hosted_tool_types,
            configured_strict_tool_dialect(ToolSchemaWire::OpenAI, req.model));
    }

    if (options.include_prompt_cache_key && !req.prompt_cache_key.empty()) {
        payload += R"(,"prompt_cache_key":")";
        payload += core::utils::escape_json_string(req.prompt_cache_key);
        payload += '"';
    }

    const std::string_view service_tier = req.service_tier.empty()
        ? std::string_view{default_service_tier_}
        : std::string_view{req.service_tier};
    if (!service_tier.empty()) {
        payload += R"(,"service_tier":")";
        payload += core::utils::escape_json_string(service_tier);
        payload += '"';
    }

    if (options.include_response_include) {
        if (include_reasoning_encrypted_) {
            payload += R"(,"include":["reasoning.encrypted_content"])";
        } else {
            payload += R"(,"include":[])";
        }
    }

    // Codex sends the catalog's default_verbosity for every model that
    // advertises support_verbosity, alongside any structured-output format.
    const bool has_verbosity = options.include_catalog_verbosity
        && req.catalog_verbosity.has_value()
        && !req.catalog_verbosity->empty();
    const bool has_format = req.response_format.type == ResponseFormat::Type::JsonSchema
        || req.response_format.type == ResponseFormat::Type::JsonObject;
    if (has_verbosity || has_format) {
        payload += R"(,"text":{)";
        if (has_verbosity) {
            payload += R"("verbosity":")";
            payload += core::utils::escape_json_string(*req.catalog_verbosity);
            payload += '"';
            if (has_format) payload += ',';
        }
        if (req.response_format.type == ResponseFormat::Type::JsonSchema) {
            payload += R"("format":{"type":"json_schema","name":"structured_output","strict":true,"schema":)";
            payload += req.response_format.schema.empty() ? R"({"type":"object"})" : req.response_format.schema;
            payload += '}';
        } else if (req.response_format.type == ResponseFormat::Type::JsonObject) {
            payload += R"("format":{"type":"json_schema","name":"json_object","strict":false,"schema":{"type":"object","additionalProperties":true}})";
        }
        payload += '}';
    }

    payload += "}";
    return payload;
}

std::string OpenAIResponsesProtocol::serialize(const ChatRequest& req) const {
    const SerializationOptions options;
    return serialize_with_options(req, options);
}

std::string OpenAIResponsesProtocol::serialize_with_options(
    const ChatRequest& req,
    const SerializationOptions& options) const {
    return serialize_with_input_items(
        req,
        build_input_items(
            req.messages,
            conversation_context_strategy() == ConversationContextStrategy::ReplayInput
                && replay_system_messages_in_input()),
        std::nullopt,
        options);
}

cpr::Header OpenAIResponsesProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept",       "text/event-stream"},
    };
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }

    return headers;
}

ReasoningCapabilities OpenAIResponsesProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    return openai_reasoning_capabilities(model);
}

std::string OpenAIResponsesProtocol::build_url(std::string_view base_url,
                                               [[maybe_unused]] std::string_view model) const {
    return std::string(base_url) + "/responses";
}

CodexResponsesProtocol::CodexResponsesProtocol(bool include_reasoning_encrypted,
                                               std::string default_service_tier,
                                               std::shared_ptr<IProviderClientIdentitySource>
                                                   client_identity_source)
    : OpenAIResponsesProtocol(include_reasoning_encrypted, std::move(default_service_tier))
    , client_identity_source_(std::move(client_identity_source))
    , rate_limit_state_(std::make_shared<RateLimitState>()) {}

cpr::Header CodexResponsesProtocol::build_headers(
    const core::auth::AuthInfo& auth) const {
    cpr::Header headers = OpenAIResponsesProtocol::build_headers(auth);
    if (const auto account = auth.properties.find("account_id");
        account != auth.properties.end() && !account->second.empty()
        && !headers.contains("chatgpt-account-id")) {
        headers["chatgpt-account-id"] = account->second;
    }
    return headers;
}

std::string CodexResponsesProtocol::serialize(const ChatRequest& request) const {
    return serialize_codex_with_input_items(request, codex_input_items(request));
}

std::unique_ptr<core::llm::ModelCatalogProvider>
CodexResponsesProtocol::model_catalog(std::string_view provider_name) const {
    return std::make_unique<CodexModelCatalogProvider>(std::string(provider_name));
}

std::string CodexResponsesProtocol::serialize_codex_with_input_items(
    const ChatRequest& request,
    const std::vector<std::string>& input_items,
    std::optional<std::string_view> previous_response_id_override) const {
    SerializationOptions options;
    options.include_reasoning_summary = true;
    options.include_catalog_verbosity = true;
    options.include_sampling_controls = false;
    std::string payload = serialize_with_input_items(
        request, input_items, previous_response_id_override, options);
    const std::string thread_id = request.session_id.empty()
        ? std::string("filo")
        : request.session_id;
    const std::string window_id = thread_id + ":0";
    std::string metadata = R"("client_metadata":{"x-codex-installation-id":")";
    metadata += core::utils::escape_json_string(installation_id());
    metadata += R"(","session_id":")";
    metadata += core::utils::escape_json_string(thread_id);
    metadata += R"(","thread_id":")";
    metadata += core::utils::escape_json_string(thread_id);
    metadata += R"(","x-codex-window-id":")";
    metadata += core::utils::escape_json_string(window_id);
    metadata += R"("})";
    append_member_before_object_end(payload, metadata);
    return payload;
}

void CodexResponsesProtocol::prepare_headers(cpr::Header& headers,
                                             const ChatRequest& request,
                                             [[maybe_unused]] std::string_view base_url) {
    const std::string thread_id = request.session_id.empty()
        ? std::string("filo")
        : request.session_id;

    std::string turn_state;
    {
        std::lock_guard lock(transport_state_->mutex);
        if (!request.transport_turn_id.empty()) {
            if (auto it = transport_state_->turn_states.find(request.transport_turn_id);
                it != transport_state_->turn_states.end()) {
                turn_state = it->second;
            }
        }
    }

    headers["x-client-request-id"] = thread_id;
    headers["x-codex-installation-id"] = installation_id();
    headers["x-codex-window-id"] = thread_id + ":0";
    headers["x-responsesapi-include-timing-metrics"] = "true";
    headers["originator"] = std::string(core::auth::codex_cli::kOriginator);
    headers["version"] = std::string(core::auth::codex_cli::kClientVersion);
    // Current Codex sends the hyphenated names. Filo has always sent the
    // underscore form as well, so it stays.
    headers["session_id"] = thread_id;
    headers["session-id"] = thread_id;
    headers["thread-id"] = thread_id;
    headers["User-Agent"] = core::auth::codex_cli::user_agent();
    if (!turn_state.empty()) {
        headers["x-codex-turn-state"] = turn_state;
    } else {
        headers.erase("x-codex-turn-state");
    }
}

std::string CodexResponsesProtocol::build_websocket_url(
    std::string_view base_url,
    [[maybe_unused]] std::string_view model) const {
    return websocket_url_for_responses(base_url);
}

void CodexResponsesProtocol::prepare_websocket_headers(cpr::Header& headers,
                                                       const ChatRequest& request,
                                                       std::string_view base_url) {
    headers.erase("Content-Type");
    headers.erase("Accept");
    prepare_headers(headers, request, base_url);
    headers["OpenAI-Beta"] = "responses_websockets=2026-02-06";
}

std::string CodexResponsesProtocol::serialize_websocket_request(
    const ChatRequest& request) const {
    return serialize_websocket_request_impl(request, /*prewarm=*/false);
}

std::string CodexResponsesProtocol::serialize_websocket_prewarm_request(
    const ChatRequest& request) const {
    if (request.transport_turn_id.empty()) {
        return {};
    }
    {
        std::lock_guard lock(transport_state_->mutex);
        if (transport_state_->last_ws_requests.contains(request.transport_turn_id)) {
            return {};
        }
    }
    return serialize_websocket_request_impl(request, /*prewarm=*/true);
}

WebSocketRequestFrame CodexResponsesProtocol::initial_websocket_request_frame(
    const ChatRequest& request) const {
    if (const std::string prewarm = serialize_websocket_prewarm_request(request);
        !prewarm.empty()) {
        return WebSocketRequestFrame{
            .payload = prewarm,
            .suppress_output = true,
        };
    }

    return WebSocketRequestFrame{
        .payload = serialize_websocket_request(request),
    };
}

WebSocketRequestFrame CodexResponsesProtocol::next_websocket_request_frame(
    const ChatRequest& request,
    const WebSocketRequestFrame& completed_frame) const {
    if (!completed_frame.suppress_output) {
        return {};
    }

    return WebSocketRequestFrame{
        .payload = serialize_websocket_request(request),
    };
}

std::string CodexResponsesProtocol::serialize_websocket_request_impl(
    const ChatRequest& request,
    bool prewarm) const {
    const auto full_input_items = codex_input_items(request);

    // Compare request controls and stable instructions without copying the
    // complete conversation just to discard it for the signature.
    std::vector<std::string> signature_input;
    if (const auto instructions = collect_instructions(request); !instructions.empty()) {
        Message developer{.role = "developer", .content = instructions};
        std::string item;
        append_message_item(item, developer);
        signature_input.push_back(std::move(item));
    }
    const std::string request_signature = serialize_codex_with_input_items(
        request,
        signature_input,
        std::string_view{});

    std::string incremental_previous_response_id;
    std::vector<std::string> incremental_input_items;
    if (!request.transport_turn_id.empty()) {
        std::lock_guard lock(transport_state_->mutex);
        if (auto last = transport_state_->last_ws_requests.find(request.transport_turn_id);
            last != transport_state_->last_ws_requests.end()
            && last->second.request_signature == request_signature
            && !last->second.response_id.empty()) {
            const auto& previous = last->second;
            const auto baseline_size = previous.request_input_items.size()
                + previous.response_items_added.size();
            if (full_input_items.size() >= baseline_size
                && has_prefix(full_input_items, previous.request_input_items)
                && std::equal(previous.response_items_added.begin(), previous.response_items_added.end(),
                    full_input_items.begin() + static_cast<std::ptrdiff_t>(previous.request_input_items.size()))
                && (full_input_items.size() > baseline_size
                    || last->second.from_prewarm)) {
                incremental_input_items = std::vector<std::string>{
                    full_input_items.begin() + static_cast<std::ptrdiff_t>(baseline_size),
                    full_input_items.end()};
                incremental_previous_response_id = last->second.response_id;
            }
        }

        transport_state_->pending_ws_requests[request.transport_turn_id] =
            TransportState::PendingWebSocketRequest{
                .request_signature = request_signature,
                .request_input_items = full_input_items,
                .prewarm = prewarm,
            };
    }

    const bool use_incremental = !incremental_previous_response_id.empty();
    std::string payload = use_incremental
        ? serialize_codex_with_input_items(
            request, incremental_input_items, std::string_view{incremental_previous_response_id})
        : serialize_codex_with_input_items(request, full_input_items);

    if (prewarm) {
        append_member_before_object_end(payload, R"("generate":false)");
    }

    active_transport_turn_id_ = request.transport_turn_id;
    active_response_items_.clear();

    if (payload.empty() || payload.front() != '{') {
        return {};
    }
    payload.erase(payload.begin());
    return std::string{R"({"type":"response.create",)"} + payload;
}

void CodexResponsesProtocol::abandon_websocket_request(const ChatRequest& request) {
    if (request.transport_turn_id.empty()) {
        return;
    }

    std::lock_guard lock(transport_state_->mutex);
    transport_state_->pending_ws_requests.erase(request.transport_turn_id);
    transport_state_->last_ws_requests.erase(request.transport_turn_id);
    if (active_transport_turn_id_ == request.transport_turn_id) {
        active_transport_turn_id_.clear();
        active_response_items_.clear();
    }
}

ParseResult CodexResponsesProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    std::string data_scratch;
    std::string event_type;
    std::string serialized_output_item;
    std::optional<CodexRateLimitUpdate> rate_limit_update;

    if (sse::parse_event_payload(raw_event, parsed, data_scratch)) {
        event_type = std::string(parsed.event);
        if (!parsed.is_done) {
            thread_local simdjson::dom::parser parser;
            const core::utils::json::ParserRetentionGuard parser_guard{parser};
            simdjson::padded_string padded(parsed.data);
            simdjson::dom::element doc;
            if (parser.parse(padded).get(doc) == simdjson::SUCCESS) {
                if (event_type.empty()) {
                    std::string_view type_from_payload;
                    if (doc["type"].get(type_from_payload) == simdjson::SUCCESS) {
                        event_type = std::string(type_from_payload);
                    }
                }

                if (event_type == "response.output_item.done") {
                    simdjson::dom::object item;
                    if (doc["item"].get(item) == simdjson::SUCCESS) {
                        serialized_output_item = serialize_assistant_output_item(item);
                    }
                } else if (event_type == "codex.rate_limits") {
                    rate_limit_update = parse_codex_rate_limit_event(doc);
                }
            }
        }
    }

    ParseResult result = OpenAIResponsesProtocol::parse_event(raw_event);

    if (rate_limit_update.has_value()) {
        rate_limit_state_->apply_event(*rate_limit_update);
        result.rate_limit = last_rate_limit();
    }

    if (!serialized_output_item.empty()) {
        active_response_items_.push_back(std::move(serialized_output_item));
    }

    if (result.done
        && event_type == "response.completed"
        && !active_transport_turn_id_.empty()
        && !last_response_id().empty()) {
        std::lock_guard lock(transport_state_->mutex);
        if (auto pending = transport_state_->pending_ws_requests.find(active_transport_turn_id_);
            pending != transport_state_->pending_ws_requests.end()) {
            transport_state_->last_ws_requests[active_transport_turn_id_] =
                TransportState::LastWebSocketRequest{
                    .response_id = last_response_id(),
                    .request_signature = std::move(pending->second.request_signature),
                    .request_input_items = std::move(pending->second.request_input_items),
                    .response_items_added = std::move(active_response_items_),
                    .from_prewarm = pending->second.prewarm,
                };
            transport_state_->pending_ws_requests.erase(pending);
            // The transport owns one socket. Old turns cannot continue on it;
            // retaining 32 complete histories here needlessly multiplies RAM.
            std::erase_if(transport_state_->last_ws_requests, [&](const auto& entry) {
                return entry.first != active_transport_turn_id_;
            });
        }
        active_transport_turn_id_.clear();
        active_response_items_.clear();
    }

    if (result.done && !active_transport_turn_id_.empty()) {
        std::lock_guard lock(transport_state_->mutex);
        transport_state_->pending_ws_requests.erase(active_transport_turn_id_);
        active_transport_turn_id_.clear();
        active_response_items_.clear();
    }

    return result;
}

std::string CodexResponsesProtocol::websocket_connection_key(
    std::string_view url,
    const cpr::Header& headers,
    const ChatRequest& request) const {
    std::string key(url);
    key += "\nsession=";
    key += request.session_id;
    key += "\nturn=";
    key += request.transport_turn_id;

    for (std::string_view name : {
             std::string_view{"Authorization"},
             std::string_view{"chatgpt-account-id"},
             std::string_view{"OpenAI-Beta"}}) {
        if (auto value = transport::find_header(headers, name);
            value.has_value() && !value->empty()) {
            key += '\n';
            key += name;
            key += ':';
            key += *value;
        }
    }
    return key;
}

void CodexResponsesProtocol::observe_response_headers(const cpr::Header& headers,
                                                      const ChatRequest& request) {
    OpenAIResponsesProtocol::observe_response_headers(headers, request);
    if (request.transport_turn_id.empty()) {
        return;
    }

    auto turn_state = transport::find_header(headers, "x-codex-turn-state");
    if (!turn_state.has_value() || turn_state->empty()) {
        return;
    }

    std::lock_guard lock(transport_state_->mutex);
    transport_state_->turn_states[request.transport_turn_id] = *turn_state;
    while (transport_state_->turn_states.size() > 32) {
        transport_state_->turn_states.erase(transport_state_->turn_states.begin());
    }
}

void CodexResponsesProtocol::reset_state() {
    OpenAIResponsesProtocol::reset_state();
    std::lock_guard lock(transport_state_->mutex);
    transport_state_->turn_states.clear();
    transport_state_->last_ws_requests.clear();
    transport_state_->pending_ws_requests.clear();
    active_response_items_.clear();
    active_transport_turn_id_.clear();
}

std::unique_ptr<ApiProtocolBase> CodexResponsesProtocol::clone() const {
    auto cloned = std::make_unique<CodexResponsesProtocol>(
        include_reasoning_encrypted_,
        default_service_tier_,
        client_identity_source_);
    share_continuity_state_with(*cloned);
    cloned->transport_state_ = transport_state_;
    cloned->rate_limit_state_ = rate_limit_state_;
    return cloned;
}

std::string CodexResponsesProtocol::installation_id() const {
    std::lock_guard lock(transport_state_->mutex);
    if (transport_state_->installation_id.empty()) {
        transport_state_->installation_id = client_identity_source_
            ? client_identity_source_->installation_id()
            : std::string("filo");
    }
    return transport_state_->installation_id;
}

ParseResult OpenAIResponsesProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};

    const std::string_view payload_sv = parsed.data;
    if (parsed.is_done) {
        // [DONE] belongs to Chat Completions. It cannot prove that a
        // Responses generation completed (or supply its status/usage).
        return {};
    }

    thread_local simdjson::dom::parser parser;
    const core::utils::json::ParserRetentionGuard parser_guard{parser};
    simdjson::padded_string padded(payload_sv);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        core::logging::debug("[responses] Failed to parse SSE payload JSON");
        return {};
    }

    std::string_view event_type = parsed.event;
    if (event_type.empty()) {
        std::string_view type_from_payload;
        if (doc["type"].get(type_from_payload) == simdjson::SUCCESS) {
            event_type = type_from_payload;
        }
    }

    ParseResult result;

    if (event_type == "response.created") {
        in_progress_response_id_ = parse_response_id(doc);
        result.stream_started = true;
        return result;
    }

    if (event_type == "response.output_text.delta") {
        std::string_view delta;
        if (doc["delta"].get(delta) == simdjson::SUCCESS && !delta.empty()) {
            saw_text_delta_ = true;
            result.chunks.push_back(StreamChunk::make_content(std::string(delta)));
        }
        return result;
    }

    if (event_type == "response.output_text.done") {
        // This event is a completed text snapshot, not another delta. Some
        // compatible providers (including Qwen) send it after the full delta
        // sequence, so forwarding it would duplicate the assistant response.
        if (saw_text_delta_) {
            return result;
        }
        std::string_view text;
        if (doc["text"].get(text) == simdjson::SUCCESS && !text.empty()) {
            saw_text_delta_ = true;
            result.chunks.push_back(StreamChunk::make_content(std::string(text)));
        }
        return result;
    }

    if (event_type == "response.reasoning_text.delta"
        || event_type == "response.reasoning_summary_text.delta") {
        std::string_view delta;
        if (doc["delta"].get(delta) == simdjson::SUCCESS && !delta.empty()) {
            result.chunks.push_back(StreamChunk::make_reasoning(std::string(delta)));
        }
        return result;
    }

    if (event_type == "response.output_item.done" || event_type == "response.output_item.added") {
        simdjson::dom::object item;
        if (doc["item"].get(item) != simdjson::SUCCESS) return result;

        std::string_view item_type;
        if (item["type"].get(item_type) != simdjson::SUCCESS) return result;

        if (event_type == "response.output_item.added") {
            if (item_type == "message") saw_text_delta_ = false;
            return result;
        }

        if (item_type == "function_call") {
            // output_item.added is a provisional item snapshot, not an
            // arguments delta. Qwen commonly reports arguments="{}" there and
            // supplies the completed JSON in output_item.done. Emitting both
            // makes the agent's delta accumulator concatenate two standalone
            // JSON documents (for example, {}{"path":"..."}).
            if (event_type != "response.output_item.done") {
                return result;
            }

            ToolCall call;
            call.type = "function";

            int64_t index = -1;
            if (item["index"].get(index) == simdjson::SUCCESS && index >= 0 && index <= INT32_MAX) {
                call.index = static_cast<int>(index);
            }

            std::string_view call_id;
            if (item["call_id"].get(call_id) == simdjson::SUCCESS) {
                call.id = std::string(call_id);
            } else {
                std::string_view id;
                if (item["id"].get(id) == simdjson::SUCCESS) {
                    call.id = std::string(id);
                }
            }

            std::string_view name;
            if (item["name"].get(name) == simdjson::SUCCESS) {
                call.function.name = std::string(name);
            }

            std::string_view arguments;
            if (item["arguments"].get(arguments) == simdjson::SUCCESS) {
                call.function.arguments = std::string(arguments);
            }
            if (call.function.arguments.empty()) {
                call.function.arguments = "{}";
            }

            if (!call.function.name.empty()) {
                StreamChunk chunk;
                chunk.tools.push_back(std::move(call));
                result.chunks.push_back(std::move(chunk));
            }
        } else if (item_type == "reasoning" && event_type == "response.output_item.done") {
            StreamChunk chunk;
            chunk.continuation_items.push_back(ContinuationItem{
                .provider = "openai",
                .kind = "reasoning",
                .payload = simdjson::to_string(item),
            });
            result.chunks.push_back(std::move(chunk));
        } else if (item_type == "message") {
            std::string_view role;
            if (item["role"].get(role) != simdjson::SUCCESS || role != "assistant") {
                return result;
            }

            std::string text = extract_assistant_output_text(item);
            if (!text.empty() && !saw_text_delta_) {
                result.chunks.push_back(StreamChunk::make_content(std::move(text)));
            }
        }

        if (preserves_output_messages() && (item_type == "message" || item_type == "function_call")) {
            if (const auto snapshot = serialize_assistant_output_item(item); !snapshot.empty()) {
                if (result.chunks.empty()) result.chunks.emplace_back();
                result.chunks.back().continuation_items.push_back({
                    .provider = "openai", .kind = "assistant_content", .payload = snapshot});
            }
        }
        if (item_type == "message") saw_text_delta_ = false;

        return result;
    }

    if (event_type == "response.completed") {
        extract_usage_from_completed(doc, result);
        std::string response_id = parse_response_id(doc);
        if (response_id.empty() && !in_progress_response_id_.empty()) {
            response_id = in_progress_response_id_;
        }
        if (!response_id.empty()) {
            last_response_id_ = std::move(response_id);
        }
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    if (event_type == "response.incomplete") {
        std::string reason = parse_incomplete_reason(doc);
        if (reason.empty()) reason = "unknown";
        result.chunks.push_back(StreamChunk::make_error(
            "\n[Responses API incomplete response: " + reason + "]"));
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    if (event_type == "response.failed" || event_type == "error") {
        const std::string message = parse_failed_message(doc);
        const std::string_view code = parse_failed_code(doc);
        result.stream_error = true;
        result.stream_error_type = std::string(code);
        result.stream_error_message = message;
        result.retryable_stream_error = !stream_retry_vetoed_
            && (code == "server_error" || code == "internal_server_error"
                || code == "server_is_overloaded" || code == "rate_limit_exceeded"
                || code == "rate_limit_error" || code == "timeout"
                || code == "request_timeout" || code == "temporarily_unavailable"
                || code == "previous_response_not_found"
                || code == "websocket_connection_limit_reached");
        cpr::Header advice;
        for (const auto key : {"headers", "response_headers"}) {
            simdjson::dom::object headers;
            if (doc[key].get(headers) != simdjson::SUCCESS) continue;
            for (const auto field : headers) {
                std::string_view value;
                if (field.value.get(value) == simdjson::SUCCESS) advice[std::string(field.key)] = std::string(value);
            }
        }
        result.retryable_stream_error = result.retryable_stream_error && !transport::retry_vetoed(advice);
        result.rate_limit.retry_after = transport::retry_after_seconds(advice);
        result.chunks.push_back(StreamChunk::make_error("\n[Responses API error: " + message + "]"));
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    return result;
}

void OpenAIResponsesProtocol::on_response(const HttpResponse& response) {
    RateLimitInfo info;

    const auto parse_int = [&](std::string_view key) -> int32_t {
        if (const auto value = transport::find_header(response.headers, key);
            value.has_value()) {
            try {
                return static_cast<int32_t>(std::stoi(*value));
            } catch (...) {}
        }
        return 0;
    };

    info.requests_limit     = parse_int("x-ratelimit-limit-requests");
    info.requests_remaining = parse_int("x-ratelimit-remaining-requests");
    if (const auto value = transport::find_header(
            response.headers, "x-ratelimit-reset-requests"); value.has_value()) {
        info.requests_reset = core::utils::time::parse_timestamp_or_duration(*value);
    }
    info.tokens_limit       = parse_int("x-ratelimit-limit-tokens");
    info.tokens_remaining   = parse_int("x-ratelimit-remaining-tokens");
    if (const auto value = transport::find_header(
            response.headers, "x-ratelimit-reset-tokens"); value.has_value()) {
        info.tokens_reset = core::utils::time::parse_timestamp_or_duration(*value);
    }
    info.retry_after        = transport::retry_after_seconds(response.headers);
    info.is_rate_limited    = (response.status_code == 429 || info.retry_after > 0);

    if (info.requests_limit > 0 && !info.is_rate_limited
        && !transport::find_header(response.headers, "x-ratelimit-remaining-requests")) {
        info.requests_remaining = info.requests_limit;
    }
    if (info.tokens_limit > 0 && !info.is_rate_limited
        && !transport::find_header(response.headers, "x-ratelimit-remaining-tokens")) {
        info.tokens_remaining = info.tokens_limit;
    }

    last_rate_limit_ = std::move(info);

    if (conversation_context_strategy()
            == ConversationContextStrategy::StatefulPreviousResponse
        && response.status_code == 200
        && !last_response_id_.empty()) {
        std::scoped_lock lock(shared_state_->mutex);
        shared_state_->sessions[active_session_id_].previous_response_id = last_response_id_;
    }
}

void CodexResponsesProtocol::on_response(const HttpResponse& response) {
    OpenAIResponsesProtocol::on_response(response);
    rate_limit_state_->apply_http(
        OpenAIResponsesProtocol::last_rate_limit(),
        parse_codex_rate_limit_headers(response.headers));
}

RateLimitInfo CodexResponsesProtocol::last_rate_limit() const noexcept {
    try {
        return rate_limit_state_->snapshot();
    } catch (...) {
        return {};
    }
}

bool OpenAIResponsesProtocol::is_retryable(
    const HttpResponse& response) const noexcept {
    if (stream_retry_vetoed_ || transport::retry_vetoed(response.headers)) return false;
    if (response.status_code == 429) {
        try {
            simdjson::dom::parser parser;
            simdjson::padded_string body(response.body);
            simdjson::dom::element doc;
            std::string_view code;
            if (parser.parse(body).get(doc) == simdjson::SUCCESS
                && doc["error"]["code"].get(code) == simdjson::SUCCESS
                && (code == "insufficient_quota" || code == "usage_limit_reached"
                    || code == "billing_hard_limit_reached")) return false;
        } catch (...) {}
    }
    return is_openai_retryable_status(response.status_code);
}

void OpenAIResponsesProtocol::observe_response_headers(
    const cpr::Header& headers, const ChatRequest&) {
    stream_retry_vetoed_ = parse_bool_header(headers, "x-should-retry") == false;
}

} // namespace core::llm::protocols
