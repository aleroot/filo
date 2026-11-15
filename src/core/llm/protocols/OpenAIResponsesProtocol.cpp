#include "OpenAIResponsesProtocol.hpp"
#include "OpenAIUsage.hpp"
#include "SseUtils.hpp"
#include "OpenAIProtocol.hpp"
#include "../CodexUsageClient.hpp"
#include "../ModelCatalogProvider.hpp"
#include "../../auth/CodexClientIdentity.hpp"
#include "../Models.hpp"
#include "../ProviderClientIdentity.hpp"
#include "../transport/HttpHeaderUtils.hpp"
#include "../../logging/Logger.hpp"
#include "../../tools/StrictToolSchema.hpp"
#include "../../tools/ToolSchema.hpp"
#include "../../tools/ToolNames.hpp"
#include "../StrictToolPolicy.hpp"
#include "../../utils/JsonUtils.hpp"
#include "../../utils/JsonWriter.hpp"
#include "../../utils/AsciiUtils.hpp"
#include "core/utils/TimeUtils.hpp"
#include "core/version/Version.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <climits>
#include <cstdint>
#include <ctime>
#include <format>
#include <limits>
#include <random>
#include <set>
#include <tuple>

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

void append_tool_array(
    std::string& payload,
    const std::vector<Tool>& tools,
    std::span<const std::string_view> hosted_tool_types = {},
    std::optional<core::tools::schema::StrictDialect> strict_dialect = std::nullopt) {
    payload += '[';
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

void append_tool_schema(
    std::string& payload,
    const std::vector<Tool>& tools,
    std::span<const std::string_view> hosted_tool_types = {},
    std::optional<core::tools::schema::StrictDialect> strict_dialect = std::nullopt) {
    payload += R"(,"tools":)";
    append_tool_array(payload, tools, hosted_tool_types, strict_dialect);
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
                         std::string_view phase = {},
                         std::string_view image_detail = {}) {
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
                if (!image_detail.empty()) {
                    payload += R"(","detail":")";
                    payload += core::utils::escape_json_string(image_detail);
                }
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
                               std::size_t fallback_idx,
                               bool namespaced = false) {
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
    if (namespaced) payload += R"(","namespace":"functions)";
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
    const Message& message, bool namespaced) {
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
            append_function_call_item(serialized, call, tool_index, namespaced);
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
    bool include_system_messages = false,
    bool namespaced = false,
    std::string_view image_detail = {},
    bool include_reasoning_effort_updates = false) {
    std::vector<std::string> items;
    std::size_t fallback_idx = 0;

    // A durable effort change is a provider control item: it is replayed
    // only by a wire that accepts one, and only where the change happened.
    const auto is_replayable_control =
        [](const ContinuationItem& continuation, bool effort_updates) {
            if (continuation.provider != "openai") return false;
            if (continuation.kind == kCheckpointContinuationKind) return true;
            return effort_updates
                && continuation.kind == kReasoningEffortContinuationKind;
        };

    for (const auto& msg : messages) {
        if (msg.role == "system" && !include_system_messages) continue;

        if (msg.role == "assistant") {
            if (auto native = native_response_items(msg, namespaced)) {
                items.insert(items.end(), native->begin(), native->end());
                continue;
            }
            for (const auto& continuation : msg.continuation_items) {
                if ((continuation.provider.empty() || continuation.provider == "openai")
                    && continuation.kind != "assistant_content"
                    && has_valid_continuation_payload(continuation)
                    && (continuation.kind != kReasoningEffortContinuationKind
                        || include_reasoning_effort_updates)) {
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
            append_message_item(item, msg, {}, image_detail);
            items.push_back(std::move(item));
        }

        // A provider checkpoint follows the user message it rides on, the
        // position codex-rs gives its compaction item after the retained user
        // messages (compact_pending_input.rs insert_summary). A durable
        // effort change rides on the message where the user selected it.
        if (msg.role != "assistant") {
            for (const auto& continuation : msg.continuation_items) {
                if (is_replayable_control(continuation, include_reasoning_effort_updates)
                    && has_valid_continuation_payload(continuation)) {
                    items.push_back(continuation.payload);
                }
            }
        }

        if (msg.role == "assistant" && !msg.tool_calls.empty()) {
            for (const auto& tc : msg.tool_calls) {
                if (tc.function.name.empty()) continue;
                std::string item;
                append_function_call_item(item, tc, ++fallback_idx, namespaced);
                items.push_back(std::move(item));
            }
        }
    }

    return items;
}

[[nodiscard]] bool responses_lite(const ChatRequest& request) {
    return request.catalog_wire_protocol == "responses_lite";
}

/// The WebSocket continuation a request extends, empty for none. An agent's
/// own step loop continues its conversation across turns on one socket, the
/// way codex-rs keeps one websocket session (connection, last request, last
/// response) for the whole thread and lets each new turn send only the new
/// items with previous_response_id (core/src/client.rs: ModelClient::
/// new_session takes the cached session, ModelClientSession's Drop stores it
/// back). Requests sent beside the conversation (keep-warm replays, side
/// queries, compaction) clear continuation_scope and stay scoped to their
/// transport turn, so they never consume the conversation's continuation.
[[nodiscard]] std::string websocket_continuation(const ChatRequest& request) {
    if (request.transport_turn_id.empty()) return {};
    if (!request.continuation_scope.empty()) return "conversation:" + request.continuation_scope;
    return "turn:" + request.transport_turn_id;
}

/// The conversation a request belongs to: its thread, its prompt cache and
/// its session state. A delegated worker owns its own conversation, so it
/// never continues the one that spawned it; every other request shares the
/// session's.
[[nodiscard]] std::string conversation_of(const ChatRequest& request) {
    return request.conversation_scope.empty()
        ? request.session_id
        : request.conversation_scope;
}

// ── `ultra` ────────────────────────────────────────────────────────────────
// `ultra` is a client-side selection the API rejects ("Invalid value:
// 'ultra'"). With a catalog that describes it, Filo follows Codex: drive the
// model at the catalog's ultra level and, for a model tuned for it, switch to
// proactive delegation. Without one, normalize_openai_reasoning_effort's
// max → xhigh → high ladder applies.

[[nodiscard]] std::string_view requested_reasoning_effort(const ChatRequest& request) {
    if (!request.effort.empty()) return request.effort;
    if (request.catalog_reasoning && request.catalog_reasoning->catalog_default_effort) {
        return *request.catalog_reasoning->catalog_default_effort;
    }
    return {};
}

/// codex-rs ModelInfo::resolve_reasoning_effort: the catalog's ultra level
/// when offered, else `max`, else the strongest offered non-ultra level.
[[nodiscard]] std::optional<std::string> catalog_ultra_effort(
    const ModelReasoningProfile& catalog) {
    const auto& levels = catalog.supported_efforts;
    const auto offered = [&](std::string_view level) {
        return level != "ultra" && std::ranges::find(levels, level) != levels.end();
    };
    if (catalog.ultra && catalog.ultra->effort && offered(*catalog.ultra->effort)) {
        return *catalog.ultra->effort;
    }
    if (offered("max")) return std::string("max");
    for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
        if (*it != "ultra") return *it;
    }
    return std::nullopt;
}

/// Filo's wording of Codex's proactive multi-agent mode message, naming the
/// delegation tools this request actually carries. Empty when ultra is not
/// selected, the model is not tuned for it, or no delegation tool is offered.
[[nodiscard]] std::string proactive_delegation_instructions(const ChatRequest& request) {
    if (requested_reasoning_effort(request) != "ultra") return {};
    const auto& catalog = request.catalog_reasoning;
    if (!catalog || !catalog->ultra || !catalog->ultra->proactive_delegation) return {};

    std::string tools;
    for (const auto& tool : request.tools) {
        if (!core::tools::names::is_subagent_tool(tool.function.name)) continue;
        if (!tools.empty()) tools += " or ";
        tools += '`';
        tools += tool.function.name;
        tools += '`';
    }
    if (tools.empty()) return {};

    return std::format(
        "\n\n[Delegation Mode]\n"
        "Proactive delegation is active (reasoning effort: ultra). Any earlier instruction "
        "to delegate only when the user explicitly asks no longer applies. User requests "
        "override this.\n"
        "Whenever handing a self-contained piece of work to another agent could save time "
        "or improve quality, do so with {} — whether you are the primary agent or a "
        "subagent. Issue independent delegations together so they run in parallel.\n",
        tools);
}

[[nodiscard]] std::string lite_tools_item(const ChatRequest& request) {
    if (request.tools.empty()) return {};
    std::string item = R"({"type":"additional_tools","role":"developer","tools":[{"type":"namespace","name":"functions","description":"","tools":)";
    append_tool_array(item, request.tools, {},
        configured_strict_tool_dialect(ToolSchemaWire::OpenAI, request.model));
    item += "}]}";
    return item;
}

// codex-rs core/src/client.rs build_responses_request: consecutive effort
// updates collapse to the latest one, because only the last selection is in
// effect. The prefix test is exact: the item is produced by this protocol.
[[nodiscard]] bool is_effort_update_item(std::string_view item) {
    return item.starts_with(R"({"type":"configuration_update")");
}

void collapse_adjacent_effort_updates(std::vector<std::string>& items) {
    std::vector<std::string> collapsed;
    collapsed.reserve(items.size());
    for (auto& item : items) {
        if (is_effort_update_item(item)
            && !collapsed.empty()
            && is_effort_update_item(collapsed.back())) {
            collapsed.back() = std::move(item);
            continue;
        }
        collapsed.push_back(std::move(item));
    }
    items = std::move(collapsed);
}

// The ChatGPT backend rejects `instructions`. The official client sends the
// same stable prompt as a developer message at the front of `input`.
[[nodiscard]] std::vector<std::string> codex_input_items(const ChatRequest& request) {
    // codex-rs protocol/src/models.rs DEFAULT_IMAGE_DETAIL is `high`;
    // client_common.rs normalize_image_detail clears it for Responses Lite.
    auto items = build_input_items(request.messages, false, responses_lite(request),
                                   responses_lite(request) ? std::string_view{} : "high",
                                   request.catalog_supports_reasoning_effort_updates);
    const std::string instructions = collect_instructions(request);
    if (!instructions.empty()) {
        Message developer;
        developer.role = "developer";
        developer.content = instructions;
        std::string item;
        append_message_item(item, developer);
        items.insert(items.begin(), std::move(item));
    }
    if (responses_lite(request)) {
        if (auto tools = lite_tools_item(request); !tools.empty())
            items.insert(items.begin(), std::move(tools));
    }
    // codex-rs compact_remote_v2_attempt.rs: a compaction request is the
    // conversation followed by a `compaction_trigger` control item.
    if (request.native_compaction) items.emplace_back(R"({"type":"compaction_trigger"})");
    collapse_adjacent_effort_updates(items);
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

/// Whether the endpoint's copy of an item and the conversation's copy are the
/// same item. The conversation names the namespace its tools live in, while
/// the endpoint's echo of the call does not, so the two copies of a tool call
/// differ by that one member. codex-rs compares the items themselves and
/// ignores the metadata it adds (responses_request_properties_mismatch and
/// response_items_equal_ignoring_internal_metadata).
[[nodiscard]] bool same_input_item(std::string_view left, std::string_view right) {
    if (left == right) return true;

    simdjson::dom::parser left_parser;
    simdjson::dom::parser right_parser;
    simdjson::dom::element left_doc;
    simdjson::dom::element right_doc;
    const simdjson::padded_string left_padded{std::string(left)};
    const simdjson::padded_string right_padded{std::string(right)};
    if (left_parser.parse(left_padded).get(left_doc) != simdjson::SUCCESS) return false;
    if (right_parser.parse(right_padded).get(right_doc) != simdjson::SUCCESS) return false;
    if (!left_doc.is_object() || !right_doc.is_object()) return false;

    std::string_view left_type;
    std::string_view right_type;
    if (left_doc["type"].get(left_type) != simdjson::SUCCESS) return false;
    if (right_doc["type"].get(right_type) != simdjson::SUCCESS) return false;
    if (left_type != "function_call" || right_type != "function_call") return false;

    const auto same_field = [](const simdjson::dom::element& first,
                                const simdjson::dom::element& second,
                                const char* key) {
        std::string_view first_value;
        std::string_view second_value;
        if (first[key].get(first_value) != simdjson::SUCCESS) return false;
        if (second[key].get(second_value) != simdjson::SUCCESS) return false;
        return first_value == second_value;
    };
    return same_field(left_doc, right_doc, "call_id")
        && same_field(left_doc, right_doc, "name")
        && same_field(left_doc, right_doc, "arguments");
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
        std::string_view ns;
        append_function_call_item(serialized, call, 0,
            item["namespace"].get(ns) == simdjson::SUCCESS && ns == "functions");
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
        std::ignore = item["phase"].get(phase);
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

/// The response an interrupt acknowledgement names. The answer carries it at
/// the top level, not inside a response object.
[[nodiscard]] std::string parse_acknowledged_response_id(simdjson::dom::element doc) {
    std::string_view id;
    if (doc["response_id"].get(id) == simdjson::SUCCESS && !id.empty()) {
        return std::string(id);
    }
    return {};
}

/// The routing token a metadata event names (codex-rs
/// header_turn_state_value_from_json). The event carries the response headers
/// as a JSON object, and the value may be a string or a list of them.
[[nodiscard]] std::string parse_metadata_turn_state(simdjson::dom::element doc) {
    simdjson::dom::object headers;
    if (doc["headers"].get(headers) != simdjson::SUCCESS) return {};

    for (auto header : headers) {
        if (!core::utils::ascii::iequals(header.key, "x-codex-turn-state")) continue;
        std::string_view value;
        if (header.value.get(value) == simdjson::SUCCESS && !value.empty()) {
            return std::string(value);
        }
        simdjson::dom::array values;
        if (header.value.get(values) == simdjson::SUCCESS) {
            for (auto first : values) {
                if (first.get(value) == simdjson::SUCCESS && !value.empty()) {
                    return std::string(value);
                }
            }
        }
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

// ── ChatGPT plan limits ─────────────────────────────────────────────────────
// The ChatGPT backend names a plan limit in `error.type`, not `error.code`
// (codex-rs codex-api/src/api_bridge.rs, UsageErrorBody). It arrives as an
// HTTP 429 body or, over the WebSocket, as an `error` event wrapping the same
// object. The wording is codex-rs protocol/src/error.rs: UsageLimitReachedError
// and UsageNotIncluded.

/// Plan or billing exhaustion: a retry cannot succeed until the user acts.
/// codex-rs maps these to UsageLimitReached, UsageNotIncluded or QuotaExceeded.
[[nodiscard]] bool is_quota_error(std::string_view value) noexcept {
    return value == "usage_limit_reached" || value == "usage_not_included"
        || value == "insufficient_quota" || value == "credit_balance_exhausted"
        || value == "organization_spend_limit_exceeded"
        || value == "project_spend_limit_exceeded"
        || value == "organization_usage_limit_exceeded"
        || value == "billing_hard_limit_reached";
}

[[nodiscard]] std::string_view trim_ascii(std::string_view value) noexcept {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.remove_suffix(1);
    return value;
}

/// codex-rs format_retry_timestamp: local clock time today, else a dated time.
[[nodiscard]] std::string codex_reset_time(int64_t resets_at, std::time_t now) {
    const auto reset = static_cast<std::time_t>(resets_at);
    std::tm reset_tm{};
    std::tm now_tm{};
    if (localtime_r(&reset, &reset_tm) == nullptr || localtime_r(&now, &now_tm) == nullptr) return {};
    char clock[16];
    if (std::strftime(clock, sizeof clock, "%I:%M %p", &reset_tm) == 0) return {};
    std::string_view time(clock);
    if (time.starts_with('0')) time.remove_prefix(1);
    if (reset_tm.tm_year == now_tm.tm_year && reset_tm.tm_yday == now_tm.tm_yday) {
        return std::string(time);
    }
    char month[8];
    if (std::strftime(month, sizeof month, "%b", &reset_tm) == 0) return {};
    const int day = reset_tm.tm_mday;
    const std::string_view suffix = day >= 11 && day <= 13 ? "th"
        : day % 10 == 1 ? "st" : day % 10 == 2 ? "nd" : day % 10 == 3 ? "rd" : "th";
    return std::format("{} {}{}, {} {}", month, day, suffix, reset_tm.tm_year + 1900, time);
}

/// codex-rs retry_suffix (`after_or == false`) and retry_suffix_after_or.
[[nodiscard]] std::string codex_retry_suffix(int64_t resets_at, bool after_or) {
    const std::string at = resets_at > 0 ? codex_reset_time(resets_at, std::time(nullptr)) : std::string{};
    if (after_or) return at.empty() ? " or try again later." : " or try again at " + at + ".";
    return at.empty() ? " Try again later." : " Try again at " + at + ".";
}

/// The user-facing text of a ChatGPT plan-limit error, or nullopt when the
/// error is not one. `headers` are the response (or wrapped event) headers.
[[nodiscard]] std::optional<std::string> codex_usage_error_message(
    simdjson::dom::element doc, const cpr::Header& headers) {
    simdjson::dom::object error;
    if (doc["error"].get(error) != simdjson::SUCCESS) return std::nullopt;
    std::string_view type;
    if (error["type"].get(type) != simdjson::SUCCESS) return std::nullopt;
    if (type == "usage_not_included") {
        return std::string(
            "To use Codex with your ChatGPT plan, upgrade to Plus: https://chatgpt.com/explore/plus.");
    }
    if (type != "usage_limit_reached") return std::nullopt;

    int64_t resets_at = 0;
    core::utils::json::ignore_error(error["resets_at"].get_int64().get(resets_at));
    std::string_view plan;
    core::utils::json::ignore_error(error["plan_type"].get_string().get(plan));

    // The limit the server reports as active, named by x-<limit>-limit-name.
    std::string limit = "codex";
    if (const auto active = transport::find_header(headers, "x-codex-active-limit")) {
        if (const auto trimmed = trim_ascii(*active); !trimmed.empty()) limit = lower_ascii(trimmed);
    }
    std::ranges::replace(limit, '_', '-');
    if (const auto name = transport::find_header(headers, "x-" + limit + "-limit-name")) {
        const auto trimmed = trim_ascii(*name);
        const std::string lowered = lower_ascii(trimmed);
        if (!trimmed.empty() && lowered != "codex" && lowered != "gpt-reserve") {
            return std::format("You’ve hit your usage limit for {}. Switch to another model now,{}",
                               trimmed, codex_retry_suffix(resets_at, true));
        }
    }

    if (const auto reached = transport::find_header(headers, "x-codex-rate-limit-reached-type")) {
        if (*reached == "workspace_owner_credits_depleted")
            return std::string("Your workspace is out of credits. Add credits to continue.");
        if (*reached == "workspace_member_credits_depleted")
            return std::string("Your workspace is out of credits. Ask your workspace owner to refill in order to continue.");
        if (*reached == "workspace_owner_usage_limit_reached")
            return std::string("You hit your spend cap set in your workspace. Increase your spend cap to continue.");
        if (*reached == "workspace_member_usage_limit_reached")
            return std::string("You hit your spend cap set by the owner of your workspace. Ask an owner to increase your spend cap to continue.");
    }

    if (const auto promo = transport::find_header(headers, "x-codex-promo-message")) {
        return std::format("You’ve hit your usage limit. {},{}", *promo, codex_retry_suffix(resets_at, true));
    }

    // PlanType deserializes the lowercase serde names (aliases hc, education).
    const auto in = [&](std::initializer_list<std::string_view> plans) {
        return std::ranges::find(plans, plan) != plans.end();
    };
    if (in({"plus"})) {
        return "You’ve hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), "
               "visit https://chatgpt.com/settings/usage to purchase more credits"
            + codex_retry_suffix(resets_at, true);
    }
    if (in({"team", "self_serve_business_prolite", "self_serve_business_usage_based", "business",
            "ent26", "enterprise_cbp_automation", "enterprise_cbp_usage_based", "sci", "sci_plus"})) {
        return "You’ve hit your usage limit. To get more access now, send a request to your admin"
            + codex_retry_suffix(resets_at, true);
    }
    if (in({"free", "go"})) {
        return "You’ve hit your usage limit. Upgrade to Plus to continue using Codex "
               "(https://chatgpt.com/explore/plus),"
            + codex_retry_suffix(resets_at, true);
    }
    if (in({"pro", "prolite", "promax"})) {
        return "You’ve hit your usage limit. Visit https://chatgpt.com/settings/usage to purchase more credits"
            + codex_retry_suffix(resets_at, true);
    }
    return "You’ve hit your usage limit." + codex_retry_suffix(resets_at, false);
}

/// codex-rs try_parse_retry_delay: a rate-limit message states when the
/// client may come back ("Please try again in 11.054s."). The delay is a
/// lower bound, so a fractional value rounds up.
[[nodiscard]] std::optional<int32_t> codex_retry_delay_from_message(
    std::string_view message) noexcept {
    constexpr std::string_view kNeedle = "try again in";
    const std::string lowered = lower_ascii(message);
    const std::size_t needle_at = lowered.find(kNeedle);
    if (needle_at == std::string::npos) return std::nullopt;

    std::size_t pos = needle_at + kNeedle.size();
    const auto skip_spaces = [&] {
        while (pos < message.size()
               && std::isspace(static_cast<unsigned char>(message[pos]))) {
            ++pos;
        }
    };
    skip_spaces();

    const std::size_t digits_begin = pos;
    while (pos < message.size()
           && std::isdigit(static_cast<unsigned char>(message[pos]))) {
        ++pos;
    }
    if (pos == digits_begin) return std::nullopt;

    std::size_t number_end = pos;
    if (pos < message.size() && message[pos] == '.') {
        const std::size_t fraction_begin = pos + 1;
        std::size_t fraction_end = fraction_begin;
        while (fraction_end < message.size()
               && std::isdigit(static_cast<unsigned char>(message[fraction_end]))) {
            ++fraction_end;
        }
        if (fraction_end > fraction_begin) number_end = fraction_end;
    }

    double seconds = 0;
    try {
        seconds = std::stod(std::string(message.substr(digits_begin, number_end - digits_begin)));
    } catch (...) {
        return std::nullopt;
    }
    pos = number_end;
    skip_spaces();

    const std::string unit = lower_ascii(message.substr(
        pos, std::min<std::size_t>(6, message.size() - pos)));
    if (unit.starts_with("ms")) {
        seconds /= 1000.0;
    } else if (unit.starts_with("s") || unit.starts_with("second")) {
        // Seconds already.
    } else {
        return std::nullopt;
    }
    if (!std::isfinite(seconds) || seconds <= 0) return std::nullopt;
    return static_cast<int32_t>(std::min(
        std::ceil(seconds),
        static_cast<double>(std::numeric_limits<int32_t>::max())));
}

/// codex-rs parse_terminal_provider_error and the quota, usage and policy
/// classes of codex-api/src/sse/responses_error.rs: the failures a retry
/// cannot fix.
[[nodiscard]] bool codex_terminal_failure_class(std::string_view code) noexcept {
    if (code == "usage_limit_reached" || code == "usage_not_included"
        || code == "insufficient_quota" || code == "credit_balance_exhausted"
        || code == "organization_spend_limit_exceeded"
        || code == "project_spend_limit_exceeded"
        || code == "cyber_policy" || code == "bio_policy"
        || code == "misalignment_policy_violation" || code == "invalid_prompt"
        || code == "context_length_exceeded" || code == "content_policy_violation"
        || code == "budget_reached" || code == "model_not_found"
        || code == "inference_hook_denied" || code == "access_program_not_enabled"
        || code == "invalid_access_program" || code == "unsupported_access_program") {
        return true;
    }
    // A request the backend rejected cannot be fixed by asking again. The
    // ChatGPT backend answers these with an HTTP status before the stream
    // starts, so upstream never classifies them; a stream that carries one
    // anyway is terminal too.
    return code == "invalid_request_error" || code == "content_filter"
        || code == "invalid_api_key" || code == "authentication_error"
        || code == "permission_denied" || code == "not_found_error";
}

/// codex-rs subagent_header_value: a delegated session names itself, and the
/// name travels in a header and in the request's client metadata. Only a
/// conservative token is allowed through: upstream drops a label that is not
/// a valid header value, and so does Filo.
[[nodiscard]] std::optional<std::string> codex_subagent_header_value(
    std::string_view label) noexcept {
    constexpr std::size_t kMaxLength = 64;
    if (label.empty() || label.size() > kMaxLength) return std::nullopt;
    std::string sanitized;
    sanitized.reserve(label.size());
    for (const char ch : label) {
        const auto byte = static_cast<unsigned char>(ch);
        const bool safe = (byte >= 'a' && byte <= 'z')
            || (byte >= 'A' && byte <= 'Z')
            || (byte >= '0' && byte <= '9')
            || ch == '-' || ch == '_' || ch == '.';
        if (!safe) return std::nullopt;
        sanitized.push_back(ch);
    }
    return sanitized;
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

    /// The account's `/wham/usage` reading, kept apart so a stream that
    /// reports fewer windows cannot erase it.
    void apply_account(RateLimitInfo account) {
        std::lock_guard lock(mutex_);
        account_ = std::move(account);
    }

    /// The streamed state, completed by the account reading: a streamed
    /// window wins over the read one of the same label.
    [[nodiscard]] RateLimitInfo snapshot() const {
        std::lock_guard lock(mutex_);
        RateLimitInfo merged = snapshot_;
        if (!account_) return merged;
        for (const auto& window : account_->usage_windows) {
            if (std::ranges::none_of(merged.usage_windows, [&](const UsageWindow& streamed) {
                    return streamed.label == window.label;
                })) {
                merged.usage_windows.push_back(window);
            }
        }
        merged.is_rate_limited |= account_->is_rate_limited;
        auto& subscription = merged.subscription;
        const auto& read = account_->subscription;
        if (subscription.tier_label.empty()) subscription.tier_label = read.tier_label;
        if (!subscription.supplemental_credits) {
            subscription.supplemental_credits = read.supplemental_credits;
        }
        if (subscription.limit_reached_reason.empty()) {
            subscription.limit_reached_reason = read.limit_reached_reason;
        }
        return merged;
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
    std::optional<RateLimitInfo> account_;
};

void OpenAIResponsesProtocol::prepare_request(ChatRequest& req) {
    std::scoped_lock lock(shared_state_->mutex);

    active_conversation_ = conversation_of(req);
    auto& session = shared_state_->sessions[active_conversation_];
    if (session.prompt_cache_key.empty()) {
        session.prompt_cache_key = active_conversation_.empty()
            ? generate_prompt_cache_key()
            : active_conversation_;
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

    // A Dynamic tail layer keeps the stable prefix cacheable when the user
    // toggles ultra mid-session.
    if (auto delegation = proactive_delegation_instructions(req); !delegation.empty()) {
        if (!req.prompt_plan.empty()) {
            req.prompt_plan.append(core::context::ContextLayer{
                .kind = core::context::ContextLayerKind::RuntimeInstructions,
                .stability = core::context::PromptStability::Dynamic,
                .name = "delegation_mode",
                .content = std::move(delegation),
            });
        } else {
            Message system;
            system.role = "system";
            system.content = std::move(delegation);
            req.messages.push_back(std::move(system));
        }
    }
}

void OpenAIResponsesProtocol::reset_state() {
    std::scoped_lock lock(shared_state_->mutex);
    shared_state_->sessions.clear();
    active_conversation_.clear();
    OpenAIResponsesProtocol::reset_attempt_state();
}

void OpenAIResponsesProtocol::reset_attempt_state() {
    in_progress_response_id_.clear();
    last_response_id_.clear();
    saw_text_delta_ = false;
    stream_retry_vetoed_ = false;
    last_rate_limit_ = {};
}

OpenAIResponsesProtocol::ResolvedReasoning
OpenAIResponsesProtocol::resolve_request_reasoning(
    const ChatRequest& request,
    std::optional<std::string_view> effort_override) const {
    const auto* catalog = uses_provider_reasoning_catalog() && request.catalog_reasoning
        ? &*request.catalog_reasoning : nullptr;
    const auto capabilities = catalog && catalog->complete
        ? catalog->effort
        : reasoning_capabilities(request.model);
    std::string_view requested_effort = request.effort;
    if (requested_effort.empty() && catalog && catalog->catalog_default_effort) {
        requested_effort = *catalog->catalog_default_effort;
    }

    std::string effort = effort_override.has_value()
        ? std::string(*effort_override)
        : capabilities.supports_effort()
            ? normalize_openai_reasoning_effort(requested_effort, capabilities)
            : std::string{};
    if (!effort_override && catalog && !catalog->supported_efforts.empty()) {
        const auto& levels = catalog->supported_efforts;
        if (requested_effort == "ultra") {
            if (auto ultra = catalog_ultra_effort(*catalog)) effort = std::move(*ultra);
        } else if (std::ranges::find(levels, requested_effort) != levels.end()) {
            effort = requested_effort;
        }
        if (!effort.empty() && std::ranges::find(levels, effort) == levels.end()) {
            // An authoritative catalog may expose only a subset of the usual
            // low/medium/high ladder. Its default is always safer than sending
            // a level the deployment does not accept.
            effort = catalog->catalog_default_effort.value_or("");
            if (std::ranges::find(levels, effort) == levels.end()) effort.clear();
        }
    }
    return {.capabilities = capabilities, .effort = std::move(effort)};
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
    // Routing fields precede the potentially multi-megabyte input: gateways may
    // inspect a body incrementally (codex-rs codex-api/src/common.rs
    // ResponsesApiRequest field order).
    if (const std::string service_tier = request_service_tier(req); !service_tier.empty()) {
        payload += R"(,"service_tier":")";
        payload += core::utils::escape_json_string(service_tier);
        payload += '"';
    }
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
    const auto resolved = resolve_request_reasoning(req, options.reasoning_effort_override);
    const auto& reasoning = resolved.capabilities;
    if (options.include_reasoning_summary
        || options.reasoning_effort_override.has_value()
        || options.tools_in_input
        || reasoning.supports_effort()) {
        // Public OpenAI/Grok/DashScope send effort only. Codex defaults to
        // `summary: auto` unless its catalog specifies otherwise.
        const std::string& effort = resolved.effort;
        if (!effort.empty() || options.include_reasoning_summary || options.tools_in_input) {
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
                payload += R"("summary":")";
                payload += core::utils::escape_json_string(options.reasoning_summary);
                payload += '"';
                first = false;
            }
            if (options.tools_in_input) {
                if (!first) payload += ',';
                payload += R"("context":"all_turns")";
            }
            payload += '}';
        }
    }

    append_input_items(payload, input_items);

    const bool has_tools = !req.tools.empty() || !options.hosted_tool_types.empty();
    payload += R"(,"tool_choice":"auto","parallel_tool_calls":)";
    payload += has_tools && !options.tools_in_input ? "true" : "false";

    if (has_tools && !options.tools_in_input) {
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
                                                   client_identity_source,
                                               std::shared_ptr<codex::IUsageSource> usage)
    : OpenAIResponsesProtocol(include_reasoning_encrypted, std::move(default_service_tier))
    , client_identity_source_(std::move(client_identity_source))
    , usage_(usage ? std::move(usage) : codex::make_usage_source())
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

std::vector<ContinuationItem> CodexResponsesProtocol::conversation_control_items(
    const ChatRequest& request) const {
    // codex-rs ModelInfo::supports_reasoning_effort_updates gates the whole
    // feature: a model that does not accept the item keeps its effort on the
    // request parameter alone.
    if (!request.catalog_supports_reasoning_effort_updates) return {};
    const auto resolved = resolve_request_reasoning(request);
    if (resolved.effort.empty()) return {};

    std::string payload = R"({"type":"configuration_update","reasoning":{"effort":")";
    payload += core::utils::escape_json_string(resolved.effort);
    payload += R"("}})";
    return {ContinuationItem{
        .provider = "openai",
        .kind = std::string(kReasoningEffortContinuationKind),
        .payload = std::move(payload),
    }};
}

std::optional<std::string> CodexResponsesProtocol::interrupt_request(
    std::string_view response_id) const {
    // codex-rs codex-api/src/endpoint/responses_websocket.rs: the frame that
    // asks the endpoint to stop the response it is serving.
    if (response_id.empty()) return std::nullopt;
    std::string frame = R"({"type":"response.interrupt","response_id":")";
    frame += core::utils::escape_json_string(response_id);
    frame += R"(","mode":"discard_partial_items"})";
    return frame;
}

std::string CodexResponsesProtocol::codex_request_body(
    const ChatRequest& request,
    const std::vector<std::string>& input_items,
    std::optional<std::string_view> previous_response_id_override) const {
    SerializationOptions options;
    const auto* reasoning = request.catalog_reasoning ? &*request.catalog_reasoning : nullptr;
    options.include_reasoning_summary = !reasoning || reasoning->summary_supported.value_or(true);
    if (reasoning && reasoning->default_summary) {
        const auto& summary = *reasoning->default_summary;
        if (summary == "none") options.include_reasoning_summary = false;
        else if (summary == "auto" || summary == "concise" || summary == "detailed")
            options.reasoning_summary = summary;
    }
    options.include_catalog_verbosity = true;
    options.include_sampling_controls = false;
    options.tools_in_input = responses_lite(request);
    return serialize_with_input_items(
        request, input_items, previous_response_id_override, options);
}

std::string CodexResponsesProtocol::client_metadata_member(
    const ChatRequest& request,
    bool websocket) const {
    const std::string thread_id = conversation_of(request).empty()
        ? std::string("filo")
        : conversation_of(request);
    const std::string window_id = thread_id + ":0";
    const std::string session_id = request.session_id.empty()
        ? thread_id
        : request.session_id;
    // codex-rs CodexResponsesMetadata::client_metadata: the request names the
    // conversation it belongs to, the worker that sent it, the thread that
    // spawned it and the turn it serves. A JSON writer keeps the object well
    // formed whatever the set of fields is.
    core::utils::JsonWriter metadata_writer(256);
    metadata_writer.key("client_metadata");
    {
        auto _metadata = metadata_writer.object();
        metadata_writer.kv_str("x-codex-installation-id", installation_id());
        metadata_writer.comma().kv_str("session_id", session_id);
        metadata_writer.comma().kv_str("thread_id", thread_id);
        metadata_writer.comma().kv_str("x-codex-window-id", window_id);
        // A delegated session names itself so the backend can attribute the
        // request.
        if (const auto label = codex_subagent_header_value(request.subagent_label);
            label.has_value()) {
            metadata_writer.comma().kv_str("x-openai-subagent", *label);
        }
        // The thread that spawned this one, and the turn this request is.
        if (!request.parent_thread_id.empty()) {
            metadata_writer.comma().kv_str(
                "x-codex-parent-thread-id", request.parent_thread_id);
        }
        if (!request.transport_turn_id.empty()) {
            metadata_writer.comma().kv_str("turn_id", request.transport_turn_id);
        }
        if (websocket && responses_lite(request)) {
            metadata_writer.comma().kv_str(
                "ws_request_header_x_openai_internal_codex_responses_lite", "true");
        }
    }
    return std::move(metadata_writer).take();
}

std::string CodexResponsesProtocol::serialize_codex_with_input_items(
    const ChatRequest& request,
    const std::vector<std::string>& input_items,
    std::optional<std::string_view> previous_response_id_override,
    bool websocket) const {
    std::string payload = codex_request_body(
        request, input_items, previous_response_id_override);
    append_member_before_object_end(payload, client_metadata_member(request, websocket));
    return payload;
}

std::string CodexResponsesProtocol::request_service_tier(const ChatRequest& request) const {
    std::string configured = request.service_tier.empty()
        ? default_service_tier_ : request.service_tier;
    if (configured == "flex") return configured;
    if (!request.catalog_service_tiers.has_value()) {
        // No catalog evidence: send what the user configured, except the
        // explicit `default`, which is never a request value.
        return configured == "default" ? std::string{} : configured;
    }
    const auto& offered = *request.catalog_service_tiers;
    if (configured.empty()) configured = request.catalog_default_service_tier;
    if (configured.empty() || configured == "default"
        || std::ranges::find(offered, configured) == offered.end()) {
        return {};
    }
    return configured;
}

void CodexResponsesProtocol::prepare_headers(cpr::Header& headers,
                                             const ChatRequest& request,
                                             [[maybe_unused]] std::string_view base_url) {
    // The thread is the conversation the request belongs to: a delegated
    // worker owns its own thread, as a Codex subagent session does.
    const std::string thread_id = conversation_of(request).empty()
        ? std::string("filo")
        : conversation_of(request);
    const std::string session_id = request.session_id.empty()
        ? thread_id
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
    // codex-rs compatibility_headers: a delegated session names itself, so
    // the backend can attribute the request to the worker that sent it.
    if (const auto label = codex_subagent_header_value(request.subagent_label);
        label.has_value()) {
        headers["x-openai-subagent"] = *label;
    } else {
        headers.erase("x-openai-subagent");
    }
    // The thread that spawned this one.
    if (!request.parent_thread_id.empty()) {
        headers["x-codex-parent-thread-id"] = request.parent_thread_id;
    } else {
        headers.erase("x-codex-parent-thread-id");
    }
    headers["x-responsesapi-include-timing-metrics"] = "true";
    headers["originator"] = std::string(core::auth::codex_cli::kOriginator);
    headers["version"] = std::string(core::auth::codex_cli::kClientVersion);
    if (responses_lite(request)) headers["x-openai-internal-codex-responses-lite"] = "true";
    else headers.erase("x-openai-internal-codex-responses-lite");
    // Current Codex sends the hyphenated names. Filo has always sent the
    // underscore form as well, so it stays.
    headers["session_id"] = session_id;
    headers["session-id"] = session_id;
    headers["thread-id"] = thread_id;
    headers["User-Agent"] = core::auth::codex_cli::user_agent();
    // codex-rs client.rs build_routing_hint_header, sent with ChatGPT auth on
    // HTTP requests and WebSocket handshakes.
    if (const std::string tier = request_service_tier(request); !tier.empty()) {
        headers["x-codex-routing-hint"] = "model=" + request.model + ";tier=" + tier;
    } else {
        headers["x-codex-routing-hint"] = "model=" + request.model;
    }
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
    const std::string continuation = websocket_continuation(request);
    // A compaction request ends with a control item that must reach the model
    // in the request that generates: never split it into prewarm + delta.
    if (continuation.empty() || request.native_compaction) {
        return {};
    }
    {
        std::lock_guard lock(transport_state_->mutex);
        if (transport_state_->last_ws_requests.contains(continuation)) {
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
    if (responses_lite(request)) {
        if (auto tools = lite_tools_item(request); !tools.empty())
            signature_input.push_back(std::move(tools));
    }
    if (const auto instructions = collect_instructions(request); !instructions.empty()) {
        Message developer{.role = "developer", .content = instructions};
        std::string item;
        append_message_item(item, developer);
        signature_input.push_back(std::move(item));
    }
    // The signature covers the request's shape: the model, its controls and
    // the stable instructions. The client metadata names the turn, so it
    // changes every turn and cannot take part in the comparison
    // (codex-rs responses_request_properties_mismatch ignores it as well).
    const std::string request_signature = codex_request_body(
        request, signature_input, std::nullopt);

    std::string incremental_previous_response_id;
    std::vector<std::string> incremental_input_items;
    // A compaction request neither continues nor becomes a baseline: its
    // trigger is a request control, not a durable conversation item.
    const std::string continuation = request.native_compaction
        ? std::string{} : websocket_continuation(request);
    if (!continuation.empty()) {
        std::lock_guard lock(transport_state_->mutex);
        if (auto last = transport_state_->last_ws_requests.find(continuation);
            last != transport_state_->last_ws_requests.end()
            && last->second.request_signature == request_signature
            && !last->second.response_id.empty()) {
            const auto& previous = last->second;
            const auto baseline_size = previous.request_input_items.size()
                + previous.response_items_added.size();
            if (full_input_items.size() >= baseline_size
                && has_prefix(full_input_items, previous.request_input_items)
                && std::equal(previous.response_items_added.begin(), previous.response_items_added.end(),
                    full_input_items.begin() + static_cast<std::ptrdiff_t>(previous.request_input_items.size()),
                    same_input_item)
                && (full_input_items.size() > baseline_size
                    || previous.from_prewarm)) {
                incremental_input_items = std::vector<std::string>{
                    full_input_items.begin() + static_cast<std::ptrdiff_t>(baseline_size),
                    full_input_items.end()};
                incremental_previous_response_id = previous.response_id;
            }
        }

        transport_state_->pending_ws_requests[continuation] =
            TransportState::PendingWebSocketRequest{
                .request_signature = request_signature,
                .request_input_items = full_input_items,
                .prewarm = prewarm,
            };
    }

    const bool use_incremental = !incremental_previous_response_id.empty();
    std::string payload = use_incremental
        ? serialize_codex_with_input_items(
            request, incremental_input_items, std::string_view{incremental_previous_response_id}, true)
        : serialize_codex_with_input_items(request, full_input_items, std::nullopt, true);

    if (prewarm) {
        append_member_before_object_end(payload, R"("generate":false)");
    }

    active_continuation_ = continuation;
    active_response_items_.clear();

    if (payload.empty() || payload.front() != '{') {
        return {};
    }
    payload.erase(payload.begin());
    return std::string{R"({"type":"response.create",)"} + payload;
}

void CodexResponsesProtocol::abandon_websocket_request(const ChatRequest& request) {
    const std::string continuation = websocket_continuation(request);
    if (continuation.empty()) {
        return;
    }

    std::lock_guard lock(transport_state_->mutex);
    transport_state_->pending_ws_requests.erase(continuation);
    transport_state_->last_ws_requests.erase(continuation);
    if (active_continuation_ == continuation) {
        active_continuation_.clear();
        active_response_items_.clear();
    }
}

ParseResult CodexResponsesProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    std::string data_scratch;
    std::string event_type;
    std::string serialized_output_item;
    std::optional<CodexRateLimitUpdate> rate_limit_update;
    std::optional<std::string> usage_error;

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
                } else if (event_type == "error") {
                    // The WebSocket wraps an HTTP failure as {"type":"error",
                    // "status":429,"error":{...},"headers":{...}}.
                    cpr::Header headers;
                    simdjson::dom::object wrapped;
                    if (doc["headers"].get(wrapped) == simdjson::SUCCESS) {
                        // codex-rs json_header_value: strings, numbers and booleans.
                        for (const auto field : wrapped) {
                            std::string_view text;
                            int64_t integer = 0;
                            double number = 0.0;
                            bool flag = false;
                            std::string value;
                            if (field.value.get(text) == simdjson::SUCCESS) value = std::string(text);
                            else if (field.value.get(integer) == simdjson::SUCCESS) value = std::to_string(integer);
                            else if (field.value.get(number) == simdjson::SUCCESS) value = std::format("{}", number);
                            else if (field.value.get(flag) == simdjson::SUCCESS) value = flag ? "true" : "false";
                            else continue;
                            headers[std::string(field.key)] = std::move(value);
                        }
                    }
                    usage_error = codex_usage_error_message(doc, headers);
                    if (usage_error) {
                        rate_limit_update = parse_codex_rate_limit_headers(headers);
                    }
                }
            }
        }
    }

    ParseResult result = OpenAIResponsesProtocol::parse_event(raw_event);

    if (rate_limit_update.has_value()) {
        rate_limit_state_->apply_event(*rate_limit_update);
        const int32_t retry_after = result.rate_limit.retry_after;
        result.rate_limit = last_rate_limit();
        result.rate_limit.retry_after = retry_after;
    }

    if (usage_error.has_value() && result.stream_error) {
        // A plan limit lifts only when the user acts or the window resets.
        result.retryable_stream_error = false;
        result.stream_error_message = *usage_error;
        for (auto& chunk : result.chunks) {
            if (chunk.is_error) chunk.content = "\n[" + *usage_error + "]";
        }
    }

    if (!serialized_output_item.empty()) {
        active_response_items_.push_back(std::move(serialized_output_item));
    }

    if (result.done
        && event_type == "response.completed"
        && !active_continuation_.empty()
        && !last_response_id().empty()) {
        std::lock_guard lock(transport_state_->mutex);
        if (auto pending = transport_state_->pending_ws_requests.find(active_continuation_);
            pending != transport_state_->pending_ws_requests.end()) {
            transport_state_->last_ws_requests[active_continuation_] =
                TransportState::LastWebSocketRequest{
                    .response_id = last_response_id(),
                    .request_signature = std::move(pending->second.request_signature),
                    .request_input_items = std::move(pending->second.request_input_items),
                    .response_items_added = std::move(active_response_items_),
                    .from_prewarm = pending->second.prewarm,
                };
            transport_state_->pending_ws_requests.erase(pending);
            // The transport owns one socket, and only its latest response can
            // be continued; retaining other complete histories wastes RAM.
            std::erase_if(transport_state_->last_ws_requests, [&](const auto& entry) {
                return entry.first != active_continuation_;
            });
        }
        active_continuation_.clear();
        active_response_items_.clear();
    }

    if (result.done && !active_continuation_.empty()) {
        std::lock_guard lock(transport_state_->mutex);
        transport_state_->pending_ws_requests.erase(active_continuation_);
        active_continuation_.clear();
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
    key += "\ncontinuation=";
    key += websocket_continuation(request);

    for (std::string_view name : {
             std::string_view{"Authorization"},
             std::string_view{"chatgpt-account-id"},
             std::string_view{"x-openai-internal-codex-responses-lite"},
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
    const auto turn_state = transport::find_header(headers, "x-codex-turn-state");
    if (turn_state.has_value()) {
        remember_turn_state(request, *turn_state);
    }
}

void CodexResponsesProtocol::observe_turn_state(const ChatRequest& request,
                                                std::string_view turn_state) {
    remember_turn_state(request, turn_state);
}

void CodexResponsesProtocol::remember_turn_state(const ChatRequest& request,
                                                 std::string_view turn_state) {
    if (request.transport_turn_id.empty() || turn_state.empty()) {
        return;
    }

    std::lock_guard lock(transport_state_->mutex);
    transport_state_->turn_states[request.transport_turn_id] = std::string(turn_state);
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
    active_continuation_.clear();
}

std::unique_ptr<ApiProtocolBase> CodexResponsesProtocol::clone() const {
    auto cloned = std::make_unique<CodexResponsesProtocol>(
        include_reasoning_encrypted_,
        default_service_tier_,
        client_identity_source_,
        usage_);
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

OpenAIResponsesProtocol::FailureVerdict
OpenAIResponsesProtocol::classify_failure(std::string_view code,
                                          std::string_view /*message*/,
                                          const cpr::Header& advice) const {
    // The generic Responses API retries only the classes it knows to be
    // transient; an unknown class fails the turn.
    const bool retryable =
        code == "server_error" || code == "internal_server_error"
        || code == "server_is_overloaded" || code == "policy_capacity_unavailable"
        || code == "rate_limit_exceeded"
        || code == "rate_limit_error" || code == "timeout"
        || code == "request_timeout" || code == "temporarily_unavailable"
        || code == "previous_response_not_found"
        || code == "websocket_connection_limit_reached";
    const int32_t delay = transport::retry_after_seconds(advice);
    return {
        .retryable = retryable,
        .retry_after_seconds = delay > 0 ? std::optional<int32_t>(delay)
                                         : std::nullopt,
    };
}

OpenAIResponsesProtocol::FailureVerdict CodexResponsesProtocol::classify_failure(
    std::string_view code,
    std::string_view message,
    const cpr::Header& advice) const {
    // codex-rs codex-api/src/sse/responses_error.rs: the ChatGPT backend
    // names the class it failed with. Only the classes a retry cannot fix
    // are terminal; a class upstream does not know is transient.
    const int32_t header_delay = transport::retry_after_seconds(advice);
    const auto delay = [header_delay](std::optional<int32_t> stated) {
        if (header_delay > 0) return std::optional<int32_t>(header_delay);
        return stated;
    };

    if (code == "rate_limit_exceeded" || code == "slow_down") {
        // A rate limit states when to come back, in the header or in the
        // message itself ("Please try again in 11.054s.").
        return {
            .retryable = true,
            .retry_after_seconds = delay(codex_retry_delay_from_message(message)),
        };
    }
    if (code == "policy_capacity_unavailable" || code == "server_is_overloaded") {
        return {
            .retryable = true,
            .retry_after_seconds = delay(std::nullopt),
        };
    }
    if (codex_terminal_failure_class(code)) {
        return {};
    }
    if (code.empty()) {
        // No class at all: upstream treats the event as terminal unless the
        // server states when to come back.
        return {
            .retryable = header_delay > 0,
            .retry_after_seconds = delay(std::nullopt),
        };
    }
    // A class upstream does not know is transient.
    return {
        .retryable = true,
        .retry_after_seconds = delay(std::nullopt),
    };
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
        // The endpoint's handle for the response it is serving: a caller that
        // wants it stopped asks for it by name.
        result.response_id = in_progress_response_id_;
        result.stream_started = true;
        return result;
    }

    // codex-rs responses_websocket.rs: the endpoint names the routing token
    // of the turn in a metadata event, so the next request of the turn stays
    // on the machine that serves it.
    if (event_type == "response.metadata"
        || event_type == "codex.response.metadata") {
        result.turn_state = parse_metadata_turn_state(doc);
        return result;
    }

    // codex-rs responses_websocket.rs: the endpoint answers an interrupt with
    // response.interrupt.accepted, or refuses it with response.interrupt.failed.
    // The answer names the response it stopped.
    if (event_type == "response.interrupt.accepted"
        || event_type == "response.interrupt.failed") {
        result.interrupt_outcome = event_type == "response.interrupt.accepted"
            ? InterruptOutcome::Accepted
            : InterruptOutcome::Rejected;
        result.response_id = parse_acknowledged_response_id(doc);
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

        // The answer to a `compaction_trigger`: codex-rs ResponseItem::
        // Compaction {id, encrypted_content}, alias `compaction_summary`.
        // codex-rs replays a prefixed id and drops a bare one
        // (client.rs prepare_response_items_for_request).
        if (item_type == "compaction" || item_type == "compaction_summary") {
            std::string_view encrypted;
            if (item["encrypted_content"].get(encrypted) != simdjson::SUCCESS || encrypted.empty()) {
                return result;
            }
            std::string payload = R"({"type":"compaction")";
            std::string_view id;
            if (item["id"].get(id) == simdjson::SUCCESS) {
                const auto separator = id.find('_');
                if (separator != std::string_view::npos && separator > 0
                    && separator + 1 < id.size()) {
                    payload += R"(,"id":")";
                    payload += core::utils::escape_json_string(id);
                    payload += '"';
                }
            }
            payload += R"(,"encrypted_content":")";
            payload += core::utils::escape_json_string(encrypted);
            payload += R"("})";
            StreamChunk chunk;
            chunk.continuation_items.push_back(ContinuationItem{
                .provider = "openai",
                .kind = std::string(kCheckpointContinuationKind),
                .payload = std::move(payload),
            });
            result.chunks.push_back(std::move(chunk));
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
        cpr::Header advice;
        for (const auto key : {"headers", "response_headers"}) {
            simdjson::dom::object headers;
            if (doc[key].get(headers) != simdjson::SUCCESS) continue;
            for (const auto field : headers) {
                std::string_view value;
                if (field.value.get(value) == simdjson::SUCCESS) advice[std::string(field.key)] = std::string(value);
            }
        }
        const auto verdict = classify_failure(code, message, advice);
        result.retryable_stream_error = !stream_retry_vetoed_
            && verdict.retryable
            && !transport::retry_vetoed(advice);
        result.rate_limit.retry_after = verdict.retry_after_seconds.value_or(0);
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
        shared_state_->sessions[active_conversation_].previous_response_id = last_response_id_;
    }
}

void CodexResponsesProtocol::on_response(const HttpResponse& response) {
    OpenAIResponsesProtocol::on_response(response);
    rate_limit_state_->apply_http(
        OpenAIResponsesProtocol::last_rate_limit(),
        parse_codex_rate_limit_headers(response.headers));
}

void CodexResponsesProtocol::enrich_rate_limit(std::string_view base_url,
                                               const cpr::Header& request_headers,
                                               const HttpResponse& response) {
    // After a transport failure the read would most likely time out too.
    if (!usage_ || response.status_code == 0) return;
    try {
        if (auto account = usage_->read(base_url, request_headers)) {
            rate_limit_state_->apply_account(std::move(*account));
        }
    } catch (...) { /* Usage is best-effort and must not fail generation. */ }
}

RateLimitInfo CodexResponsesProtocol::last_rate_limit() const noexcept {
    try {
        return rate_limit_state_->snapshot();
    } catch (...) {
        return {};
    }
}

std::string CodexResponsesProtocol::format_error_message(
    const HttpResponse& response) const {
    if (!response.body.empty()) {
        simdjson::dom::parser parser;
        const simdjson::padded_string body(response.body);
        simdjson::dom::element doc;
        if (parser.parse(body).get(doc) == simdjson::SUCCESS) {
            if (auto message = codex_usage_error_message(doc, response.headers)) {
                return "[" + *message + "]";
            }
        }
    }
    return OpenAIResponsesProtocol::format_error_message(response);
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
            std::string_view type;
            if (parser.parse(body).get(doc) == simdjson::SUCCESS
                && ((doc["error"]["code"].get(code) == simdjson::SUCCESS && is_quota_error(code))
                    || (doc["error"]["type"].get(type) == simdjson::SUCCESS
                        && is_quota_error(type)))) return false;
        } catch (...) {}
    }
    return is_openai_retryable_status(response.status_code);
}

void OpenAIResponsesProtocol::observe_response_headers(
    const cpr::Header& headers, const ChatRequest&) {
    stream_retry_vetoed_ = parse_bool_header(headers, "x-should-retry") == false;
}

} // namespace core::llm::protocols
