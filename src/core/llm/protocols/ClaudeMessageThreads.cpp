#include "ClaudeMessageThreads.hpp"

#include "../ClaudeBilling.hpp"
#include "../../logging/Logger.hpp"
#include "../../utils/JsonUtils.hpp"

#include <simdjson.h>

#include <array>
#include <regex>
#include <type_traits>
#include <variant>

namespace core::llm::protocols::claude_threads {

namespace {

constexpr std::uint64_t kDigestSeed = 0x6d736774687264ULL; // "msgthrd"

/// Fields a thread fingerprints: every request must repeat them unchanged.
constexpr std::array<std::string_view, 8> kFixedFields{
    "model", "tool_choice", "thinking", "mcp_servers", "context_management",
    "output_config", "container", "cache_control"};

[[nodiscard]] std::uint64_t digest(std::string_view bytes) noexcept {
    // 0 means "no digest" to the ledger.
    const auto value = claude::xxh64(bytes, kDigestSeed);
    return value == 0 ? 1 : value;
}

[[nodiscard]] bool is_billing_block(simdjson::dom::element block) {
    std::string_view text;
    return block["text"].get(text) == simdjson::SUCCESS
        && text.find(claude::kBillingMarker) != std::string_view::npos;
}

struct ParsedBody {
    simdjson::dom::parser parser;
    simdjson::padded_string bytes;
    simdjson::dom::object root;
    simdjson::dom::array messages;
    std::optional<simdjson::dom::array> system;

    explicit ParsedBody(std::string_view body) : bytes(body) {}

    [[nodiscard]] bool parse() {
        if (parser.parse(bytes).get_object().get(root) != simdjson::SUCCESS) return false;
        if (root["messages"].get_array().get(messages) != simdjson::SUCCESS) return false;
        // Someone else already threads this body.
        if (simdjson::dom::element thread; root["thread"].get(thread) == simdjson::SUCCESS) return false;
        simdjson::dom::element system_value;
        if (root["system"].get(system_value) == simdjson::SUCCESS) {
            simdjson::dom::array blocks;
            // A string system mixes attribution with the prompt and cannot be split.
            if (system_value.get_array().get(blocks) != simdjson::SUCCESS) return false;
            system = blocks;
        }
        return true;
    }
};

void append_member(std::string& out, std::string_view key, std::string_view value) {
    if (out.size() > 1) out += ',';
    out += '"';
    out += core::utils::escape_json_string(key);
    out += "\":";
    out += value;
}

} // namespace

std::optional<continuation::WireTranscript> wire_transcript(std::string_view unsigned_body,
                                                            std::string_view fixed_extra) {
    ParsedBody body(unsigned_body);
    if (!body.parse()) return std::nullopt;

    continuation::WireTranscript transcript;
    transcript.messages.reserve(body.messages.size());
    for (const auto message : body.messages) {
        continuation::WireMessage wire{.digest = digest(simdjson::to_string(message))};
        std::string_view role;
        simdjson::dom::element content;
        if (message["role"].get(role) == simdjson::SUCCESS && role == "assistant"
            && message["content"].get(content) == simdjson::SUCCESS) {
            wire.reply_digest = digest(simdjson::to_string(content));
        }
        transcript.messages.push_back(wire);
    }

    // The billing block changes on every request, and a continue resends it.
    std::string create_only = "system:";
    if (body.system) {
        for (const auto block : *body.system) {
            if (is_billing_block(block)) continue;
            create_only += simdjson::to_string(block);
            create_only += '\x1f';
        }
    }
    create_only += "\x1etools:";
    if (simdjson::dom::element tools; body.root["tools"].get(tools) == simdjson::SUCCESS) {
        create_only += simdjson::to_string(tools);
    }
    transcript.create_only_digest = digest(create_only);

    std::string fixed(fixed_extra);
    for (const auto field : kFixedFields) {
        fixed += '\x1e';
        fixed += field;
        fixed += ':';
        if (simdjson::dom::element value; body.root[field].get(value) == simdjson::SUCCESS) {
            fixed += simdjson::to_string(value);
        }
    }
    transcript.fixed_digest = digest(fixed);
    return transcript;
}

std::uint64_t reply_digest(std::string_view assistant_content) {
    simdjson::dom::parser parser;
    const simdjson::padded_string bytes(assistant_content);
    simdjson::dom::element content;
    if (parser.parse(bytes)["content"].get(content) != simdjson::SUCCESS) return 0;
    return digest(simdjson::to_string(content));
}

std::optional<std::string> threaded_body(std::string_view unsigned_body,
                                         const continuation::Plan& plan,
                                         InheritedFields fields) {
    if (!continuation::is_threaded(plan)) return std::nullopt;
    ParsedBody body(unsigned_body);
    if (!body.parse()) return std::nullopt;

    const auto* next = std::get_if<continuation::Continue>(&plan);
    if (next && (next->previous_message_id.empty() || next->first_new_message >= body.messages.size())) {
        return std::nullopt;
    }
    const bool inherit = next && fields != InheritedFields::full;
    const bool billing_only = fields == InheritedFields::billing_only;

    std::string out;
    out.reserve(unsigned_body.size() + 96);
    out += '{';
    for (const auto field : body.root) {
        if (next && field.key == "messages") {
            std::string messages = "[";
            std::size_t index = 0;
            for (const auto message : body.messages) {
                if (index++ < next->first_new_message) continue;
                if (messages.size() > 1) messages += ',';
                messages += simdjson::to_string(message);
            }
            messages += ']';
            append_member(out, field.key, messages);
        } else if (inherit && field.key == "system") {
            if (!billing_only) continue;
            std::string system = "[";
            for (const auto block : *body.system) {
                if (!is_billing_block(block)) continue;
                if (system.size() > 1) system += ',';
                system += simdjson::to_string(block);
            }
            system += ']';
            if (system.size() > 2) append_member(out, field.key, system);
        } else if (inherit && field.key == "tools") {
            continue;
        } else {
            append_member(out, field.key, simdjson::to_string(field.value));
        }
    }
    if (next) {
        append_member(out, "thread",
                      R"({"type":"continue","previous_message_id":")"
                          + core::utils::escape_json_string(next->previous_message_id) + "\"}");
    } else {
        append_member(out, "thread", R"({"type":"create"})");
    }
    out += '}';
    return out;
}

RejectionDetails classify_rejection(int status_code, std::string_view body) {
    if (status_code != 400 && status_code != 404) return {};
    RejectionDetails details{.kind = Rejection::other};
    simdjson::dom::parser parser;
    const simdjson::padded_string bytes(body);
    simdjson::dom::element doc;
    if (parser.parse(bytes).get(doc) != simdjson::SUCCESS) return details;
    std::string_view code;
    if (doc["error"]["details"]["error_code"].get(code) != simdjson::SUCCESS) return details;

    if (code == "thread_not_found") details.kind = Rejection::not_found;
    else if (code == "thread_fingerprint_mismatch") details.kind = Rejection::fingerprint_mismatch;
    else if (code == "thread_already_continued") details.kind = Rejection::already_continued;
    else if (code == "thread_continue_in_flight") details.kind = Rejection::continue_in_flight;
    else if (code == "thread_unsupported_request") details.kind = Rejection::unsupported_request;

    if (details.kind == Rejection::fingerprint_mismatch) {
        std::string_view message;
        details.names_inherited_fields = true;
        if (doc["error"]["message"].get(message) == simdjson::SUCCESS) {
            static const std::regex kFields(R"(differing field\(s\):\s*([A-Za-z_,\s-]+))",
                                            std::regex::icase);
            std::match_results<std::string_view::const_iterator> match;
            if (std::regex_search(message.begin(), message.end(), match, kFields)) {
                const auto named = match[1].str();
                details.names_inherited_fields = std::regex_search(
                    named, std::regex(R"(\b(system|tools)\b)", std::regex::icase));
            }
        }
    }
    return details;
}

std::string_view to_string(Rejection rejection) noexcept {
    switch (rejection) {
        case Rejection::none: return "none";
        case Rejection::not_found: return "thread_not_found";
        case Rejection::fingerprint_mismatch: return "thread_fingerprint_mismatch";
        case Rejection::already_continued: return "thread_already_continued";
        case Rejection::continue_in_flight: return "thread_continue_in_flight";
        case Rejection::unsupported_request: return "thread_unsupported_request";
        case Rejection::other: return "unrecognized";
    }
    return "unknown";
}

bool ThreadSession::suspended(std::string_view model) const {
    std::lock_guard lock(mutex_);
    return suspended_models_.contains(std::string(model));
}

void ThreadSession::escalate(InheritedFields fields) noexcept {
    const auto wanted = static_cast<std::uint8_t>(fields);
    auto current = inherited_.load(std::memory_order_acquire);
    while (current < wanted
           && !inherited_.compare_exchange_weak(current, wanted, std::memory_order_acq_rel)) {
    }
}

void ThreadSession::suspend(std::string_view model) {
    std::lock_guard lock(mutex_);
    suspended_models_.emplace(model);
}

bool RequestThread::available(std::string_view scope, std::string_view model) const noexcept {
    try {
        return session_ && !scope.empty() && !session_->suspended(model);
    } catch (...) {
        return false;
    }
}

void RequestThread::plan(std::string_view scope, std::string_view model, std::string_view unsigned_body,
                         std::string_view fixed_extra) noexcept {
    reset();
    if (!available(scope, model)) return;
    try {
        scope_ = scope;
        model_ = model;
        auto transcript = wire_transcript(unsigned_body, fixed_extra);
        if (!transcript) {
            core::logging::debug("[threads] {}: request shape cannot be threaded", scope_);
            return;
        }
        attempt_ = session_->ledger().open(scope_, std::move(*transcript));
        std::visit([&](const auto& plan) {
            using Plan = std::decay_t<decltype(plan)>;
            if constexpr (std::is_same_v<Plan, continuation::Create>) {
                core::logging::debug("[threads] {}: create ({})", scope_, continuation::to_string(plan.reason));
            } else if constexpr (std::is_same_v<Plan, continuation::Continue>) {
                const auto total = attempt_.transcript().messages.size();
                core::logging::debug("[threads] {}: continue {} sending {} of {} messages", scope_,
                                     plan.previous_message_id, total - plan.first_new_message, total);
            } else {
                core::logging::debug("[threads] {}: unthreaded ({})", scope_, continuation::to_string(plan.reason));
            }
        }, attempt_.plan());
    } catch (const std::exception& e) {
        drop(e.what());
    } catch (...) {
        drop("unknown error");
    }
}

std::optional<std::string> RequestThread::body(std::string_view unsigned_body) const noexcept {
    if (!threaded()) return std::nullopt;
    try {
        // plan() accepted these bytes' shape and the ledger keeps the cut in
        // range, so only allocation can fail here; the complete body is sent.
        auto cut = threaded_body(unsigned_body, attempt_.plan(), session_->inherited_fields());
        if (!cut) core::logging::warn("[threads] {}: the body could not be cut; sending it whole", scope_);
        return cut;
    } catch (...) {
        return std::nullopt;
    }
}

bool RequestThread::on_rejection(int status_code, std::string_view response_body) noexcept {
    if (!threaded()) return false;
    try {
        const auto rejection = classify_rejection(status_code, response_body);
        if (rejection.kind == Rejection::none) return false;
        const bool continuing = std::holds_alternative<continuation::Continue>(attempt_.plan());
        core::logging::info("[threads] {}: HTTP {} {} on a thread {}", scope_, status_code,
                            to_string(rejection.kind), continuing ? "continue" : "create");
        const auto fields = session_->inherited_fields();
        switch (rejection.kind) {
            case Rejection::not_found:
            case Rejection::already_continued:
            case Rejection::continue_in_flight:
            case Rejection::fingerprint_mismatch:
                if (continuing) {
                    if (rejection.kind == Rejection::fingerprint_mismatch && rejection.names_inherited_fields
                        && fields != InheritedFields::full) {
                        core::logging::info("[threads] continues now send the full system and tools");
                        session_->escalate(InheritedFields::full);
                    }
                    attempt_.replan(continuation::Create{continuation::CreateReason::replay_after_rejection});
                    return true;
                }
                suspend_on_success_ = true;
                break;
            case Rejection::unsupported_request:
                session_->suspend(model_);
                break;
            case Rejection::other:
                // A continue that sent the billing block may be refused without
                // a code; the documented shape sends no system at all. The
                // rejection left the thread unchanged, so the same continue
                // can go again.
                if (continuing && fields == InheritedFields::billing_only) {
                    core::logging::info("[threads] continues now leave out the system prompt");
                    session_->escalate(InheritedFields::omitted);
                    return true;
                }
                suspend_on_success_ = true;
                break;
            case Rejection::none:
                return false;
        }
    } catch (...) {
        // Classifying is best effort; the plain request is always the answer.
    }
    attempt_.replan(continuation::Unthreaded{continuation::UnthreadedReason::rejected});
    return true;
}

void RequestThread::on_success() noexcept {
    if (!std::exchange(suspend_on_success_, false) || !session_) return;
    try {
        // The same request without the thread was accepted: threads are what
        // this model refuses here.
        core::logging::warn("[threads] {} refused a thread request the plain request passes; "
                            "sending it unthreaded for the rest of the session", model_);
        session_->suspend(model_);
    } catch (...) {
        // Without the suspension the next request walks the ladder again.
    }
}

void RequestThread::observe_event(std::string_view raw_event) noexcept {
    // A reply a fallback model wrote is not stored in the thread.
    if (threaded() && raw_event.find("fallback_message") != std::string_view::npos) fallback_reply_ = true;
}

void RequestThread::observe_response_id(std::string_view id) noexcept {
    if (!threaded()) return;
    try {
        attempt_.observe_response_id(id);
    } catch (...) {
        attempt_.abandon();
    }
}

void RequestThread::settle(const ReplyOutcome& outcome) noexcept {
    if (!threaded()) return;
    try {
        // Not stored, or stored with a turn that must not be continued; and
        // without the reply's native blocks the next request cannot prove it
        // replays the stored turn.
        if (outcome.stop_reason == "refusal" || outcome.stop_reason == "pause_turn"
            || outcome.incomplete_tool_call || fallback_reply_ || !outcome.assistant_content) {
            attempt_.abandon();
            return;
        }
        attempt_.commit(reply_digest(*outcome.assistant_content));
    } catch (...) {
        attempt_.abandon();
    }
}

void RequestThread::reset() noexcept {
    attempt_ = {};
    scope_.clear();
    model_.clear();
    fallback_reply_ = false;
    suspend_on_success_ = false;
}

void RequestThread::drop(std::string_view why) noexcept {
    try {
        if (threaded()) core::logging::warn("[threads] {}: sending this request unthreaded: {}", scope_, why);
    } catch (...) {
    }
    attempt_.replan(continuation::Unthreaded{continuation::UnthreadedReason::dropped});
}

} // namespace core::llm::protocols::claude_threads
