#include "WebBackendAdapters.hpp"

#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../llm/MimoModelTraits.hpp"
#include "../llm/ProviderDefinition.hpp"
#include "../llm/protocols/MimoProtocol.hpp"
#include "../llm/protocols/SseUtils.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace core::tools::web {
namespace {

constexpr std::size_t kMaxSearchBytes = 4 * 1024 * 1024;
constexpr std::string_view kPluginHelp =
    "Enable the Web Search Plugin and check its quota at "
    "https://platform.xiaomimimo.com/console/plugin.";

[[nodiscard]] std::string redact_credentials(std::string text,
                                             const core::auth::AuthInfo& auth) {
    const auto redact = [&](std::string_view secret) {
        if (secret.empty()) return;
        std::size_t pos = 0;
        while ((pos = text.find(secret, pos)) != std::string::npos) {
            text.replace(pos, secret.size(), "[REDACTED]");
            pos += std::string_view("[REDACTED]").size();
        }
    };
    for (const auto& [name, value] : auth.headers) {
        redact(value);
        if (core::utils::ascii::iequals(name, "Authorization")
            && core::utils::ascii::istarts_with(value, "Bearer ")) {
            redact(std::string_view(value).substr(7));
        }
    }
    for (const auto& [name, value] : auth.query_params) redact(value);
    return text;
}

[[nodiscard]] bool is_mimo_provider(const core::llm::ProviderMetadata& metadata) {
    const auto* definition = core::llm::find_builtin_provider_definition(metadata.provider_name);
    return metadata.api_type == core::config::ApiType::OpenAI
        && ((definition && definition->registry_provider == "mimo")
            || core::llm::is_mimo_endpoint(metadata.base_url));
}

[[nodiscard]] std::string search_url(std::string base_url) {
    base_url = core::utils::str::trim_trailing_slashes(base_url);
    if (base_url.ends_with("/chat/completions")) return base_url;
    if (!base_url.ends_with("/v1")) base_url += "/v1";
    return base_url + "/chat/completions";
}

[[nodiscard]] std::string search_payload(const SearchRequest& request,
                                         std::string_view model) {
    // XiaomiMiMo/MiMo-Code, packages/cli/src/tool/websearch/mimo.ts:
    // force a hosted search on a separate turn and stop at the source packet.
    core::utils::JsonWriter writer(512);
    {
        auto _root = writer.object();
        writer.kv_str("model", model).comma().key("messages");
        {
            auto _messages = writer.array();
            auto _message = writer.object();
            writer.kv_str("role", "user").comma().kv_str("content", request.query);
        }
        writer.comma().key("tools");
        {
            auto _tools = writer.array();
            auto _tool = writer.object();
            writer.kv_str("type", "web_search").comma()
                  .kv_num("max_keyword", 3).comma()
                  .kv_bool("force_search", true).comma()
                  // Keep the reference client's budget of at most five hits
                  // per keyword; the returned list has a separate total cap.
                  .kv_num("limit", std::clamp(request.limit, 1, 5));
        }
        writer.comma().kv_num("max_completion_tokens", 256).comma()
              .kv_float("temperature", 1.0).comma().kv_float("top_p", 0.95).comma()
              .kv_bool("stream", true).comma().key("thinking");
        {
            auto _thinking = writer.object();
            writer.kv_str("type", "disabled");
        }
    }
    return std::move(writer).take();
}

struct SearchStream {
    SearchResponse result{.backend = "mimo-native-web-search"};
    std::string error;

    void event(std::string_view frame, int limit, const core::auth::AuthInfo& auth) {
        core::llm::protocols::sse::ParsedEventView event;
        if (!core::llm::protocols::sse::parse_event_payload(frame, event) || event.is_done) return;
        simdjson::dom::parser parser;
        simdjson::padded_string padded(event.data);
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return;
        if (simdjson::dom::element api_error;
            doc["error"].get(api_error) == simdjson::SUCCESS && !api_error.is_null()) {
            error = "MiMo web search API error: "
                + redact_credentials(simdjson::minify(api_error), auth).substr(0, 1000);
            return;
        }
        simdjson::dom::element choice;
        if (doc["choices"].at(0).get(choice) != simdjson::SUCCESS) return;
        // Gateways may send annotations in delta (SSE) or message (JSON).
        for (const auto field : {"delta", "message"}) {
            simdjson::dom::array annotations;
            if (choice[field]["annotations"].get_array().get(annotations) != simdjson::SUCCESS) continue;
            for (const auto annotation : annotations) {
                if (result.results.size() >= static_cast<std::size_t>(limit)) break;
                simdjson::dom::object object;
                if (annotation.get_object().get(object) != simdjson::SUCCESS) continue;
                SearchHit hit{
                    .title = core::utils::json::first_string_field_or_empty(object, {"title", "site_name"}),
                    .url = core::utils::json::first_string_field_or_empty(object, {"url"}),
                    .snippet = core::utils::json::first_string_field_or_empty(object, {"summary"}),
                };
                if (hit.url.empty()) continue;
                const auto date = core::utils::json::first_string_field_or_empty(object, {"publish_time"});
                if (!date.empty()) hit.snippet = date + (hit.snippet.empty() ? "" : " — " + hit.snippet);
                hit.title = redact_credentials(std::move(hit.title), auth);
                hit.url = redact_credentials(std::move(hit.url), auth);
                hit.snippet = redact_credentials(std::move(hit.snippet), auth);
                detail::add_hit_if_present(result, std::move(hit));
            }
        }
    }
};

class MimoWebSearchBackend final : public IWebSearchBackend {
public:
    explicit MimoWebSearchBackend(MimoWebSearchOptions options) : options_(options) {
        if (options_.timeout_ms <= 0) options_.timeout_ms = 30000;
    }

    [[nodiscard]] std::string_view name() const noexcept override {
        return "mimo-native-web-search";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata && metadata->credential_source && !metadata->base_url.empty()
            && is_mimo_provider(*metadata);
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request, const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata || !supports(context)) {
            return std::unexpected("MiMo web search requires an active MiMo provider with credentials.");
        }
        if (!request.domains.allowed_domains.empty() || !request.domains.blocked_domains.empty()) {
            return std::unexpected("MiMo web search does not support allowed_domains or blocked_domains filters.");
        }
        const auto model = context.model_name.empty() ? metadata->default_model : context.model_name;
        if (model.empty()) return std::unexpected("MiMo web search requires an active model.");
        const auto cancelled = [&] { return context.cancellation_requested && context.cancellation_requested(); };
        if (cancelled()) return std::unexpected("MiMo web search cancelled.");

        core::llm::protocols::MimoProtocol protocol;
        const auto auth = metadata->credential_source->get_auth();
        auto headers = protocol.build_headers(auth);
        headers["Accept"] = "text/event-stream";
        SearchStream stream;
        std::string buffer, error_body;
        std::size_t received_bytes = 0;
        const int limit = std::clamp(request.limit, 1, 20);
        // Use the effective chat gateway. In particular, Token Plan keys must
        // never be redirected to the pay-as-you-go catalog endpoint.
        const auto response = cpr::Post(
            cpr::Url{search_url(metadata->base_url)}, headers,
            cpr::Body{search_payload(request, model)},
            cpr::Timeout{options_.timeout_ms}, cpr::Redirect{false},
            cpr::ProgressCallback{[&](cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t,
                cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t, intptr_t) { return !cancelled(); }},
            cpr::WriteCallback{[&](std::string_view bytes, intptr_t) {
                if (cancelled()) return false;
                received_bytes += bytes.size();
                if (received_bytes > kMaxSearchBytes) return false;
                // Keep the bounded body intact until redaction: truncating
                // first could expose a partial credential echoed by a proxy.
                error_body.append(bytes);
                buffer.append(bytes);
                while (true) {
                    auto end = buffer.find("\n\n");
                    std::size_t delimiter = 2;
                    const auto crlf = buffer.find("\r\n\r\n");
                    if (crlf != std::string::npos && (end == std::string::npos || crlf < end)) {
                        end = crlf;
                        delimiter = 4;
                    }
                    if (end == std::string::npos) break;
                    stream.event(std::string_view(buffer).substr(0, end), limit, auth);
                    buffer.erase(0, end + delimiter);
                    if (!stream.result.results.empty() || !stream.error.empty()) return false;
                }
                return true;
            }});
        if (cancelled()) return std::unexpected("MiMo web search cancelled.");
        if (received_bytes > kMaxSearchBytes) return std::unexpected("MiMo web search response exceeded its size limit.");
        if (response.status_code == 409) {
            return std::unexpected("MiMo web search quota exhausted. " + std::string(kPluginHelp));
        }
        if (response.status_code != 0 && (response.status_code < 200 || response.status_code >= 300)) {
            return std::unexpected(protocol.format_error_message(
                {static_cast<int>(response.status_code), redact_credentials(std::move(error_body), auth), response.header}));
        }
        // Also accept an unframed JSON response or a final SSE frame without
        // the trailing blank line. Both still require real citation sources.
        if (!buffer.empty() && stream.result.results.empty() && stream.error.empty()) stream.event(buffer, limit, auth);
        if (!stream.error.empty()) return std::unexpected(stream.error);
        // The write callback deliberately aborts once sources arrive, just as
        // MiMo Code's Stream.runHead does. CURLE_WRITE_ERROR is expected here.
        if (response.status_code >= 200 && response.status_code < 300 && !stream.result.results.empty()) {
            return std::move(stream.result);
        }
        if (response.error.code != cpr::ErrorCode::OK) {
            // Transport messages may contain endpoint userinfo or query
            // strings; the numeric error conveys the failure without them.
            return std::unexpected("MiMo web search request failed (transport code "
                + std::to_string(static_cast<int>(response.error.code)) + ").");
        }
        return std::unexpected("MiMo web search did not return search sources. " + std::string(kPluginHelp));
    }

private:
    MimoWebSearchOptions options_;
};

} // namespace

std::shared_ptr<IWebSearchBackend> make_mimo_web_search_backend(MimoWebSearchOptions options) {
    return std::make_shared<MimoWebSearchBackend>(options);
}

} // namespace core::tools::web
