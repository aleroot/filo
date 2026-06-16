#include "ModelCatalogProvider.hpp"

#include "../utils/StringUtils.hpp"
#include "../utils/UriUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace core::llm {
namespace {

constexpr ModelCapabilities CAP_TEXT =
    static_cast<uint32_t>(ModelCapability::TextInput) |
    static_cast<uint32_t>(ModelCapability::TextOutput) |
    static_cast<uint32_t>(ModelCapability::Streaming) |
    static_cast<uint32_t>(ModelCapability::SystemPrompts);

constexpr ModelCapabilities CAP_TOOLS =
    CAP_TEXT |
    static_cast<uint32_t>(ModelCapability::FunctionCalling) |
    static_cast<uint32_t>(ModelCapability::ParallelToolCalls);

constexpr ModelCapabilities CAP_FULL =
    CAP_TOOLS |
    static_cast<uint32_t>(ModelCapability::JsonMode) |
    static_cast<uint32_t>(ModelCapability::Vision);

[[nodiscard]] std::string to_string_copy(std::string_view value) {
    return std::string(value.data(), value.size());
}

[[nodiscard]] std::string lower_ascii(std::string_view input) {
    return core::utils::str::to_lower_ascii_copy(input);
}

[[nodiscard]] bool contains_ascii(std::string_view haystack, std::string_view needle) {
    return lower_ascii(haystack).find(lower_ascii(needle)) != std::string::npos;
}

[[nodiscard]] std::string strip_prefix(std::string_view value, std::string_view prefix) {
    if (value.starts_with(prefix)) {
        value.remove_prefix(prefix.size());
    }
    return to_string_copy(value);
}

bool get_string_field(simdjson::dom::object object,
                      const char* key,
                      std::string& out) {
    std::string_view value;
    if (object[key].get(value) != simdjson::SUCCESS) {
        return false;
    }
    out = to_string_copy(value);
    return true;
}

[[nodiscard]] int32_t clamp_i32(int64_t value) {
    if (value < 0) return 0;
    if (value > std::numeric_limits<int32_t>::max()) return std::numeric_limits<int32_t>::max();
    return static_cast<int32_t>(value);
}

[[nodiscard]] int32_t get_int_field(simdjson::dom::object object,
                                    const char* key,
                                    int32_t fallback = 0) {
    int64_t value = 0;
    if (object[key].get(value) != simdjson::SUCCESS) {
        return fallback;
    }
    return clamp_i32(value);
}

[[nodiscard]] bool get_bool_field(simdjson::dom::object object,
                                  const char* key,
                                  bool fallback = false) {
    bool value = false;
    if (object[key].get(value) != simdjson::SUCCESS) {
        return fallback;
    }
    return value;
}

[[nodiscard]] bool array_contains_string(simdjson::dom::object object,
                                         const char* key,
                                         std::string_view needle) {
    simdjson::dom::array array;
    if (object[key].get(array) != simdjson::SUCCESS) {
        return false;
    }

    for (simdjson::dom::element item : array) {
        std::string_view value;
        if (item.get(value) == simdjson::SUCCESS && value == needle) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool array_contains_object_string(simdjson::dom::object object,
                                                const char* key,
                                                const char* field,
                                                std::string_view needle) {
    simdjson::dom::array array;
    if (object[key].get(array) != simdjson::SUCCESS) {
        return false;
    }

    for (simdjson::dom::element item : array) {
        simdjson::dom::object nested;
        if (item.get(nested) != simdjson::SUCCESS) {
            continue;
        }
        std::string value;
        if (get_string_field(nested, field, value) && value == needle) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] ParameterConstraints standard_constraints(double max_temperature = 2.0) {
    ParameterConstraints constraints;
    constraints.temperature = {0.0, max_temperature};
    constraints.top_p = {0.0, 1.0};
    constraints.frequency_penalty = {-2.0, 2.0};
    constraints.presence_penalty = {-2.0, 2.0};
    return constraints;
}

[[nodiscard]] ModelTier infer_tier(std::string_view model_id, bool reasoning = false) {
    if (reasoning || contains_ascii(model_id, "thinking")) {
        return ModelTier::Reasoning;
    }
    if (contains_ascii(model_id, "flash") ||
        contains_ascii(model_id, "lite") ||
        contains_ascii(model_id, "mini") ||
        contains_ascii(model_id, "nano") ||
        contains_ascii(model_id, "haiku")) {
        return ModelTier::Fast;
    }
    if (contains_ascii(model_id, "pro") ||
        contains_ascii(model_id, "opus") ||
        contains_ascii(model_id, "max")) {
        return ModelTier::Powerful;
    }
    return ModelTier::Balanced;
}

[[nodiscard]] bool is_embedding_model(std::string_view model_id) {
    return contains_ascii(model_id, "embedding") || contains_ascii(model_id, "embed");
}

[[nodiscard]] bool is_reasoning_model(std::string_view model_id) {
    const auto lower = lower_ascii(model_id);
    return lower.starts_with("o1") ||
           lower.starts_with("o3") ||
           lower.starts_with("o4") ||
           lower.starts_with("gpt-5") ||
           contains_ascii(model_id, "reasoning") ||
           contains_ascii(model_id, "thinking");
}

[[nodiscard]] ModelCapabilities openai_codex_capabilities(std::string_view model_id,
                                                          simdjson::dom::object object) {
    if (is_embedding_model(model_id)) {
        return static_cast<uint32_t>(ModelCapability::Embeddings);
    }

    ModelCapabilities caps = CAP_TEXT |
        static_cast<uint32_t>(ModelCapability::FunctionCalling) |
        static_cast<uint32_t>(ModelCapability::JsonMode);
    if (get_bool_field(object, "supports_parallel_tool_calls")) {
        caps |= static_cast<uint32_t>(ModelCapability::ParallelToolCalls);
    }
    if (get_bool_field(object, "supports_image_detail_original") ||
        array_contains_string(object, "input_modalities", "image") ||
        contains_ascii(model_id, "gpt-4") ||
        contains_ascii(model_id, "gpt-5") ||
        contains_ascii(model_id, "4o") ||
        contains_ascii(model_id, "vision")) {
        caps |= static_cast<uint32_t>(ModelCapability::Vision);
    }
    std::string default_reasoning_level;
    if (is_reasoning_model(model_id) ||
        get_string_field(object, "default_reasoning_level", default_reasoning_level) ||
        array_contains_object_string(object, "supported_reasoning_levels", "effort", "minimal") ||
        array_contains_object_string(object, "supported_reasoning_levels", "effort", "low") ||
        array_contains_object_string(object, "supported_reasoning_levels", "effort", "medium") ||
        array_contains_object_string(object, "supported_reasoning_levels", "effort", "high") ||
        array_contains_object_string(object, "supported_reasoning_levels", "effort", "xhigh")) {
        caps |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }
    return caps;
}

[[nodiscard]] int32_t infer_kimi_context_window(std::string_view model_id) {
    if (contains_ascii(model_id, "moonshot-v1-8k")) return 8192;
    if (contains_ascii(model_id, "moonshot-v1-32k")) return 32768;
    if (contains_ascii(model_id, "moonshot-v1-128k")) return 128000;
    if (contains_ascii(model_id, "kimi-k2.7-code") ||
        contains_ascii(model_id, "kimi-k2.6") ||
        contains_ascii(model_id, "kimi-k2-6") ||
        contains_ascii(model_id, "kimi-k2.5") ||
        contains_ascii(model_id, "kimi-k2-5") ||
        contains_ascii(model_id, "kimi-for-coding")) {
        return 256000;
    }
    return 128000;
}

[[nodiscard]] ModelCapabilities kimi_capabilities(std::string_view model_id,
                                                  bool supports_reasoning,
                                                  bool supports_image_in,
                                                  bool supports_video_in) {
    ModelCapabilities caps = CAP_TOOLS | static_cast<uint32_t>(ModelCapability::JsonMode);
    if (supports_reasoning ||
        contains_ascii(model_id, "thinking") ||
        contains_ascii(model_id, "reason") ||
        contains_ascii(model_id, "kimi-for-coding") ||
        contains_ascii(model_id, "kimi-code")) {
        caps |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }
    if (supports_image_in ||
        supports_video_in ||
        contains_ascii(model_id, "vl") ||
        contains_ascii(model_id, "vision") ||
        contains_ascii(model_id, "kimi-k2") ||
        contains_ascii(model_id, "kimi-for-coding") ||
        contains_ascii(model_id, "kimi-code")) {
        caps |= static_cast<uint32_t>(ModelCapability::Vision);
    }
    if (supports_video_in ||
        contains_ascii(model_id, "kimi-k2") ||
        contains_ascii(model_id, "kimi-for-coding") ||
        contains_ascii(model_id, "kimi-code")) {
        caps |= static_cast<uint32_t>(ModelCapability::VideoInput);
    }
    return caps;
}

[[nodiscard]] ModelCapabilities gemini_capabilities(std::string_view model_id,
                                                    bool generate_content,
                                                    bool count_tokens,
                                                    bool embed_content,
                                                    bool thinking) {
    if (embed_content || is_embedding_model(model_id)) {
        return static_cast<uint32_t>(ModelCapability::Embeddings);
    }

    ModelCapabilities caps = generate_content ? CAP_FULL : CAP_TEXT;
    caps |= static_cast<uint32_t>(ModelCapability::PromptCaching);
    if (count_tokens) {
        caps |= static_cast<uint32_t>(ModelCapability::TokenCounting);
    }
    if (thinking) {
        caps |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }
    return caps;
}

[[nodiscard]] ModelCatalogResult parse_json(std::string_view body,
                                            simdjson::dom::element& doc,
                                            simdjson::dom::parser& parser) {
    ModelCatalogResult result;
    if (body.empty()) {
        result.error = "empty model catalog response";
        return result;
    }

    if (parser.parse(body.data(), body.size()).get(doc) != simdjson::SUCCESS) {
        result.error = "invalid model catalog JSON";
    }
    return result;
}

} // namespace

GeminiModelCatalogProvider::GeminiModelCatalogProvider(std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "gemini";
    }
}

std::string_view GeminiModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string GeminiModelCatalogProvider::model_list_path(std::string_view page_token) const {
    std::string path = "/v1beta/models?pageSize=1000";
    if (!page_token.empty()) {
        path += "&pageToken=";
        path += core::utils::uri::percent_encode_uri_query_component(page_token);
    }
    return path;
}

ModelCatalogResult GeminiModelCatalogProvider::parse_models_response(std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    ModelCatalogResult result = parse_json(body, doc, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (doc["models"].get(models) != simdjson::SUCCESS) {
        result.error = "Gemini model catalog response is missing models[]";
        return result;
    }
    get_string_field(doc.get_object(), "nextPageToken", result.next_page_token);

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;

        std::string name;
        if (!get_string_field(object, "name", name)) continue;

        std::string base_model_id;
        get_string_field(object, "baseModelId", base_model_id);

        const std::string id = !base_model_id.empty()
            ? base_model_id
            : strip_prefix(name, "models/");
        if (id.empty()) continue;

        const bool generate_content = array_contains_string(object, "supportedGenerationMethods", "generateContent");
        const bool count_tokens = array_contains_string(object, "supportedGenerationMethods", "countTokens");
        const bool embed_content = array_contains_string(object, "supportedGenerationMethods", "embedContent") ||
                                   array_contains_string(object, "supportedGenerationMethods", "batchEmbedContents");

        bool thinking = false;
        (void)object["thinking"].get(thinking);
        thinking = thinking || contains_ascii(id, "thinking");

        ModelInfo info;
        info.canonical_id = id;
        info.provider = provider_name_;
        if (!get_string_field(object, "displayName", info.display_name)) {
            info.display_name = id;
        }
        info.context_window = get_int_field(object, "inputTokenLimit", 1048576);
        info.max_output_tokens = get_int_field(object, "outputTokenLimit", 65536);
        info.capabilities = gemini_capabilities(id, generate_content, count_tokens, embed_content, thinking);
        info.tier = infer_tier(id, thinking);
        info.constraints = standard_constraints();

        result.models.push_back(std::move(info));
    }

    return result;
}

OpenAICompatibleModelCatalogProvider::OpenAICompatibleModelCatalogProvider(std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "openai";
    }
}

std::string_view OpenAICompatibleModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string OpenAICompatibleModelCatalogProvider::model_list_path(std::string_view page_token) const {
    (void)page_token;
    return "/models";
}

ModelCatalogResult OpenAICompatibleModelCatalogProvider::parse_models_response(std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    ModelCatalogResult result = parse_json(body, doc, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (doc["data"].get(models) == simdjson::SUCCESS) {
        for (simdjson::dom::element element : models) {
            simdjson::dom::object object;
            if (element.get(object) != simdjson::SUCCESS) continue;

            ModelInfo info;
            if (!get_string_field(object, "id", info.canonical_id) || info.canonical_id.empty()) {
                continue;
            }
            info.provider = provider_name_;
            if (!get_string_field(object, "display_name", info.display_name)) {
                info.display_name = info.canonical_id;
            }
            if (is_embedding_model(info.canonical_id)) {
                info.capabilities = static_cast<uint32_t>(ModelCapability::Embeddings);
            }
            info.context_window = get_int_field(object, "context_window", info.context_window);
            info.max_output_tokens = get_int_field(object, "max_output_tokens", info.max_output_tokens);
            info.max_reasoning_tokens = get_int_field(
                object,
                "max_reasoning_tokens",
                info.max_reasoning_tokens);

            result.models.push_back(std::move(info));
        }

        return result;
    }

    if (doc["models"].get(models) != simdjson::SUCCESS) {
        result.error = "OpenAI-compatible model catalog response is missing data[] or models[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;

        if (!get_bool_field(object, "supported_in_api", true)) {
            continue;
        }

        ModelInfo info;
        if (!get_string_field(object, "slug", info.canonical_id) &&
            !get_string_field(object, "id", info.canonical_id)) {
            continue;
        }
        if (info.canonical_id.empty()) continue;

        info.provider = provider_name_;
        if (!get_string_field(object, "display_name", info.display_name)) {
            info.display_name = info.canonical_id;
        }
        info.capabilities = openai_codex_capabilities(info.canonical_id, object);
        const int32_t codex_context_window = get_int_field(object, "context_window");
        const int32_t codex_max_context_window = get_int_field(object, "max_context_window");
        if (codex_context_window > 0) {
            info.context_window = codex_context_window;
        } else if (codex_max_context_window > 0) {
            info.context_window = codex_max_context_window;
        }
        info.max_output_tokens = get_int_field(object, "max_output_tokens");
        info.max_reasoning_tokens = get_int_field(
            object,
            "max_reasoning_tokens");
        info.tier = infer_tier(info.canonical_id, info.supports(ModelCapability::Reasoning));
        info.constraints = standard_constraints();

        result.models.push_back(std::move(info));
    }

    return result;
}

KimiModelCatalogProvider::KimiModelCatalogProvider(std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "kimi";
    }
}

std::string_view KimiModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string KimiModelCatalogProvider::model_list_path(std::string_view page_token) const {
    (void)page_token;
    return "/models";
}

ModelCatalogResult KimiModelCatalogProvider::parse_models_response(std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    ModelCatalogResult result = parse_json(body, doc, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (doc["data"].get(models) != simdjson::SUCCESS) {
        result.error = "Kimi model catalog response is missing data[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;

        ModelInfo info;
        if (!get_string_field(object, "id", info.canonical_id) || info.canonical_id.empty()) {
            continue;
        }
        info.provider = provider_name_;
        if (!get_string_field(object, "display_name", info.display_name)) {
            info.display_name = info.canonical_id;
        }
        info.context_window = get_int_field(
            object,
            "context_length",
            infer_kimi_context_window(info.canonical_id));
        info.max_output_tokens = get_int_field(object, "max_output_tokens", 8192);

        const bool supports_reasoning = get_bool_field(object, "supports_reasoning");
        const bool supports_image_in = get_bool_field(object, "supports_image_in");
        const bool supports_video_in = get_bool_field(object, "supports_video_in");

        info.capabilities = kimi_capabilities(
            info.canonical_id,
            supports_reasoning,
            supports_image_in,
            supports_video_in);
        info.tier = infer_tier(info.canonical_id, info.supports(ModelCapability::Reasoning));
        info.constraints = standard_constraints();

        result.models.push_back(std::move(info));
    }

    return result;
}

AnthropicModelCatalogProvider::AnthropicModelCatalogProvider(std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "anthropic";
    }
}

std::string_view AnthropicModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string AnthropicModelCatalogProvider::model_list_path(std::string_view page_token) const {
    std::string path = "/v1/models?limit=1000";
    if (!page_token.empty()) {
        path += "&after_id=";
        path += core::utils::uri::percent_encode_uri_query_component(page_token);
    }
    return path;
}

ModelCatalogResult AnthropicModelCatalogProvider::parse_models_response(std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    ModelCatalogResult result = parse_json(body, doc, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (doc["data"].get(models) != simdjson::SUCCESS) {
        result.error = "Anthropic model catalog response is missing data[]";
        return result;
    }
    bool has_more = false;
    if (doc["has_more"].get(has_more) == simdjson::SUCCESS && has_more) {
        get_string_field(doc.get_object(), "last_id", result.next_page_token);
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;

        ModelInfo info;
        if (!get_string_field(object, "id", info.canonical_id) || info.canonical_id.empty()) {
            continue;
        }
        info.provider = provider_name_;
        if (!get_string_field(object, "display_name", info.display_name)) {
            info.display_name = info.canonical_id;
        }

        result.models.push_back(std::move(info));
    }

    return result;
}

std::unique_ptr<ModelCatalogProvider>
make_model_catalog_provider(config::ApiType api_type, std::string_view provider_name) {
    switch (api_type) {
        case config::ApiType::Gemini:
            return std::make_unique<GeminiModelCatalogProvider>(to_string_copy(provider_name));
        case config::ApiType::Anthropic:
            return std::make_unique<AnthropicModelCatalogProvider>(to_string_copy(provider_name));
        case config::ApiType::Kimi:
            return std::make_unique<KimiModelCatalogProvider>(to_string_copy(provider_name));
        case config::ApiType::OpenAI:
        case config::ApiType::DashScope:
            return std::make_unique<OpenAICompatibleModelCatalogProvider>(to_string_copy(provider_name));
        case config::ApiType::Unknown:
        case config::ApiType::Ollama:
        case config::ApiType::LlamaCppLocal:
            return nullptr;
    }
    return nullptr;
}

} // namespace core::llm
