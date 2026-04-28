#include "OpenAIChatCompletionStreamEncoder.hpp"

#include "../../core/utils/JsonWriter.hpp"

namespace exec::gateway::openai {

using core::utils::JsonWriter;

namespace {

void write_usage(JsonWriter& writer, const core::llm::TokenUsage& usage) {
    auto _usage = writer.object();
    writer.kv_num("prompt_tokens", usage.prompt_tokens).comma()
        .kv_num("completion_tokens", usage.completion_tokens).comma()
        .kv_num("total_tokens", usage.total_tokens);
}

void write_chunk_prefix(JsonWriter& writer, const StreamChunkContext& context) {
    writer.kv_str("id", context.response_id).comma()
        .kv_str("object", "chat.completion.chunk").comma()
        .kv_num("created", context.created).comma()
        .kv_str("model", context.model).comma();
}

void maybe_write_usage_null(JsonWriter& writer, const StreamChunkContext& context) {
    if (context.include_usage) {
        writer.comma().kv_null("usage");
    }
}

} // namespace

std::string error_body(std::string_view message) {
    JsonWriter writer(256 + message.size());
    {
        auto _root = writer.object();
        writer.key("error");
        {
            auto _error = writer.object();
            writer.kv_str("message", message).comma()
                .kv_str("type", "api_error");
        }
    }
    return std::move(writer).take();
}

std::string role_chunk(const StreamChunkContext& context) {
    JsonWriter writer(512);
    {
        auto _root = writer.object();
        write_chunk_prefix(writer, context);
        writer.key("choices");
        {
            auto _choices = writer.array();
            auto _choice = writer.object();
            writer.kv_num("index", 0).comma().key("delta");
            {
                auto _delta = writer.object();
                writer.kv_str("role", "assistant");
            }
            writer.comma().kv_null("finish_reason");
        }
        maybe_write_usage_null(writer, context);
    }
    return std::move(writer).take();
}

std::string content_chunk(const StreamChunkContext& context,
                          std::string_view content) {
    JsonWriter writer(512 + content.size());
    {
        auto _root = writer.object();
        write_chunk_prefix(writer, context);
        writer.key("choices");
        {
            auto _choices = writer.array();
            auto _choice = writer.object();
            writer.kv_num("index", 0).comma().key("delta");
            {
                auto _delta = writer.object();
                writer.kv_str("content", content);
            }
            writer.comma().kv_null("finish_reason");
        }
        maybe_write_usage_null(writer, context);
    }
    return std::move(writer).take();
}

std::string tool_calls_chunk(
    const StreamChunkContext& context,
    const std::vector<core::llm::ToolCall>& tool_calls) {
    JsonWriter writer(1024);
    {
        auto _root = writer.object();
        write_chunk_prefix(writer, context);
        writer.key("choices");
        {
            auto _choices = writer.array();
            auto _choice = writer.object();
            writer.kv_num("index", 0).comma().key("delta");
            {
                auto _delta = writer.object();
                writer.key("tool_calls");
                {
                    auto _tool_calls = writer.array();
                    for (std::size_t i = 0; i < tool_calls.size(); ++i) {
                        const auto& tool_call = tool_calls[i];
                        if (i > 0) writer.comma();

                        auto _tool = writer.object();
                        const int index = tool_call.index >= 0
                            ? tool_call.index
                            : static_cast<int>(i);
                        writer.kv_num("index", index);
                        if (!tool_call.id.empty()) {
                            writer.comma().kv_str("id", tool_call.id);
                        }
                        if (!tool_call.type.empty()) {
                            writer.comma().kv_str("type", tool_call.type);
                        }
                        writer.comma().key("function");
                        {
                            auto _function = writer.object();
                            bool wrote_function_field = false;
                            if (!tool_call.function.name.empty()) {
                                writer.kv_str("name", tool_call.function.name);
                                wrote_function_field = true;
                            }
                            if (!tool_call.function.arguments.empty()) {
                                if (wrote_function_field) writer.comma();
                                writer.kv_str("arguments", tool_call.function.arguments);
                            }
                        }
                    }
                }
            }
            writer.comma().kv_null("finish_reason");
        }
        maybe_write_usage_null(writer, context);
    }
    return std::move(writer).take();
}

std::string finish_chunk(const StreamChunkContext& context,
                         std::string_view finish_reason) {
    JsonWriter writer(512);
    {
        auto _root = writer.object();
        write_chunk_prefix(writer, context);
        writer.key("choices");
        {
            auto _choices = writer.array();
            auto _choice = writer.object();
            writer.kv_num("index", 0).comma().key("delta");
            {
                auto _delta = writer.object();
            }
            writer.comma().kv_str("finish_reason", finish_reason);
        }
        maybe_write_usage_null(writer, context);
    }
    return std::move(writer).take();
}

std::string usage_chunk(const StreamChunkContext& context,
                        const core::llm::TokenUsage& usage) {
    JsonWriter writer(512);
    {
        auto _root = writer.object();
        write_chunk_prefix(writer, context);
        writer.key("choices");
        {
            auto _choices = writer.array();
        }
        writer.comma().key("usage");
        write_usage(writer, usage);
    }
    return std::move(writer).take();
}

} // namespace exec::gateway::openai
