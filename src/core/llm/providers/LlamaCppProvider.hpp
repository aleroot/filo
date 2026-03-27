#pragma once

#include "../LLMProvider.hpp"
#include "core/config/ConfigManager.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <llama-cpp.h>

namespace core::llm::providers {

class LlamaCppProvider final : public LLMProvider,
                               public std::enable_shared_from_this<LlamaCppProvider> {
public:
    explicit LlamaCppProvider(const core::config::ProviderConfig& config);

    // Signals any in-flight inference to abort.  The actual thread is detached
    // and keeps the provider alive via shared_from_this() until it exits.
    ~LlamaCppProvider();

    void stream_response(
        const ChatRequest& request,
        std::function<void(const StreamChunk&)> callback) override;

    [[nodiscard]] std::string get_last_model() const override;
    [[nodiscard]] ProviderCapabilities capabilities() const override;
    [[nodiscard]] bool should_estimate_cost() const override;

    // Exposed for testing: builds the JSON Schema object for a tool's parameters.
    [[nodiscard]] static std::string build_tool_parameters_schema(const core::llm::Tool& tool);

private:
    [[nodiscard]] bool ensure_model_loaded(std::string& error);
    [[nodiscard]] std::string effective_model_label(const ChatRequest& request) const;

    std::string model_path_;
    std::string default_model_;
    std::string chat_template_override_;

    int context_size_ = 0;
    int batch_size_ = 0;
    int threads_ = 0;
    int threads_batch_ = 0;
    int gpu_layers_ = 0;
    int top_k_ = 40;
    int seed_ = LLAMA_DEFAULT_SEED;

    float temperature_ = 0.2f;
    float top_p_ = 0.95f;

    bool use_mmap_ = true;
    bool use_mlock_ = false;

    mutable std::mutex state_mutex_;
    llama_model_ptr model_;
    std::string last_model_;

    std::mutex inference_mutex_;

    // Set to true when the provider is being destroyed.  The inference thread
    // checks this flag and exits early so the shared_ptr it holds is released
    // promptly instead of letting a potentially long generation run to completion.
    std::atomic<bool> abort_requested_{false};
};

} // namespace core::llm::providers
