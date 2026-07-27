#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <string_view>

namespace tui::editor {

// A private, self-deleting scratch file holding the prompt draft handed to an
// external editor.
//
// The buffer lives in a fresh owner-only directory so concurrent Filo
// instances (and other users on the machine) can never observe or collide with
// each other's drafts.
class PromptBuffer {
public:
    [[nodiscard]] static std::expected<PromptBuffer, std::string> create(std::string_view text);

    ~PromptBuffer();

    PromptBuffer(PromptBuffer&& other) noexcept;
    PromptBuffer& operator=(PromptBuffer&& other) noexcept;

    PromptBuffer(const PromptBuffer&) = delete;
    PromptBuffer& operator=(const PromptBuffer&) = delete;

    [[nodiscard]] const std::filesystem::path& file() const noexcept { return file_; }

    // Opaque per-session token; backends that speak a wire protocol use it to
    // correlate their messages.
    [[nodiscard]] const std::string& session_id() const noexcept { return session_id_; }

    [[nodiscard]] std::expected<std::string, std::string> read() const;

private:
    PromptBuffer(std::filesystem::path directory, std::filesystem::path file, std::string session_id)
        : directory_(std::move(directory)), file_(std::move(file)), session_id_(std::move(session_id)) {}

    void discard() noexcept;

    std::filesystem::path directory_;
    std::filesystem::path file_;
    std::string session_id_;
};

} // namespace tui::editor
