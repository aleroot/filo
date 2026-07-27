#include "PromptBuffer.hpp"

#include "core/session/SessionStore.hpp"

#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>

namespace tui::editor {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kDirectoryPrefix = "filo-prompt-edit-";
constexpr std::string_view kBufferFileName = "prompt.md";
constexpr int kMaxDirectoryAttempts = 4;

// Two session ids produce the 16 hexadecimal characters required by detached
// editor protocols. Owner-only permissions keep the scratch directory private.
[[nodiscard]] std::string make_session_id() {
    return core::session::SessionStore::generate_id() + core::session::SessionStore::generate_id();
}

} // namespace

std::expected<PromptBuffer, std::string> PromptBuffer::create(std::string_view text) {
    std::error_code ec;
    const fs::path temp_root = fs::temp_directory_path(ec);
    if (ec) {
        return std::unexpected("Could not locate a temporary directory: " + ec.message());
    }

    for (int attempt = 0; attempt < kMaxDirectoryAttempts; ++attempt) {
        std::string session_id = make_session_id();
        const fs::path directory = temp_root / (std::string(kDirectoryPrefix) + session_id);

        if (!fs::create_directory(directory, ec)) {
            // Either the name collided or creation failed; both are worth one
            // more roll of the dice before giving up.
            ec.clear();
            continue;
        }

        fs::permissions(directory, fs::perms::owner_all, fs::perm_options::replace, ec);
        if (ec) {
            fs::remove_all(directory, ec);
            return std::unexpected("Could not secure the temporary editor directory.");
        }

        const fs::path file = directory / kBufferFileName;
        {
            std::ofstream output(file, std::ios::binary | std::ios::trunc);
            if (!output) {
                fs::remove_all(directory, ec);
                return std::unexpected("Could not write the temporary editor buffer.");
            }
            output << text;
            if (!output) {
                fs::remove_all(directory, ec);
                return std::unexpected("Could not write the temporary editor buffer.");
            }
        }

        fs::permissions(
            file,
            fs::perms::owner_read | fs::perms::owner_write,
            fs::perm_options::replace,
            ec);
        if (ec) {
            fs::remove_all(directory, ec);
            return std::unexpected("Could not secure the temporary editor buffer.");
        }

        return PromptBuffer(directory, file, std::move(session_id));
    }

    return std::unexpected("Could not allocate a unique external editor session.");
}

PromptBuffer::~PromptBuffer() {
    discard();
}

PromptBuffer::PromptBuffer(PromptBuffer&& other) noexcept
    : directory_(std::move(other.directory_)),
      file_(std::move(other.file_)),
      session_id_(std::move(other.session_id_)) {
    other.directory_.clear();
    other.file_.clear();
}

PromptBuffer& PromptBuffer::operator=(PromptBuffer&& other) noexcept {
    if (this != &other) {
        discard();
        directory_ = std::move(other.directory_);
        file_ = std::move(other.file_);
        session_id_ = std::move(other.session_id_);
        other.directory_.clear();
        other.file_.clear();
    }
    return *this;
}

std::expected<std::string, std::string> PromptBuffer::read() const {
    std::ifstream input(file_, std::ios::binary);
    if (!input) {
        return std::unexpected("Could not reload the edited prompt.");
    }
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void PromptBuffer::discard() noexcept {
    if (directory_.empty()) {
        return;
    }
    std::error_code ec;
    fs::remove_all(directory_, ec);
    directory_.clear();
    file_.clear();
}

} // namespace tui::editor
