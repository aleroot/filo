#include "ToolResultStore.hpp"

#include "../session/SessionStore.hpp"
#include "../utils/JsonWriter.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>

namespace core::agent {
namespace {

std::atomic<std::uint64_t> temporary_sequence{0};

[[nodiscard]] std::uint64_t fnv1a64(std::string_view text) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] bool is_utf8_continuation(unsigned char byte) noexcept {
    return (byte & 0xc0U) == 0x80U;
}

[[nodiscard]] std::size_t first_non_space_after(
    std::string_view text,
    std::size_t position) noexcept {
    while (position < text.size()
           && std::isspace(static_cast<unsigned char>(text[position]))) {
        ++position;
    }
    return position;
}

} // namespace

ToolResultStore::ToolResultStore(std::filesystem::path root)
    : root_(std::move(root)) {}

std::filesystem::path ToolResultStore::default_root() {
    return core::session::SessionStore::default_sessions_dir().parent_path()
        / "tool-results";
}

std::expected<StoredToolResult, std::string> ToolResultStore::store(
    std::string_view session_id,
    std::string_view tool_call_id,
    std::string_view content) const {
    const std::string session = sanitize_component(session_id, "unscoped");
    const std::string call = sanitize_component(tool_call_id, "unknown-call");
    const std::string digest = std::format("{:016x}", fnv1a64(content));
    const std::string filename = std::format("{}-{}.result", call, digest);
    const std::filesystem::path directory = root_ / session;
    const std::filesystem::path destination = directory / filename;

    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        return std::unexpected(std::format(
            "Cannot create tool-result directory '{}': {}",
            directory.string(),
            ec.message()));
    }
    std::filesystem::permissions(
        directory,
        std::filesystem::perms::owner_all,
        std::filesystem::perm_options::replace,
        ec);
    if (ec) {
        return std::unexpected(std::format(
            "Cannot secure tool-result directory '{}': {}",
            directory.string(),
            ec.message()));
    }

    const bool destination_exists = std::filesystem::exists(destination, ec);
    if (ec) {
        return std::unexpected(std::format(
            "Cannot inspect tool-result destination '{}': {}",
            destination.string(),
            ec.message()));
    }
    if (!destination_exists) {
        const auto sequence = temporary_sequence.fetch_add(1, std::memory_order_relaxed);
        const std::filesystem::path temporary = std::filesystem::path{
            destination.string() + std::format(".tmp-{}", sequence)};
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                return std::unexpected(std::format(
                    "Cannot open temporary tool-result file '{}'.",
                    temporary.string()));
            }
            output.write(content.data(), static_cast<std::streamsize>(content.size()));
            output.flush();
            if (!output) {
                std::filesystem::remove(temporary, ec);
                return std::unexpected(std::format(
                    "Cannot write temporary tool-result file '{}'.",
                    temporary.string()));
            }
        }
        std::filesystem::permissions(
            temporary,
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace,
            ec);
        if (ec) {
            const std::string message = ec.message();
            std::error_code remove_error;
            std::filesystem::remove(temporary, remove_error);
            return std::unexpected(std::format(
                "Cannot secure temporary tool-result file '{}': {}",
                temporary.string(),
                message));
        }
        std::filesystem::rename(temporary, destination, ec);
        if (ec) {
            std::error_code exists_error;
            if (std::filesystem::exists(destination, exists_error) && !exists_error) {
                std::filesystem::remove(temporary, ec);
            } else {
                const std::string message = ec.message();
                std::filesystem::remove(temporary, ec);
                return std::unexpected(std::format(
                    "Cannot publish tool-result file '{}': {}",
                    destination.string(),
                    message));
            }
        }
    }

    return StoredToolResult{
        .reference = session + "/" + filename,
        .digest = digest,
        .original_bytes = content.size(),
    };
}

std::expected<ToolResultChunk, std::string> ToolResultStore::read(
    std::string_view session_id,
    std::string_view reference,
    std::uint64_t offset,
    std::size_t limit) const {
    if (!valid_reference(reference)) {
        return std::unexpected("Invalid tool-result reference.");
    }
    const std::string expected_prefix = sanitize_component(session_id, "unscoped") + "/";
    if (!reference.starts_with(expected_prefix)) {
        return std::unexpected("Tool result does not belong to this session.");
    }

    const std::filesystem::path path = root_ / std::filesystem::path{reference};
    std::error_code ec;
    const std::uint64_t total = std::filesystem::file_size(path, ec);
    if (ec) {
        return std::unexpected("Tool result was not found.");
    }
    if (offset > total) {
        return std::unexpected(std::format(
            "Offset {} exceeds tool-result size {}.", offset, total));
    }

    limit = std::clamp<std::size_t>(limit, 4, kMaxReadChars);
    if (offset == total) {
        return ToolResultChunk{
            .offset = offset,
            .next_offset = offset,
            .total_bytes = total,
            .complete = true,
        };
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) return std::unexpected("Tool result could not be opened.");
    input.seekg(static_cast<std::streamoff>(offset));

    const std::uint64_t available = total - offset;
    const std::size_t requested = static_cast<std::size_t>(
        std::min<std::uint64_t>(available, limit + 4));
    std::string buffer(requested, '\0');
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    buffer.resize(static_cast<std::size_t>(input.gcount()));

    std::size_t prefix = 0;
    while (prefix < buffer.size()
           && is_utf8_continuation(static_cast<unsigned char>(buffer[prefix]))) {
        ++prefix;
    }
    const std::uint64_t adjusted_offset = offset + prefix;
    std::size_t kept = std::min(limit, buffer.size() - prefix);
    if (adjusted_offset + kept < total) {
        while (kept > 0 && prefix + kept < buffer.size()
               && is_utf8_continuation(
                   static_cast<unsigned char>(buffer[prefix + kept]))) {
            --kept;
        }
    }

    const std::uint64_t next = adjusted_offset + kept;
    return ToolResultChunk{
        .content = buffer.substr(prefix, kept),
        .offset = adjusted_offset,
        .next_offset = next,
        .total_bytes = total,
        .complete = next >= total,
    };
}

std::string ToolResultStore::attach_reference(
    std::string compact_payload,
    const StoredToolResult& stored) {
    core::utils::JsonWriter annotation(192 + stored.reference.size());
    {
        auto object = annotation.object();
        annotation.kv_str("reference", stored.reference).comma()
                  .kv_str("reader", "read_tool_result").comma()
                  .kv_num("original_bytes", static_cast<std::int64_t>(stored.original_bytes)).comma()
                  .kv_str("digest_fnv1a64", stored.digest);
    }
    std::string annotation_json = std::move(annotation).take();

    const std::size_t open = first_non_space_after(compact_payload, 0);
    const std::size_t close = compact_payload.find_last_not_of(" \t\r\n");
    if (open < compact_payload.size() && compact_payload[open] == '{'
        && close != std::string::npos && compact_payload[close] == '}') {
        const std::size_t first_value = first_non_space_after(compact_payload, open + 1);
        std::string fragment;
        fragment.reserve(annotation_json.size() + 12);
        if (first_value < close) fragment += ',';
        fragment += R"("offload":)";
        fragment += annotation_json;
        compact_payload.insert(close, fragment);
        return compact_payload;
    }

    core::utils::JsonWriter fallback(256 + compact_payload.size() + annotation_json.size());
    {
        auto object = fallback.object();
        fallback.kv_bool("offloaded", true).comma()
                .kv_str("reference", stored.reference).comma()
                .kv_str("reader", "read_tool_result").comma()
                .kv_str("preview", compact_payload);
    }
    return std::move(fallback).take();
}

std::string ToolResultStore::sanitize_component(
    std::string_view value,
    std::string_view fallback) {
    std::string sanitized;
    sanitized.reserve(std::min<std::size_t>(value.size(), 80));
    for (const unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
            sanitized.push_back(static_cast<char>(ch));
        } else {
            sanitized.push_back('_');
        }
        if (sanitized.size() == 80) break;
    }
    if (sanitized.empty() || sanitized == "." || sanitized == "..") {
        sanitized = fallback;
    }
    return sanitized;
}

bool ToolResultStore::valid_reference(std::string_view reference) noexcept {
    if (reference.empty() || reference.size() > 192
        || reference.front() == '/' || reference.back() == '/') {
        return false;
    }
    std::size_t separators = 0;
    std::size_t component_size = 0;
    for (const unsigned char ch : reference) {
        if (ch == '/') {
            if (component_size == 0 || ++separators > 1) return false;
            component_size = 0;
            continue;
        }
        if (!(std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.')) return false;
        ++component_size;
    }
    return separators == 1 && component_size > 0
        && !reference.contains("../") && !reference.contains("/..");
}

} // namespace core::agent
