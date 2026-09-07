#include "ResourceReader.hpp"
#include "../ToolArgumentUtils.hpp"
#include "../ToolPolicy.hpp"
#include <fstream>
#include <algorithm>
#include <cstdint>

namespace core::tools::read {
std::expected<Resource, std::string> ResourceReader::read(
    const std::string& path, const Options& options, const ToolInvocationContext& invocation) const {
    if (invocation.cancellation_requested && invocation.cancellation_requested())
        return std::unexpected("Read cancelled.");
    Resource resource{.uri = path};
    const bool http = path.starts_with("http://") || path.starts_with("https://");
    if (http) {
        if (auto error = policy::enforce_url_policy(names::kRead, path)) return std::unexpected(*error);
        // Both initial URL and redirects must clear read's policy as well as
        // the existing fetch policy. The transport receives the operation name.
        auto fetched = (web_ ? *web_ : web::WebAccess::instance()).fetch(
            {.url = path, .policy_tool = std::string(names::kRead), .preserve_bytes = true},
            invocation);
        if (!fetched) return std::unexpected(fetched.error());
        resource.uri = fetched->final_url;
        resource.kind = "web";
        resource.text = std::move(fetched->text);
        resource.truncated = fetched->truncated;
    } else if (path.starts_with("result://")) {
        if (invocation.session_context.session_id.empty())
            return std::unexpected("Stored results require a scoped session.");
        uint64_t offset = 0;
        do {
            auto chunk = store_.read(invocation.session_context.session_id, path.substr(9), offset);
            if (!chunk) return std::unexpected(chunk.error());
            const auto remaining = kMaxSourceBytes - resource.text.size();
            resource.text.append(chunk->content, 0, remaining);
            resource.truncated = !chunk->complete || chunk->content.size() > remaining;
            if (chunk->complete || resource.text.size() == kMaxSourceBytes) break;
            // A chunk that does not advance would otherwise spin forever on a
            // truncated or concurrently rewritten store entry.
            if (chunk->next_offset <= offset) {
                resource.truncated = true;
                break;
            }
            offset = chunk->next_offset;
        } while (true);
        resource.kind = "result";
    } else {
        if (path.contains("://")) return std::unexpected("Unsupported resource scheme. Available: local paths, HTTP(S), result://.");
        std::filesystem::path resolved;
        const auto& context = invocation.session_context;
        if (auto error = detail::check_workspace_access(path, path, context, &resolved, names::kRead))
            return std::unexpected(*error);
        resource.uri = resolved.string();
        std::error_code ec;
        const auto status = std::filesystem::status(resolved, ec);
        if (ec || !std::filesystem::exists(status)) return std::unexpected("Cannot read path: path does not exist.");
        if (std::filesystem::is_directory(status)) {
            resource.kind = "directory";
            std::vector<std::string> entries;
            std::filesystem::directory_iterator iterator(resolved, ec), end;
            if (ec) return std::unexpected("Cannot list directory: " + ec.message());
            std::size_t visited = 0;
            for (; iterator != end; iterator.increment(ec)) {
                if (ec) return std::unexpected("Cannot complete directory listing: " + ec.message());
                if (++visited > 8192) { resource.truncated = true; break; }
                if (invocation.cancellation_requested && invocation.cancellation_requested())
                    return std::unexpected("Read cancelled.");
                const auto& entry = *iterator;
                if (detail::check_workspace_access(entry.path(), entry.path().string(), context, nullptr, names::kRead)) continue;
                std::error_code type_ec;
                const bool directory = entry.is_directory(type_ec);
                entries.push_back(entry.path().filename().string() + (directory ? "/\n" : "\n"));
            }
            if (ec) return std::unexpected("Cannot complete directory listing: " + ec.message());
            std::ranges::sort(entries);
            for (const auto& entry : entries) {
                if (resource.text.size() + entry.size() > kMaxSourceBytes) { resource.truncated = true; break; }
                resource.text += entry;
            }
        } else if (std::filesystem::is_regular_file(status)) {
            std::ifstream input(resolved, std::ios::binary);
            if (!input) return std::unexpected("Cannot open resource.");
            resource.text = read_prefix(input, kMaxSourceBytes + 1);
            if (input.bad()) return std::unexpected("Resource read failed.");
            resource.truncated = resource.text.size() > kMaxSourceBytes;
        } else return std::unexpected("Resource is not a regular file or directory.");
    }
    if (resource.text.size() > kMaxSourceBytes) resource.text.resize(kMaxSourceBytes);
    const std::string source_digest = digest(resource.text);
    const auto& resolved_uri = resource.uri;
    const auto ext = std::filesystem::path(http
        ? resolved_uri.substr(0, resolved_uri.find_first_of("?#")) : resolved_uri).extension().string();
    std::expected<Resource, std::string> decoded = std::move(resource);
    if (ext == ".ipynb" && decoded->kind != "directory") decoded = decode_notebook(std::move(*decoded), options);
    else if (!options.cell.empty())
        return std::unexpected("select.cell applies only to notebooks.");
    if (!decoded) return decoded;
    if (decoded->text.find('\0') != std::string::npos)
        return std::unexpected("Binary resource has no supported text decoder.");
    decoded->digest = source_digest;
    if (!options.expected_digest.empty() && options.expected_digest != decoded->digest)
        return std::unexpected("Resource changed since the referenced read. Read it again before using old locations.");
    return decoded;
}
} // namespace core::tools::read
