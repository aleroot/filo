#pragma once

#include <cpr/cpr.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>

namespace core::llm::protocols {

/**
 * A request a protocol needs the provider to make.
 *
 * The provider owns the transport: it is the only part of Filo that opens a
 * socket to a model endpoint. A protocol that needs the endpoint to do
 * something before the chat request itself (upload a file, count the tokens
 * of a long history) describes that request here, and the provider performs
 * it under the same rules as the chat request.
 *
 * A protocol cannot open a socket of its own, so it cannot send anything the
 * provider has not decided on.
 */
struct ProtocolRequest {
    std::string url;
    cpr::Header headers;
    // The body to send. Ignored when @c multipart is set.
    std::string body;
    // A multipart body, for the endpoints that take a file.
    std::optional<cpr::Multipart> multipart;
    std::chrono::milliseconds timeout{std::chrono::seconds(30)};
    // The connect timeout. Zero leaves the transport's default in place.
    std::chrono::milliseconds connect_timeout{0};
    // The most response bytes to keep. The rest is discarded.
    std::size_t max_response_bytes{256 * 1024};
};

/**
 * The outcome of a request the provider made.
 */
struct ProtocolResponse {
    int status_code{0};
    std::string body;
    // The transport error, if the request did not complete. Empty when it did.
    std::string error;
};

/**
 * Performs a request on a protocol's behalf.
 *
 * Empty when the protocol is used outside a provider. A protocol that needs a
 * request and has none must fail rather than send it by another route.
 */
using ProtocolRequester = std::function<ProtocolResponse(const ProtocolRequest&)>;

} // namespace core::llm::protocols
