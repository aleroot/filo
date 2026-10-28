#include "ProviderEgress.hpp"

#include "../../policy/PolicySnapshot.hpp"

#include <cpr/cpr.h>

#include <cstdint>
#include <string>

namespace core::llm::transport {

ProviderEgress::ProviderEgress(std::string_view provider_name,
                               std::string_view base_url,
                               const std::string& session_id,
                               const std::atomic_bool& cancelled)
    : scope_(core::policy::active_policy(),
             core::policy::Subject{.kind = core::policy::PrincipalKind::model_provider,
                                   .id = std::string(provider_name)},
             // The provider is sent the conversation to run inference on it.
             core::policy::Action{.operation = core::policy::OperationKind::infer,
                                  .flow = core::policy::DataFlow::outbound},
             base_url)
    , cancelled_(cancelled) {
    if (!scope_.enforcing()) return;
    // The question belongs to the chat that sent the request, unless the
    // caller, a tool call for instance, has already named one.
    if (!session_id.empty() && core::policy::ApprovalOriginScope::current() == nullptr) {
        origin_.emplace(session_id,
                        [&cancelled] { return cancelled.load(std::memory_order_acquire); });
    }
    refusal_ = scope_.authorize(base_url);
}

protocols::ProtocolResponse ProviderEgress::post(
    const protocols::ProtocolRequest& request) {
    protocols::ProtocolResponse response;
    cpr::Session session;
    session.SetUrl(cpr::Url{request.url});
    session.SetHeader(request.headers);
    if (request.multipart) {
        session.SetMultipart(*request.multipart);
    } else {
        session.SetBody(cpr::Body{request.body});
    }
    session.SetTimeout(cpr::Timeout{static_cast<int32_t>(request.timeout.count())});
    if (request.connect_timeout.count() > 0) {
        session.SetConnectTimeout(
            cpr::ConnectTimeout{static_cast<int32_t>(request.connect_timeout.count())});
    }
    // The response is read into a bounded buffer: a protocol's request must
    // not be able to pull an unbounded body into the process.
    std::string body;
    const std::size_t max_bytes = request.max_response_bytes;
    session.SetWriteCallback(cpr::WriteCallback{[&](std::string_view bytes, intptr_t) {
        if (bytes.size() > max_bytes - body.size()) return false;
        body.append(bytes);
        return true;
    }});
    session.SetProgressCallback(cpr::ProgressCallback{[&](cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t,
        cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t, intptr_t) {
        return !cancelled_.load(std::memory_order_acquire);
    }});

    if (const auto refusal = scope_.admit(session)) {
        response.error = *refusal;
        return response;
    }
    const auto performed = session.Post();
    response.status_code = performed.status_code;
    response.body = std::move(body);
    if (performed.error) response.error = performed.error.message;
    return response;
}

} // namespace core::llm::transport
