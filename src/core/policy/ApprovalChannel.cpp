#include "ApprovalChannel.hpp"

namespace core::policy {

bool DeferredApprovalChannel::attach(std::shared_ptr<IApprovalChannel> delegate) {
    if (!delegate) return false;
    std::lock_guard lock(mutex_);
    if (delegate_) return false;
    delegate_ = std::move(delegate);
    return true;
}

bool DeferredApprovalChannel::attached() const {
    std::lock_guard lock(mutex_);
    return static_cast<bool>(delegate_);
}

ApprovalOutcome DeferredApprovalChannel::request(const ApprovalRequest& request) {
    std::shared_ptr<IApprovalChannel> delegate;
    {
        std::lock_guard lock(mutex_);
        delegate = delegate_;
    }
    if (!delegate) return ApprovalOutcome::unavailable;
    return delegate->request(request);
}

namespace {

thread_local const ApprovalContext* current_origin = nullptr;

} // namespace

ApprovalOriginScope::ApprovalOriginScope(std::string session_id,
                                         std::function<bool()> stopped)
    : context_(ApprovalContext{.session_id = std::move(session_id),
                               .stopped = std::move(stopped)})
    , previous_(current_origin) {
    current_origin = &context_;
}

ApprovalOriginScope::~ApprovalOriginScope() {
    current_origin = previous_;
}

const ApprovalContext* ApprovalOriginScope::current() noexcept {
    return current_origin;
}

} // namespace core::policy
