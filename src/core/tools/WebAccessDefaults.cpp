#include "WebAccess.hpp"

#include "WebBackendAdapters.hpp"

namespace core::tools::web {

WebAccess make_default_web_access() {
    return WebAccess(
        {
            make_zai_web_search_backend(),
            make_kimi_web_search_backend(),
            make_anthropic_web_search_backend(),
            make_openai_web_search_backend(),
        },
        {
            make_zai_web_fetch_backend(),
            make_direct_web_fetch_backend(),
        });
}

} // namespace core::tools::web
