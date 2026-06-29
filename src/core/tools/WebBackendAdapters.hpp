#pragma once

#include "WebAccess.hpp"

#include <memory>

namespace core::tools::web {

[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_kimi_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_anthropic_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_openai_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_zai_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebFetchBackend> make_zai_web_fetch_backend();
[[nodiscard]] std::shared_ptr<IWebFetchBackend> make_direct_web_fetch_backend();

} // namespace core::tools::web
