#include "ApiProtocol.hpp"

#include "../ModelCatalogProvider.hpp"

namespace core::llm::protocols {

std::unique_ptr<core::llm::ModelCatalogProvider>
ApiProtocolBase::model_catalog(std::string_view) const {
    // The default lives out of line: returning a unique_ptr from an inline
    // method instantiates its deleter where ModelCatalogProvider is only
    // forward-declared, which GCC rejects.
    return nullptr;
}

} // namespace core::llm::protocols
