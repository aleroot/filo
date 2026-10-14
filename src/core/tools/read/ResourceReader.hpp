#pragma once
#include "ReadTypes.hpp"
#include "../WebAccess.hpp"
#include "../../agent/ToolResultStore.hpp"

namespace core::tools::read {
/// A batch entry that could not be read, and why.
struct UnreadPath {
    std::string path;
    std::string error;
};
/// The outcome of reading every path of a call.
struct Batch {
    std::vector<Resource> sources;
    /// The requested path of each source, index for index.
    std::vector<std::string> paths;
    std::vector<UnreadPath> unread;
};
class ResourceReader {
public:
    explicit ResourceReader(core::agent::ToolResultStore store = core::agent::ToolResultStore{}, web::WebAccess* web = nullptr)
        : store_(std::move(store)), web_(web) {}
    /// Reads every path of @p options. A multi-path read without a question
    /// returns the readable part with the rest in Batch::unread and fails
    /// only when nothing was readable; otherwise the first error fails it.
    [[nodiscard]] std::expected<Batch, std::string> read_all(
        const Options& options, const ToolInvocationContext& invocation) const;
    [[nodiscard]] std::expected<Resource, std::string> read(
        const std::string& path, const Options& options, const ToolInvocationContext& invocation) const;
private:
    core::agent::ToolResultStore store_;
    web::WebAccess* web_;
};
} // namespace core::tools::read
