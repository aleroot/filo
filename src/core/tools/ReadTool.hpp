#pragma once
#include "Tool.hpp"
#include "read/ResourceReader.hpp"
#include "read/ReaderWorker.hpp"

namespace core::tools {
// Facade only: acquisition, decoding, presentation and model execution are
// independently testable. The ordinary text path retains its legacy contract.
class ReadTool final : public Tool {
public:
    ReadTool() = default;
    ReadTool(read::ResourceReader resources, read::ReaderWorker worker)
        : resources_(std::move(resources)), worker_(std::move(worker)) {}
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
    std::string execute(const std::string& json_args, const ToolInvocationContext& invocation) override;
private:
    read::ResourceReader resources_;
    read::ReaderWorker worker_;
};
} // namespace core::tools
