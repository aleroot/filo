#pragma once

#include "core/context/SessionContext.hpp"
#include "core/workspace/Workspace.hpp"

#include <string>

namespace test_support {

inline core::context::SessionContext make_session_context(
    core::workspace::WorkspaceSnapshot snapshot,
    core::context::SessionTransport transport = core::context::SessionTransport::cli,
    std::string session_id = {})
{
    return core::context::make_session_context(
        std::move(snapshot),
        transport,
        std::move(session_id));
}

inline core::context::SessionContext make_workspace_session_context(
    core::context::SessionTransport transport = core::context::SessionTransport::cli,
    std::string session_id = {})
{
    return test_support::make_session_context(
        core::workspace::Workspace::get_instance().snapshot(),
        transport,
        std::move(session_id));
}

} // namespace test_support
