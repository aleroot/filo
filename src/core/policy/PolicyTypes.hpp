#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace core::policy {

// The three outcomes a policy decision can take.
//
// "ask" is a request for a decision, never a permission. Only an allow
// authorizes an operation, and only when the trusted code performing it
// applies the decision.
enum class Effect { allow, deny, ask };

// The result of evaluating a request against a policy.
//
// A decision is advisory until the operation's trusted enforcement point
// applies it. An "ask" is never itself permission to execute.
struct Decision {
    Effect effect{Effect::deny};
    std::string reason;
    // Identifies the rule that decided, when that rule has a label. Empty
    // when the policy default decided, or when the matching rule is unnamed.
    std::string rule_id;
    // True when a rule decided, including a rule with no label. False when
    // the policy default decided. Callers that treat a configured destination
    // as the default must use this, not an empty rule_id: an unnamed deny is
    // still a deny.
    bool matched_rule{false};
};

// Who is asking for a resource.
//
// Supplied by Filo's trusted adapters, never by model arguments or by remote
// MCP annotations. Adding a kind here is the extension point for a new trust
// boundary; the matching machinery does not change.
enum class PrincipalKind {
    web_tool,       // a built-in tool that reaches the network
    child_process,  // a confined process tree Filo launched
    mcp_server,     // a configured MCP server
    model_provider, // remote inference
};

// What kind of operation is being performed.
//
// This describes the shape of the operation, not who performs it.
enum class OperationKind {
    read,    // fetch a resource
    write,   // send or modify a resource
    connect, // open a connection
    invoke,  // call a remote procedure
    infer,   // remote inference
};

// Which way data flows relative to the destination.
//
// This is what makes "block uploads" expressible: a rule can require
// DataFlow::inbound and refuse DataFlow::outbound. It is only meaningful when
// the enforcement point genuinely knows the direction. An operation that
// cannot be characterised must use DataFlow::bidirectional, never a
// convenient value, because the grant shown to the user must not be broader
// than the enforcement the user is relying on.
enum class DataFlow { none, inbound, outbound, bidirectional };

// Who performs an operation.
struct Subject {
    PrincipalKind kind{PrincipalKind::web_tool};
    // Identifies the principal within its kind: a tool name, an MCP server
    // name, a provider name. "*" matches any identifier.
    std::string id{"*"};

    [[nodiscard]] friend bool operator==(const Subject&, const Subject&) = default;
};

// What is being done.
struct Action {
    OperationKind operation{OperationKind::read};
    DataFlow flow{DataFlow::none};

    [[nodiscard]] friend bool operator==(const Action&, const Action&) = default;
};

// Which resource is being reached.
struct Resource {
    // Canonical lower-case host or IP literal, without brackets and without a
    // trailing dot. Parsing, name resolution and address binding belong to
    // the transport adapter, not to the policy.
    std::string host;
    std::uint16_t port{0};

    [[nodiscard]] friend bool operator==(const Resource&, const Resource&) = default;
};

// A single request for a resource: who, what, where.
struct NetworkIntent {
    Subject subject;
    Action action;
    Resource resource;
};

// The name of each value, for decisions and diagnostics.
//
// They are the same words the policy documents use, so a decision can be
// quoted without translating it.
[[nodiscard]] constexpr std::string_view to_string(Effect effect) noexcept {
    switch (effect) {
        case Effect::allow: return "allow";
        case Effect::deny: return "deny";
        case Effect::ask: return "ask";
    }
    std::unreachable();
}

[[nodiscard]] constexpr std::string_view to_string(PrincipalKind kind) noexcept {
    switch (kind) {
        case PrincipalKind::web_tool: return "web_tool";
        case PrincipalKind::child_process: return "child_process";
        case PrincipalKind::mcp_server: return "mcp_server";
        case PrincipalKind::model_provider: return "model_provider";
    }
    std::unreachable();
}

[[nodiscard]] constexpr std::string_view to_string(OperationKind kind) noexcept {
    switch (kind) {
        case OperationKind::read: return "read";
        case OperationKind::write: return "write";
        case OperationKind::connect: return "connect";
        case OperationKind::invoke: return "invoke";
        case OperationKind::infer: return "infer";
    }
    std::unreachable();
}

[[nodiscard]] constexpr std::string_view to_string(DataFlow flow) noexcept {
    switch (flow) {
        case DataFlow::none: return "none";
        case DataFlow::inbound: return "inbound";
        case DataFlow::outbound: return "outbound";
        case DataFlow::bidirectional: return "bidirectional";
    }
    std::unreachable();
}

} // namespace core::policy
