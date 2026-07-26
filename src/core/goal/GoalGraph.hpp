#pragma once

#include "GoalTypes.hpp"

#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::goal {

// ---------------------------------------------------------------------------
// GoalGraph — a versioned DAG of goal nodes with enforced invariants.
//
// Storage is flat and index-addressed (NodeId == vector index + 1) so ready-set
// computation and topological walks are cache-friendly and allocation-free
// after construction. The topological order is memoized and invalidated on
// structural mutation. All structural invariants (acyclicity, referential
// integrity, unique names) are checked by validate() before a plan may run.
//
// Thread-safety: GoalGraph is NOT internally synchronized. The engine and
// scheduler own it exclusively during execution; parallel wave workers only
// read const copies of nodes.
// ---------------------------------------------------------------------------
class GoalGraph {
public:
    static constexpr std::size_t kMaxNodes = 64;
    static constexpr std::size_t kMaxEdges = 128;
    static constexpr std::size_t kMaxObjectiveChars = 4096;

    /// Append a node. The node's id field is assigned by the graph.
    [[nodiscard]] std::expected<NodeId, std::string> add_node(Node node);

    /// Append an edge between existing nodes. Duplicate edges are rejected.
    [[nodiscard]] std::expected<void, std::string> add_edge(NodeId from,
                                                            NodeId to,
                                                            EdgeCondition condition);

    void set_objective(std::string_view objective);
    [[nodiscard]] std::string_view objective() const noexcept { return objective_; }

    [[nodiscard]] std::optional<NodeId> find(std::string_view name) const;
    [[nodiscard]] const Node* node(NodeId id) const noexcept;
    [[nodiscard]] Node* node(NodeId id) noexcept;
    [[nodiscard]] std::span<const Node> nodes() const noexcept { return nodes_; }
    [[nodiscard]] std::span<const Edge> edges() const noexcept { return edges_; }
    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }
    [[nodiscard]] bool empty() const noexcept { return nodes_.empty(); }

    [[nodiscard]] std::vector<Edge> out_edges(NodeId id) const;
    [[nodiscard]] std::vector<Edge> in_edges(NodeId id) const;

    /// Structural validation: non-empty, unique names, referential integrity,
    /// acyclicity (Kahn), sane retry targets. Must pass before execution.
    [[nodiscard]] std::expected<void, std::string> validate() const;

    /// Nodes eligible to run now: state == Ready, or Pending with all hard
    /// dependencies (Always/OnPass in-edges) succeeded and no conditional
    /// in-edges (those are activated exclusively by the scheduler).
    [[nodiscard]] std::vector<NodeId> ready_nodes() const;

    /// Guarded state transition with a legality table. Illegal transitions
    /// are rejected with an explanatory error instead of corrupting the graph.
    [[nodiscard]] std::expected<void, std::string> transition(NodeId id, NodeState next);

    /// Retry requeue: force a settled node back to Ready. This is the ONLY
    /// sanctioned way to revive a Succeeded node — a downstream verification
    /// failure invalidates an earlier success, so the work must run again.
    /// Kept separate from transition() so ordinary execution can never
    /// resurrect a completed node by accident.
    [[nodiscard]] std::expected<void, std::string> requeue(NodeId id);

    /// True when no node is Pending, Ready, or Running.
    [[nodiscard]] bool is_terminal() const noexcept;

    /// Terminal outcome, if the graph is terminal: Blocked > Failed > Completed.
    [[nodiscard]] std::optional<RunState> outcome() const noexcept;

    /// Plan versioning: bumped on replan so traces stay attributable to the
    /// exact plan that produced them (plans are treated as immutable once
    /// running; a replan creates a new version rather than mutating history).
    [[nodiscard]] int plan_version() const noexcept { return plan_version_; }
    void bump_plan_version() noexcept { ++plan_version_; }
    void set_plan_version(int version) noexcept { plan_version_ = version < 1 ? 1 : version; }

    /// Memoized topological order over all edges (validate() guarantees a DAG).
    [[nodiscard]] const std::vector<NodeId>& topological_order() const;

    /// Compact layered ASCII rendering for `/goal graph`.
    [[nodiscard]] std::string render_ascii() const;

    // -----------------------------------------------------------------------
    // Persistence — hand-written JSON, no external schema dependency.
    // -----------------------------------------------------------------------
    [[nodiscard]] std::string to_json() const;
    [[nodiscard]] static std::expected<GoalGraph, std::string> from_json(std::string_view json);

private:
    [[nodiscard]] bool has_conditional_in_edges(NodeId id) const;
    [[nodiscard]] bool dependencies_succeeded(NodeId id) const;
    void recompute_topological_order() const;

    std::string objective_;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    int plan_version_ = 1;

    mutable std::vector<NodeId> topo_cache_;
    mutable bool topo_dirty_ = true;
};

} // namespace core::goal
