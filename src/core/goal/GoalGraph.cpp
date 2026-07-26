#include "GoalGraph.hpp"

#include "core/utils/JsonUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <format>
#include <simdjson.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace core::goal {

namespace {

constexpr std::string_view kTruncationMarker = "... [truncated]";

[[nodiscard]] std::string clamp_copy(std::string_view value, std::size_t max_chars) {
    std::string normalized(value);
    if (normalized.size() <= max_chars) {
        return normalized;
    }
    if (max_chars <= kTruncationMarker.size()) {
        normalized.resize(max_chars);
        return normalized;
    }
    normalized.resize(max_chars - kTruncationMarker.size());
    normalized += kTruncationMarker;
    return normalized;
}

[[nodiscard]] int clamp_attempts(int value) noexcept {
    return std::clamp(value, 1, Node::kMaxAttempts);
}

[[nodiscard]] std::string_view state_icon(NodeState state) noexcept {
    switch (state) {
    case NodeState::Succeeded: return "[x]";
    case NodeState::Failed:    return "[!]";
    case NodeState::Running:   return "[>]";
    case NodeState::Ready:     return "[o]";
    case NodeState::Blocked:   return "[#]";
    case NodeState::Skipped:   return "[-]";
    case NodeState::Pending:   return "[ ]";
    }
    return "[ ]";
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

std::expected<NodeId, std::string> GoalGraph::add_node(Node node) {
    if (nodes_.size() >= kMaxNodes) {
        return std::unexpected(std::format("goal graph is full (max {} nodes)", kMaxNodes));
    }
    node.name = clamp_copy(node.name, Node::kMaxNameChars);
    if (node.name.empty()) {
        return std::unexpected("goal node requires a name");
    }
    if (find(node.name).has_value()) {
        return std::unexpected(std::format("duplicate goal node name '{}'", node.name));
    }
    node.directive = clamp_copy(node.directive, Node::kMaxDirectiveChars);
    node.check_command = clamp_copy(node.check_command, Node::kMaxCheckChars);
    node.acceptance = clamp_copy(node.acceptance, Node::kMaxAcceptanceChars);
    node.result_summary = clamp_copy(node.result_summary, Node::kMaxResultChars);
    node.max_attempts = clamp_attempts(node.max_attempts);
    if (node.lessons.size() > Node::kMaxLessons) {
        node.lessons.resize(Node::kMaxLessons);
    }
    for (auto& lesson : node.lessons) {
        lesson = clamp_copy(lesson, Node::kMaxLessonChars);
    }

    const NodeId id = static_cast<NodeId>(nodes_.size()) + 1;
    node.id = id;
    nodes_.push_back(std::move(node));
    return id;
}

std::expected<void, std::string> GoalGraph::add_edge(NodeId from,
                                                     NodeId to,
                                                     EdgeCondition condition) {
    if (node(from) == nullptr || node(to) == nullptr) {
        return std::unexpected("goal edge references an unknown node");
    }
    if (from == to) {
        return std::unexpected("goal edge cannot be a self-loop");
    }
    if (edges_.size() >= kMaxEdges) {
        return std::unexpected(std::format("goal graph is full (max {} edges)", kMaxEdges));
    }
    const auto duplicate = std::ranges::any_of(edges_, [&](const Edge& edge) {
        return edge.from == from && edge.to == to && edge.condition == condition;
    });
    if (duplicate) {
        return std::unexpected("duplicate goal edge");
    }
    edges_.push_back(Edge{.from = from, .to = to, .condition = condition});
    topo_dirty_ = true;
    return {};
}

void GoalGraph::set_objective(std::string_view objective) {
    objective_ = clamp_copy(objective, kMaxObjectiveChars);
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

std::optional<NodeId> GoalGraph::find(std::string_view name) const {
    for (const Node& node : nodes_) {
        if (node.name == name) {
            return node.id;
        }
    }
    return std::nullopt;
}

const Node* GoalGraph::node(NodeId id) const noexcept {
    if (id == kInvalidNodeId || id > nodes_.size()) {
        return nullptr;
    }
    return &nodes_[id - 1];
}

Node* GoalGraph::node(NodeId id) noexcept {
    if (id == kInvalidNodeId || id > nodes_.size()) {
        return nullptr;
    }
    return &nodes_[id - 1];
}

std::vector<Edge> GoalGraph::out_edges(NodeId id) const {
    std::vector<Edge> out;
    for (const Edge& edge : edges_) {
        if (edge.from == id) {
            out.push_back(edge);
        }
    }
    return out;
}

std::vector<Edge> GoalGraph::in_edges(NodeId id) const {
    std::vector<Edge> out;
    for (const Edge& edge : edges_) {
        if (edge.to == id) {
            out.push_back(edge);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Validation and readiness
// ---------------------------------------------------------------------------

std::expected<void, std::string> GoalGraph::validate() const {
    if (nodes_.empty()) {
        return std::unexpected("goal graph has no nodes");
    }

    std::unordered_set<std::string_view> names;
    for (const Node& node : nodes_) {
        if (node.name.empty()) {
            return std::unexpected(std::format("goal node #{} has no name", node.id));
        }
        if (!names.insert(node.name).second) {
            return std::unexpected(std::format("duplicate goal node name '{}'", node.name));
        }
        if (node.retry_target != kInvalidNodeId) {
            const Node* target = this->node(node.retry_target);
            if (target == nullptr) {
                return std::unexpected(std::format(
                    "goal node '{}' references an unknown retry target", node.name));
            }
            if (target->kind != NodeKind::Work) {
                return std::unexpected(std::format(
                    "goal node '{}' retry target '{}' must be a work node",
                    node.name, target->name));
            }
        }
    }

    // Kahn's algorithm over all edges — conditional back-routes would create
    // cycles, so rejecting cycles here keeps the whole design acyclic.
    std::unordered_map<NodeId, std::size_t> in_degree;
    in_degree.reserve(nodes_.size());
    for (const Node& node : nodes_) {
        in_degree.emplace(node.id, 0);
    }
    for (const Edge& edge : edges_) {
        if (!in_degree.contains(edge.from) || !in_degree.contains(edge.to)) {
            return std::unexpected("goal edge references an unknown node");
        }
        ++in_degree[edge.to];
    }

    std::deque<NodeId> queue;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            queue.push_back(id);
        }
    }
    std::size_t visited = 0;
    while (!queue.empty()) {
        const NodeId id = queue.front();
        queue.pop_front();
        ++visited;
        for (const Edge& edge : edges_) {
            if (edge.from != id) {
                continue;
            }
            if (--in_degree[edge.to] == 0) {
                queue.push_back(edge.to);
            }
        }
    }
    if (visited != nodes_.size()) {
        return std::unexpected(
            "goal graph contains a cycle; retry loops must use attempt budgets, not edges");
    }
    return {};
}

bool GoalGraph::has_conditional_in_edges(NodeId id) const {
    return std::ranges::any_of(edges_, [&](const Edge& edge) {
        return edge.to == id && !is_dependency(edge.condition);
    });
}

bool GoalGraph::dependencies_succeeded(NodeId id) const {
    return std::ranges::all_of(edges_, [&](const Edge& edge) {
        if (edge.to != id || !is_dependency(edge.condition)) {
            return true;
        }
        const Node* dependency = node(edge.from);
        return dependency != nullptr && dependency->state == NodeState::Succeeded;
    });
}

std::vector<NodeId> GoalGraph::ready_nodes() const {
    std::vector<NodeId> ready;
    for (const Node& node : nodes_) {
        if (node.state == NodeState::Ready) {
            ready.push_back(node.id);
            continue;
        }
        if (node.state != NodeState::Pending) {
            continue;
        }
        if (has_conditional_in_edges(node.id)) {
            continue; // scheduler-activated only
        }
        if (dependencies_succeeded(node.id)) {
            ready.push_back(node.id);
        }
    }
    return ready;
}

std::expected<void, std::string> GoalGraph::transition(NodeId id, NodeState next) {
    Node* target = node(id);
    if (target == nullptr) {
        return std::unexpected("goal transition references an unknown node");
    }
    const NodeState from = target->state;
    const auto legal = [](NodeState a, NodeState b) noexcept {
        switch (a) {
        case NodeState::Pending:
            return b == NodeState::Ready || b == NodeState::Running
                || b == NodeState::Skipped;
        case NodeState::Ready:
            return b == NodeState::Running || b == NodeState::Pending
                || b == NodeState::Skipped;
        case NodeState::Running:
            return b == NodeState::Succeeded || b == NodeState::Failed
                || b == NodeState::Blocked || b == NodeState::Ready;
        case NodeState::Failed:
            return b == NodeState::Ready || b == NodeState::Skipped;
        case NodeState::Blocked:
            return b == NodeState::Ready;
        case NodeState::Succeeded:
            return false;
        case NodeState::Skipped:
            return b == NodeState::Pending;
        }
        return false;
    };
    if (!legal(from, next)) {
        return std::unexpected(std::format(
            "illegal goal node transition {} -> {} for '{}'",
            to_string(from), to_string(next), target->name));
    }
    target->state = next;
    return {};
}

std::expected<void, std::string> GoalGraph::requeue(NodeId id) {
    Node* target = node(id);
    if (target == nullptr) {
        return std::unexpected("goal requeue references an unknown node");
    }
    if (target->state == NodeState::Running) {
        return std::unexpected(std::format(
            "cannot requeue '{}' while it is running", target->name));
    }
    target->state = NodeState::Ready;
    return {};
}

bool GoalGraph::is_terminal() const noexcept {
    return std::ranges::all_of(nodes_, [](const Node& node) {
        return core::goal::is_terminal(node.state);
    });
}

std::optional<RunState> GoalGraph::outcome() const noexcept {
    if (!is_terminal()) {
        return std::nullopt;
    }
    const auto any = [&](NodeState state) {
        return std::ranges::any_of(nodes_, [&](const Node& node) {
            return node.state == state;
        });
    };
    if (any(NodeState::Blocked)) return RunState::Blocked;
    if (any(NodeState::Failed)) return RunState::Failed;
    return RunState::Completed;
}

const std::vector<NodeId>& GoalGraph::topological_order() const {
    if (topo_dirty_) {
        recompute_topological_order();
        topo_dirty_ = false;
    }
    return topo_cache_;
}

void GoalGraph::recompute_topological_order() const {
    topo_cache_.clear();
    if (nodes_.empty()) {
        return;
    }

    std::vector<std::size_t> in_degree(nodes_.size() + 1, 0);
    for (const Edge& edge : edges_) {
        ++in_degree[edge.to];
    }
    std::deque<NodeId> queue;
    for (const Node& node : nodes_) {
        if (in_degree[node.id] == 0) {
            queue.push_back(node.id);
        }
    }
    topo_cache_.reserve(nodes_.size());
    while (!queue.empty()) {
        const NodeId id = queue.front();
        queue.pop_front();
        topo_cache_.push_back(id);
        for (const Edge& edge : edges_) {
            if (edge.from == id && --in_degree[edge.to] == 0) {
                queue.push_back(edge.to);
            }
        }
    }
    // If a cycle slipped through (unvalidated graph), fall back to id order so
    // rendering and iteration stay total and deterministic.
    if (topo_cache_.size() != nodes_.size()) {
        topo_cache_.clear();
        for (const Node& node : nodes_) {
            topo_cache_.push_back(node.id);
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

std::string GoalGraph::render_ascii() const {
    if (nodes_.empty()) {
        return "  No goal graph. Use `/goal plan <objective>` to create one.\n";
    }

    // Depth = longest path over all edges; layers render the fan-out structure.
    std::vector<std::size_t> depth(nodes_.size() + 1, 0);
    for (const NodeId id : topological_order()) {
        for (const Edge& edge : out_edges(id)) {
            depth[edge.to] = std::max(depth[edge.to], depth[id] + 1);
        }
    }
    std::size_t max_depth = 0;
    for (const Node& node : nodes_) {
        max_depth = std::max(max_depth, depth[node.id]);
    }

    std::string out;
    out += std::format("  Goal graph (plan v{}) — {} node(s), {} edge(s)\n",
                       plan_version_, nodes_.size(), edges_.size());
    if (!objective_.empty()) {
        out += std::format("  Objective: {}\n", objective_);
    }

    for (std::size_t level = 0; level <= max_depth; ++level) {
        for (const Node& node : nodes_) {
            if (depth[node.id] != level) {
                continue;
            }
            out += std::format("  {} {} ({}", state_icon(node.state), node.name,
                               to_string(node.kind));
            if (node.kind == NodeKind::Work || node.kind == NodeKind::Verify) {
                out += std::format(", attempt {}/{}", node.attempts, node.max_attempts);
            }
            if (node.parallelizable) {
                out += ", parallel";
            }
            out += ")\n";
            if (node.kind == NodeKind::Verify && !node.check_command.empty()) {
                out += std::format("      check: {}\n", node.check_command);
            }
            if (!node.lessons.empty()) {
                out += std::format("      lessons: {}\n", node.lessons.size());
            }
        }
    }

    if (!edges_.empty()) {
        out += "  edges:\n";
        for (const Edge& edge : edges_) {
            const Node* from = node(edge.from);
            const Node* to = node(edge.to);
            if (from == nullptr || to == nullptr) {
                continue;
            }
            out += std::format("    {} --{}--> {}\n", from->name,
                               to_string(edge.condition), to->name);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

std::string GoalGraph::to_json() const {
    std::string out;
    out.reserve(1024 + nodes_.size() * 256);
    out += "{\"version\":1,\"plan_version\":";
    out += std::to_string(plan_version_);
    out += ",\"objective\":\"";
    core::utils::append_escaped(out, objective_);
    out += "\",\"nodes\":[";
    bool first_node = true;
    for (const Node& node : nodes_) {
        if (!std::exchange(first_node, false)) {
            out += ',';
        }
        out += "{\"id\":";
        out += std::to_string(node.id);
        out += ",\"kind\":\"";
        out += to_string(node.kind);
        out += "\",\"name\":\"";
        core::utils::append_escaped(out, node.name);
        out += "\",\"directive\":\"";
        core::utils::append_escaped(out, node.directive);
        out += "\",\"check\":\"";
        core::utils::append_escaped(out, node.check_command);
        out += "\",\"acceptance\":\"";
        core::utils::append_escaped(out, node.acceptance);
        out += "\",\"max_attempts\":";
        out += std::to_string(node.max_attempts);
        out += ",\"attempts\":";
        out += std::to_string(node.attempts);
        out += ",\"state\":\"";
        out += to_string(node.state);
        out += "\",\"parallel\":";
        out += node.parallelizable ? "true" : "false";
        out += ",\"retry_target\":";
        out += std::to_string(node.retry_target);
        out += ",\"result\":\"";
        core::utils::append_escaped(out, node.result_summary);
        out += "\",\"lessons\":[";
        bool first_lesson = true;
        for (const std::string& lesson : node.lessons) {
            if (!std::exchange(first_lesson, false)) {
                out += ',';
            }
            out += '\"';
            core::utils::append_escaped(out, lesson);
            out += '\"';
        }
        out += "]}";
    }
    out += "],\"edges\":[";
    bool first_edge = true;
    for (const Edge& edge : edges_) {
        if (!std::exchange(first_edge, false)) {
            out += ',';
        }
        out += "{\"from\":";
        out += std::to_string(edge.from);
        out += ",\"to\":";
        out += std::to_string(edge.to);
        out += ",\"condition\":\"";
        out += to_string(edge.condition);
        out += "\"}";
    }
    out += "]}";
    return out;
}

std::expected<GoalGraph, std::string> GoalGraph::from_json(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    const simdjson::padded_string padded(json);
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS || !doc.is_object()) {
        return std::unexpected("goal graph snapshot is not valid JSON");
    }

    GoalGraph graph;
    int64_t plan_version = 0;
    if (doc["plan_version"].get(plan_version) == simdjson::SUCCESS && plan_version > 0) {
        graph.plan_version_ = static_cast<int>(plan_version);
    }
    std::string_view sv;
    if (doc["objective"].get(sv) == simdjson::SUCCESS) {
        graph.set_objective(sv);
    }

    simdjson::dom::array nodes;
    if (doc["nodes"].get(nodes) != simdjson::SUCCESS) {
        return std::unexpected("goal graph snapshot is missing nodes");
    }
    for (simdjson::dom::element element : nodes) {
        Node node;
        if (element["kind"].get(sv) == simdjson::SUCCESS) {
            const auto kind = node_kind_from_string(sv);
            if (!kind.has_value()) {
                return std::unexpected(
                    std::format("goal graph snapshot has unknown node kind '{}'", sv));
            }
            node.kind = *kind;
        }
        if (element["name"].get(sv) == simdjson::SUCCESS) node.name = std::string(sv);
        if (element["directive"].get(sv) == simdjson::SUCCESS) node.directive = std::string(sv);
        if (element["check"].get(sv) == simdjson::SUCCESS) node.check_command = std::string(sv);
        if (element["acceptance"].get(sv) == simdjson::SUCCESS) node.acceptance = std::string(sv);
        int64_t number = 0;
        if (element["max_attempts"].get(number) == simdjson::SUCCESS) {
            node.max_attempts = static_cast<int>(number);
        }
        if (element["attempts"].get(number) == simdjson::SUCCESS) {
            node.attempts = static_cast<int>(number);
        }
        if (element["state"].get(sv) == simdjson::SUCCESS) {
            node.state = node_state_from_string(sv);
            // A snapshot taken mid-run must not resurrect phantom work.
            if (node.state == NodeState::Running) {
                node.state = NodeState::Ready;
            }
        }
        bool flag = false;
        if (element["parallel"].get(flag) == simdjson::SUCCESS) node.parallelizable = flag;
        if (element["retry_target"].get(number) == simdjson::SUCCESS) {
            node.retry_target = static_cast<NodeId>(number);
        }
        if (element["result"].get(sv) == simdjson::SUCCESS) node.result_summary = std::string(sv);
        simdjson::dom::array lessons;
        if (element["lessons"].get(lessons) == simdjson::SUCCESS) {
            for (simdjson::dom::element lesson : lessons) {
                if (lesson.get(sv) == simdjson::SUCCESS) {
                    node.lessons.emplace_back(sv);
                }
            }
        }
        const auto added = graph.add_node(std::move(node));
        if (!added.has_value()) {
            return std::unexpected(added.error());
        }
    }

    simdjson::dom::array edges;
    if (doc["edges"].get(edges) == simdjson::SUCCESS) {
        for (simdjson::dom::element element : edges) {
            int64_t from = 0;
            int64_t to = 0;
            if (element["from"].get(from) != simdjson::SUCCESS
                || element["to"].get(to) != simdjson::SUCCESS) {
                return std::unexpected("goal graph snapshot has a malformed edge");
            }
            EdgeCondition condition = EdgeCondition::Always;
            if (element["condition"].get(sv) == simdjson::SUCCESS) {
                const auto parsed = edge_condition_from_string(sv);
                if (!parsed.has_value()) {
                    return std::unexpected(std::format(
                        "goal graph snapshot has unknown edge condition '{}'", sv));
                }
                condition = *parsed;
            }
            const auto added = graph.add_edge(static_cast<NodeId>(from),
                                              static_cast<NodeId>(to), condition);
            if (!added.has_value()) {
                return std::unexpected(added.error());
            }
        }
    }

    if (const auto valid = graph.validate(); !valid.has_value()) {
        return std::unexpected(std::format("goal graph snapshot is invalid: {}", valid.error()));
    }
    return graph;
}

} // namespace core::goal
