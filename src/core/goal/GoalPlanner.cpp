#include "GoalPlanner.hpp"

#include "../llm/OneShotCompletion.hpp"

#include <format>
#include <simdjson.h>
#include <unordered_map>
#include <unordered_set>

namespace core::goal {

namespace {

[[nodiscard]] std::string_view trim(std::string_view value) noexcept {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

[[nodiscard]] bool has_dependency_path(const GoalGraph &graph, NodeId from,
                                       NodeId target) {
  std::vector<NodeId> frontier{from};
  std::unordered_set<NodeId> visited;
  while (!frontier.empty()) {
    const NodeId current = frontier.back();
    frontier.pop_back();
    if (!visited.insert(current).second) {
      continue;
    }
    for (const Edge &edge : graph.out_edges(current)) {
      if (!is_dependency(edge.condition)) {
        continue;
      }
      if (edge.to == target) {
        return true;
      }
      frontier.push_back(edge.to);
    }
  }
  return false;
}

[[nodiscard]] bool valid_auto_plan(const GoalGraph &graph) {
  NodeId writer = kInvalidNodeId;
  std::size_t exploration_nodes = 0;

  for (const Node &node : graph.nodes()) {
    if (!node.check_command.empty() || !node.verification_recipe_ids.empty()) {
      return false;
    }
    if (node.kind != NodeKind::Work && node.kind != NodeKind::FanIn) {
      return false;
    }
    if (node.kind == NodeKind::FanIn) {
      if (node.parallelizable) {
        return false;
      }
      continue;
    }
    if (node.workspace_access == WorkspaceAccess::ExclusiveWrite) {
      if (writer != kInvalidNodeId || node.parallelizable) {
        return false;
      }
      writer = node.id;
    } else {
      ++exploration_nodes;
      if (!node.parallelizable || !graph.in_edges(node.id).empty()) {
        return false;
      }
    }
  }
  if (writer == kInvalidNodeId || exploration_nodes > 4 ||
      !graph.out_edges(writer).empty()) {
    return false;
  }
  for (const Edge &edge : graph.edges()) {
    if (!is_dependency(edge.condition)) {
      return false;
    }
  }
  for (const Node &node : graph.nodes()) {
    if (node.id != writer && !has_dependency_path(graph, node.id, writer)) {
      return false;
    }
  }
  return true;
}

/// Extract the first balanced {...} region, tolerating prose or markdown
/// fences around the JSON payload the planner contract demands.
[[nodiscard]] std::string_view extract_json_object(std::string_view text) {
    const auto open = text.find('{');
    if (open == std::string_view::npos) {
        return {};
    }
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = open; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            if (--depth == 0) {
                return text.substr(open, i - open + 1);
            }
        }
    }
    return {};
}

[[nodiscard]] std::string build_planner_prompt(std::string_view objective,
                                               std::string_view context,
                                               PlanProfile profile) {
  std::string prompt;
  prompt +=
      "You are the planning module of an autonomous coding agent. Decompose "
      "the\n"
      "goal below into a directed acyclic graph of typed nodes. Rules:\n"
      "- Every work node declares `workspace_access` as `shared_read` or\n"
      "  `exclusive_write`.\n"
      "- Only shared_read exploration nodes may set `parallel` to true.\n"
      "- Retry loops use `max_attempts`, NEVER graph back-edges.\n"
      "- Keep the graph small: 2 to 8 nodes. Fan out only when branches are\n"
      "  genuinely independent.\n";
  if (profile == PlanProfile::AutoExecution) {
    prompt +=
        "- This is an AUTO preflight plan. Create zero to four independent\n"
        "  shared_read exploration nodes, optionally a fan_in node, and "
        "exactly one\n"
        "  final exclusive_write work node.\n"
        "- Do NOT emit verify nodes or shell commands. Filo owns "
        "deterministic\n"
        "  verification outside this graph through trusted repository "
        "recipes.\n"
        "- The final writer must depend on every exploration branch, directly "
        "or\n"
        "  through fan_in.\n\n";
  } else {
    prompt +=
        "- Prefer repository verification recipe ids supplied in context. A "
        "verify\n"
        "  node lists them in `verification_recipes`; do not invent ids.\n"
        "- Every verify node MUST set `retry_target` to the work node it "
        "checks.\n"
        "- Legacy `check` shell strings are unsupported in newly planned "
        "graphs.\n\n";
  }
  if (!trim(context).empty()) {
    prompt += "Repository context:\n";
    prompt += context;
    prompt += "\n\n";
  }
  prompt += "Goal:\n";
  prompt += objective;
  prompt +=
      "\n\nRespond with ONLY a JSON object (no prose, no fences):\n"
      "{\n"
      "  \"nodes\": [\n"
      "    {\"name\": \"explore-x\", \"kind\": \"work\", \"directive\": "
      "\"...\",\n"
      "     \"workspace_access\": \"shared_read\", \"parallel\": true,\n"
      "     \"max_attempts\": 2},\n"
      "    {\"name\": \"do-x\", \"kind\": \"work\", \"directive\": \"...\",\n"
      "     \"workspace_access\": \"exclusive_write\", \"max_attempts\": 3}";
  if (profile == PlanProfile::FullGoal) {
    prompt += ",\n"
              "    {\"name\": \"verify-x\", \"kind\": \"verify\",\n"
              "     \"directive\": \"Verify the acceptance criteria\",\n"
              "     \"verification_recipes\": [], \"retry_target\": \"do-x\"}";
  }
  prompt += "\n  ],\n"
            "  \"edges\": [\n"
            "    {\"from\": \"explore-x\", \"to\": \"do-x\", \"condition\": "
            "\"always\"}";
  if (profile == PlanProfile::FullGoal) {
    prompt += ",\n"
              "    {\"from\": \"do-x\", \"to\": \"verify-x\", "
              "\"condition\": \"on_pass\"}";
  }
  prompt += "\n  ]\n"
            "}\n"
            "Valid kinds: work, verify, fan_in, gate. "
            "Valid conditions: always, on_pass, on_fail, on_exhausted.\n";
  return prompt;
}

} // namespace

GoalPlanner::GoalPlanner(CompletionFn complete)
    : complete_(std::move(complete)) {}

GoalGraph GoalPlanner::fallback_linear_plan(std::string_view objective,
                                            PlanProfile profile,
                                            std::span<const std::string>
                                                verification_recipe_ids) {
  GoalGraph graph;
  graph.set_objective(objective);

  Node work;
  work.kind = NodeKind::Work;
  work.name = "goal";
  work.directive = std::string(objective);
  work.max_attempts = 3;
  work.workspace_access = WorkspaceAccess::ExclusiveWrite;
  const NodeId work_id = *graph.add_node(std::move(work));

  if (profile == PlanProfile::AutoExecution) {
    return graph;
  }

  Node verify;
  verify.kind = NodeKind::Verify;
  verify.name = "verify";
  verify.directive = "Verify the goal is achieved using concrete evidence";
  verify.acceptance = std::string(objective);
  verify.retry_target = work_id;
  verify.verification_recipe_ids.assign(verification_recipe_ids.begin(),
                                        verification_recipe_ids.end());
  const NodeId verify_id = *graph.add_node(std::move(verify));

  (void)graph.add_edge(work_id, verify_id, EdgeCondition::OnPass);
  return graph;
}

std::expected<GoalGraph, std::string> GoalPlanner::parse_plan_json(
    std::string_view json,
    std::string_view objective) {
    const std::string_view payload = extract_json_object(json);
    if (payload.empty()) {
        return std::unexpected("planner response contained no JSON object");
    }

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    const simdjson::padded_string padded(payload);
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS || !doc.is_object()) {
        return std::unexpected("planner response is not valid JSON");
    }

    GoalGraph graph;
    graph.set_objective(objective);

    simdjson::dom::array nodes;
    if (doc["nodes"].get(nodes) != simdjson::SUCCESS) {
        return std::unexpected("planner response is missing \"nodes\"");
    }

    std::unordered_map<std::string, NodeId> ids;
    std::unordered_map<std::string, std::string> pending_retry_targets;
    for (simdjson::dom::element element : nodes) {
        Node node;
        std::string_view sv;
        if (element["name"].get(sv) != simdjson::SUCCESS || trim(sv).empty()) {
            return std::unexpected("planner node is missing a name");
        }
        node.name = std::string(trim(sv));
        if (element["kind"].get(sv) == simdjson::SUCCESS) {
            const auto kind = node_kind_from_string(sv);
            if (!kind.has_value()) {
                return std::unexpected(
                    std::format("planner node '{}' has unknown kind '{}'", node.name, sv));
            }
            node.kind = *kind;
        }
        if (element["directive"].get(sv) == simdjson::SUCCESS) {
            node.directive = std::string(sv);
        }
        bool has_workspace_access = false;
        if (element["workspace_access"].get(sv) == simdjson::SUCCESS) {
          const auto access = workspace_access_from_string(sv);
          if (!access.has_value()) {
            return std::unexpected(std::format(
                "planner node '{}' has unknown workspace access '{}'",
                node.name, sv));
          }
          node.workspace_access = *access;
          has_workspace_access = true;
        }
        if (element["check"].get(sv) == simdjson::SUCCESS) {
            node.check_command = std::string(sv);
        }
        if (element["acceptance"].get(sv) == simdjson::SUCCESS) {
            node.acceptance = std::string(sv);
        }
        bool parallel = false;
        if (element["parallel"].get(parallel) == simdjson::SUCCESS) {
            node.parallelizable = parallel;
            if (parallel && has_workspace_access &&
                node.workspace_access != WorkspaceAccess::SharedRead) {
              return std::unexpected(std::format(
                  "planner node '{}' requests parallel write access",
                  node.name));
            }
            if (parallel && !has_workspace_access) {
              node.workspace_access = WorkspaceAccess::SharedRead;
            }
        }
        simdjson::dom::array verification_recipes;
        if (element["verification_recipes"].get(verification_recipes) ==
            simdjson::SUCCESS) {
          for (simdjson::dom::element recipe : verification_recipes) {
            if (recipe.get(sv) != simdjson::SUCCESS) {
              return std::unexpected(std::format(
                  "planner node '{}' verification recipes must be strings",
                  node.name));
            }
            node.verification_recipe_ids.emplace_back(sv);
          }
        }
        int64_t attempts = 0;
        if (element["max_attempts"].get(attempts) == simdjson::SUCCESS && attempts > 0) {
            node.max_attempts = static_cast<int>(attempts);
        }
        // Only kinds the scheduler executes are accepted from a model plan.
        switch (node.kind) {
        case NodeKind::Work:
        case NodeKind::Verify:
        case NodeKind::FanIn:
        case NodeKind::Gate:
            break;
        default:
            return std::unexpected(std::format(
                "planner node '{}' uses reserved kind '{}'", node.name, to_string(node.kind)));
        }

        std::string retry_target_name;
        if (element["retry_target"].get(sv) == simdjson::SUCCESS) {
            retry_target_name = std::string(trim(sv));
        }

        const auto added = graph.add_node(std::move(node));
        if (!added.has_value()) {
            return std::unexpected(added.error());
        }
        ids.emplace(graph.node(*added)->name, *added);
        if (!retry_target_name.empty()) {
            // Resolved in a second pass once all names are known.
            pending_retry_targets.emplace(graph.node(*added)->name,
                                          std::move(retry_target_name));
        }
    }

    // Second pass: resolve retry targets by name.
    for (const auto& [node_name, target_name] : pending_retry_targets) {
        const auto target = ids.find(target_name);
        if (target == ids.end()) {
            return std::unexpected(std::format(
                "planner node '{}' references unknown retry target '{}'",
                node_name, target_name));
        }
        const auto self = graph.find(node_name);
        graph.node(*self)->retry_target = target->second;
    }

    simdjson::dom::array edges;
    if (doc["edges"].get(edges) == simdjson::SUCCESS) {
        for (simdjson::dom::element element : edges) {
            std::string_view from;
            std::string_view to;
            if (element["from"].get(from) != simdjson::SUCCESS
                || element["to"].get(to) != simdjson::SUCCESS) {
                return std::unexpected("planner edge is missing from/to");
            }
            EdgeCondition condition = EdgeCondition::Always;
            std::string_view sv;
            if (element["condition"].get(sv) == simdjson::SUCCESS) {
                const auto parsed = edge_condition_from_string(sv);
                if (!parsed.has_value()) {
                    return std::unexpected(
                        std::format("planner edge has unknown condition '{}'", sv));
                }
                condition = *parsed;
            }
            const auto from_it = ids.find(std::string(trim(from)));
            const auto to_it = ids.find(std::string(trim(to)));
            if (from_it == ids.end() || to_it == ids.end()) {
                return std::unexpected(std::format(
                    "planner edge '{}' -> '{}' references an unknown node", from, to));
            }
            const auto added = graph.add_edge(from_it->second, to_it->second, condition);
            if (!added.has_value()) {
                return std::unexpected(added.error());
            }
        }
    }

    if (const auto valid = graph.validate(); !valid.has_value()) {
        return std::unexpected(std::format("planner produced an invalid graph: {}",
                                           valid.error()));
    }
    return graph;
}

std::expected<GoalGraph, std::string>
GoalPlanner::plan(std::string_view objective, std::string_view context,
                  PlanProfile profile,
                  std::span<const std::string>
                      fallback_verification_recipe_ids) const {
  if (trim(objective).empty()) {
    return std::unexpected("cannot plan an empty goal objective");
  }
  if (!complete_) {
    return fallback_linear_plan(objective, profile,
                                fallback_verification_recipe_ids);
  }

  const auto response =
      complete_(build_planner_prompt(objective, context, profile));
  if (!response.has_value()) {
    if (core::llm::is_cancelled_error(response.error())) {
      return std::unexpected(response.error());
    }
    // Model unreachable: degrade to the deterministic fallback rather
    // than blocking the goal.
    return fallback_linear_plan(objective, profile,
                                fallback_verification_recipe_ids);
  }

  auto parsed = parse_plan_json(*response, objective);
  if (!parsed.has_value()) {
    return fallback_linear_plan(objective, profile,
                                fallback_verification_recipe_ids);
  }
  if (profile == PlanProfile::AutoExecution) {
    if (!valid_auto_plan(*parsed)) {
      return fallback_linear_plan(objective, profile,
                                  fallback_verification_recipe_ids);
    }
  }
  return parsed;
}

} // namespace core::goal
