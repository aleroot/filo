#include "GoalReflector.hpp"

#include <format>
#include <sstream>

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

[[nodiscard]] std::string clamp_lesson(std::string_view lesson) {
    std::string out(trim(lesson));
    if (out.size() > Node::kMaxLessonChars) {
        out.resize(Node::kMaxLessonChars);
    }
    return out;
}

} // namespace

GoalReflector::GoalReflector(CompletionFn complete)
    : complete_(std::move(complete)) {}

Reflection GoalReflector::heuristic(const Node& node, const Verdict& verdict) {
    Reflection reflection;
    const std::string reason = verdict.reason.empty()
        ? std::format("node '{}' failed without a recorded reason", node.name)
        : verdict.reason;
    reflection.critique = reason;

    // The lesson must restate the concrete failure signal: it is replayed
    // verbatim into the next attempt's prompt, where the verdict is no longer
    // in scope.
    reflection.lessons.push_back(
        verdict.deterministic
            ? std::format("The verification check failed: {}. Read the command output, "
                          "fix the root cause rather than the symptom, and re-run the "
                          "check before reporting success.", reason)
            : std::format("Address the reviewer's objection: {}", reason));
    return reflection;
}

Reflection GoalReflector::parse_response(std::string_view response) {
    Reflection reflection;

    const auto lessons_pos = response.find("LESSONS:");
    const std::string_view critique_region = lessons_pos == std::string_view::npos
        ? response
        : response.substr(0, lessons_pos);

    std::string_view critique = trim(critique_region);
    constexpr std::string_view kCritiqueMarker = "CRITIQUE:";
    if (critique.starts_with(kCritiqueMarker)) {
        critique = trim(critique.substr(kCritiqueMarker.size()));
    }
    reflection.critique = std::string(critique);

    if (lessons_pos != std::string_view::npos) {
        std::istringstream stream(
            std::string(response.substr(lessons_pos + 8)) );
        std::string line;
        while (std::getline(stream, line)) {
            std::string_view item = trim(line);
            if (item.starts_with("- ") || item.starts_with("* ")) {
                item = item.substr(2);
            } else if (item.size() > 2 && item[0] >= '0' && item[0] <= '9'
                       && (item[1] == '.' || item[1] == ')')) {
                item = trim(item.substr(2));
            }
            item = trim(item);
            if (item.empty()) {
                continue;
            }
            reflection.lessons.push_back(clamp_lesson(item));
            if (reflection.lessons.size() >= Node::kMaxLessons) {
                break;
            }
        }
    }

    if (reflection.critique.empty() && reflection.lessons.empty()) {
        reflection.critique = std::string(trim(response));
    }
    return reflection;
}

Reflection GoalReflector::reflect(const Node& node,
                                  const Verdict& verdict,
                                  const VerifyContext& context) const {
    if (!complete_) {
        return heuristic(node, verdict);
    }

    std::string prompt;
    prompt += "You are the reflection module of an autonomous coding agent. A work node\n"
              "failed its verification. Produce a sharp critique and actionable lessons\n"
              "that will make the NEXT attempt succeed. Be specific: name files, commands,\n"
              "and root causes visible in the evidence. Do not repeat prior lessons.\n\n";
    prompt += std::format("Work node: {}\nDirective:\n{}\n\n", node.name,
                          node.directive.empty() ? "(unspecified)" : node.directive);
    prompt += std::format("Verification verdict ({}): {}\n",
                          verdict.deterministic ? "deterministic" : "model-judged",
                          verdict.reason.empty() ? "failed" : verdict.reason);
    if (!verdict.evidence.empty()) {
        prompt += "\nEvidence:\n" + verdict.evidence + "\n";
    }
    if (!context.work_output.empty()) {
        prompt += "\nWork output:\n" + context.work_output + "\n";
    }
    if (!node.lessons.empty()) {
        prompt += "\nPrior lessons (do not repeat):\n";
        for (const std::string& lesson : node.lessons) {
            prompt += "- " + lesson + "\n";
        }
    }
    prompt += "\nRespond with EXACTLY this contract:\n"
              "CRITIQUE:\n<one short paragraph>\n"
              "LESSONS:\n- <lesson 1>\n- <lesson 2>\n";

    const auto response = complete_(prompt);
    if (!response.has_value()) {
        return heuristic(node, verdict);
    }

    Reflection reflection = parse_response(*response);
    if (reflection.critique.empty() && reflection.lessons.empty()) {
        return heuristic(node, verdict);
    }
    return reflection;
}

std::string GoalReflector::build_retry_context(const Node& node) {
    if (node.lessons.empty()) {
        return {};
    }
    std::string out;
    out += "\n\nLessons from previous failed attempts (apply them, do not repeat them):\n";
    for (const std::string& lesson : node.lessons) {
        out += "- " + lesson + "\n";
    }
    return out;
}

} // namespace core::goal
