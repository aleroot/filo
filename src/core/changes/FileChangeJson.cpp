#include "FileChangeJson.hpp"

#include "../utils/JsonUtils.hpp"

#include <array>
#include <cstdint>
#include <format>
#include <string_view>
#include <vector>

namespace core::changes {
namespace {

constexpr std::array kKinds{
    FileChangeKind::Added, FileChangeKind::Modified,
    FileChangeKind::Deleted, FileChangeKind::Renamed,
};
constexpr std::array kContents{
    FileChangeContent::Text, FileChangeContent::BudgetSpent,
    FileChangeContent::TooLarge, FileChangeContent::Binary,
};

template <typename Enum, std::size_t N>
Enum parse_enum(std::string_view text, const std::array<Enum, N>& values, Enum fallback) {
    for (const auto value : values) {
        if (to_string(value) == text) {
            return value;
        }
    }
    return fallback;
}

void append_string_field(std::string& out, std::string_view name, std::string_view value) {
    out += '"';
    out += name;
    out += "\":\"";
    core::utils::append_escaped(out, value);
    out += '"';
}

void append_bool_field(std::string& out, std::string_view name, bool value) {
    out += std::format(",\"{}\":{}", name, value ? "true" : "false");
}

void append_files(std::string& out, const std::vector<FileChange>& files) {
    out += '[';
    for (std::size_t i = 0; i < files.size(); ++i) {
        const auto& change = files[i];
        if (i > 0) {
            out += ',';
        }
        out += '{';
        append_string_field(out, "kind", to_string(change.kind));
        out += ',';
        append_string_field(out, "content", to_string(change.content));
        out += ',';
        append_string_field(out, "path", change.path);
        if (!change.previous_path.empty()) {
            out += ',';
            append_string_field(out, "previous_path", change.previous_path);
        }
        out += std::format(",\"added\":{},\"deleted\":{}", change.added, change.deleted);
        if (!change.diff.empty()) {
            out += ',';
            append_string_field(out, "diff", change.diff);
        }
        out += '}';
    }
    out += ']';
}

std::vector<FileChange> parse_files(simdjson::dom::array array) {
    std::vector<FileChange> files;
    for (simdjson::dom::element element : array) {
        FileChange change;
        std::string_view text;
        if (element["path"].get(text) != simdjson::SUCCESS || text.empty()) {
            continue;
        }
        change.path = std::string(text);
        if (element["kind"].get(text) == simdjson::SUCCESS) {
            change.kind = parse_enum(text, kKinds, FileChangeKind::Modified);
        }
        if (element["content"].get(text) == simdjson::SUCCESS) {
            change.content = parse_enum(text, kContents, FileChangeContent::Text);
        }
        if (element["previous_path"].get(text) == simdjson::SUCCESS) {
            change.previous_path = std::string(text);
        }
        if (element["diff"].get(text) == simdjson::SUCCESS) {
            change.diff = std::string(text);
        }
        std::uint64_t count = 0;
        if (element["added"].get(count) == simdjson::SUCCESS) {
            change.added = static_cast<std::size_t>(count);
        }
        if (element["deleted"].get(count) == simdjson::SUCCESS) {
            change.deleted = static_cast<std::size_t>(count);
        }
        files.push_back(std::move(change));
    }
    return files;
}

} // namespace

void append_turn_changes_json(std::string& out, const TurnChanges& changes) {
    out += "{\"files\":";
    append_files(out, changes.files);
    if (changes.partial_enumeration) {
        append_bool_field(out, "partial_enumeration", true);
    }
    if (changes.unscoped_mutations) {
        append_bool_field(out, "unscoped_mutations", true);
    }
    if (!changes.unscoped_tools.empty()) {
        out += ",\"unscoped_tools\":";
        core::utils::append_string_array(out, changes.unscoped_tools);
    }
    if (changes.reverted) {
        append_bool_field(out, "reverted", true);
    }
    out += '}';
}

TurnChanges parse_turn_changes_json(simdjson::dom::object object) {
    TurnChanges changes;
    if (simdjson::dom::array files; object["files"].get(files) == simdjson::SUCCESS) {
        changes.files = parse_files(files);
    }
    changes.partial_enumeration =
        core::utils::json::bool_field(object, "partial_enumeration");
    changes.unscoped_mutations =
        core::utils::json::bool_field(object, "unscoped_mutations");
    // A summary written before tool names were kept still carries the flag, so
    // the gap survives a resume even when nothing can be named for it.
    if (simdjson::dom::array tools;
        object["unscoped_tools"].get(tools) == simdjson::SUCCESS) {
        for (simdjson::dom::element element : tools) {
            if (std::string_view name; element.get(name) == simdjson::SUCCESS) {
                changes.unscoped_tools.emplace_back(name);
            }
        }
    }
    changes.reverted = core::utils::json::bool_field(object, "reverted");
    return changes;
}

} // namespace core::changes
