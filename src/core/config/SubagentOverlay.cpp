#include "SubagentOverlay.hpp"

#include "ConfigSource.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/JsonValue.hpp"

#include <exception>
#include <system_error>
#include <utility>

namespace core::config {
namespace {

using core::utils::JsonValue;

[[nodiscard]] std::string string_member(const JsonValue& object, std::string_view key) {
    const auto* value = object.find(key);
    return value && value->is_string() ? std::string(value->as_string()) : std::string{};
}

} // namespace

std::expected<SubagentModelChoices, std::string>
read_subagent_overlay(const std::filesystem::path& file) {
    std::optional<JsonValue> document;
    try {
        document = ConfigSource::read_document(file);
    } catch (const std::exception& error) {
        return std::unexpected(error.what());
    }
    SubagentModelChoices choices;
    if (!document) return choices;
    const auto* subagents = document->find("subagents");
    if (!subagents || !subagents->is_object()) return choices;
    for (const auto& [name, entry] : subagents->members()) {
        if (!entry.is_object()) continue;
        SubagentModelChoice choice{string_member(entry, "provider"), string_member(entry, "model")};
        if (choice.provider.empty() && choice.model.empty()) continue;
        choices.emplace(name, std::move(choice));
    }
    return choices;
}

bool write_subagent_model(const std::filesystem::path& file,
                          std::string_view name,
                          const std::optional<SubagentModelChoice>& choice,
                          std::string* error) {
    const auto fail = [error](std::string message) {
        if (error) *error = std::move(message);
        return false;
    };
    std::optional<JsonValue> document;
    try {
        document = ConfigSource::read_document(file);
    } catch (const std::exception& read_error) {
        return fail(read_error.what());
    }
    JsonValue root = document ? std::move(*document) : JsonValue::object();
    JsonValue* subagents = root.find("subagents");
    if (!subagents || !subagents->is_object()) {
        subagents = &root.set("subagents", JsonValue::object());
    }

    if (choice) {
        JsonValue* entry = subagents->find(name);
        if (!entry || !entry->is_object()) entry = &subagents->set(name, JsonValue::object());
        // Only what this setting owns is replaced: a hand-written entry keeps
        // its other fields (max_steps, allowed tools, ...).
        const auto put = [entry](std::string_view key, const std::string& value) {
            if (value.empty()) {
                entry->erase(key);
            } else {
                entry->set(key, JsonValue(value));
            }
        };
        put("provider", choice->provider);
        put("model", choice->model);
    } else if (JsonValue* entry = subagents->find(name); entry && entry->is_object()) {
        entry->erase("provider");
        entry->erase("model");
        if (entry->members().empty()) subagents->erase(name);
    }
    if (subagents->members().empty()) root.erase("subagents");

    if (root.members().empty()) {
        std::error_code remove_error;
        std::filesystem::remove(file, remove_error);
        if (remove_error) return fail(file.string() + ": " + remove_error.message());
        return true;
    }
    std::error_code mkdir_error;
    std::filesystem::create_directories(file.parent_path(), mkdir_error);
    std::string write_error;
    if (!core::utils::atomic_write_file(file, root.dump() + "\n", &write_error)) {
        return fail(std::move(write_error));
    }
    return true;
}

} // namespace core::config
