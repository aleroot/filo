#pragma once

#include <expected>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace core::config {

/// The file a settings scope keeps subagent model choices in. Its shape is a
/// config.json fragment, `{"subagents":{"<name>":{"provider":..,"model":..}}}`,
/// so the normal configuration loader layers it like any other overlay.
inline constexpr std::string_view kSubagentOverlayFile = "subagents.json";

struct SubagentModelChoice {
    std::string provider;
    std::string model;

    bool operator==(const SubagentModelChoice&) const = default;
};

/// What one scope file sets, by subagent name. An absent file sets nothing;
/// a malformed one is an error, so a save never rewrites what it could not read.
using SubagentModelChoices = std::map<std::string, SubagentModelChoice, std::less<>>;

[[nodiscard]] std::expected<SubagentModelChoices, std::string>
read_subagent_overlay(const std::filesystem::path& file);

/// Sets (or, with nullopt, removes) one subagent's provider and model in
/// `file`, keeping every other key the file holds. A file left with nothing
/// in it is deleted. Not locked: the caller serializes writers.
[[nodiscard]] bool write_subagent_model(const std::filesystem::path& file,
                                        std::string_view name,
                                        const std::optional<SubagentModelChoice>& choice,
                                        std::string* error = nullptr);

} // namespace core::config
