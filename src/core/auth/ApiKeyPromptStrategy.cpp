#include "ApiKeyPromptStrategy.hpp"

#include "ui/ConsoleAuthUI.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/utils/JsonWriter.hpp"
#include "core/utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace core::auth {
namespace {

constexpr std::string_view kDefaultInstructions =
    "Paste an API key for this provider. It will be saved in Filo's "
    "auth_defaults.json overlay and used as the default provider.";

struct AuthOverlayProvider {
    std::string model;
    std::string auth_type;
    std::string api_key;
};

void write_api_key_overlay(std::string_view config_dir,
                           std::string_view default_provider,
                           const std::vector<ApiKeyProviderSeed>& provider_seeds,
                           std::string_view api_key) {
    if (api_key.empty()) {
        throw std::runtime_error("API key cannot be empty.");
    }
    if (default_provider.empty() || provider_seeds.empty()) {
        throw std::runtime_error("Provider login is not configured correctly.");
    }

    const std::filesystem::path dir(config_dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / "auth_defaults.json";

    std::unordered_map<std::string, AuthOverlayProvider> providers;
    if (std::filesystem::exists(path)) {
        try {
            simdjson::padded_string json =
                simdjson::padded_string::load(path.string());
            simdjson::dom::parser parser;
            simdjson::dom::element doc = parser.parse(json);
            simdjson::dom::object providers_obj;
            if (doc["providers"].get(providers_obj) == simdjson::SUCCESS) {
                for (auto field : providers_obj) {
                    simdjson::dom::object provider_obj;
                    if (field.value.get(provider_obj) != simdjson::SUCCESS) {
                        continue;
                    }
                    AuthOverlayProvider saved;
                    std::string_view value;
                    if (provider_obj["model"].get(value) == simdjson::SUCCESS) {
                        saved.model = std::string(value);
                    }
                    if (provider_obj["auth_type"].get(value) == simdjson::SUCCESS) {
                        saved.auth_type = std::string(value);
                    }
                    if (provider_obj["api_key"].get(value) == simdjson::SUCCESS) {
                        saved.api_key = std::string(value);
                    }
                    providers[std::string(field.key)] = std::move(saved);
                }
            }
        } catch (...) {
            providers.clear();
        }
    }

    for (const auto& seed : provider_seeds) {
        if (seed.provider_name.empty()) continue;
        AuthOverlayProvider& selected = providers[seed.provider_name];
        selected.model = seed.model;
        selected.auth_type.clear();
        selected.api_key = std::string(api_key);
    }

    std::vector<std::string> names;
    names.reserve(providers.size());
    for (const auto& [name, provider] : providers) {
        if (!provider.model.empty()
            || !provider.auth_type.empty()
            || !provider.api_key.empty()) {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());

    core::utils::JsonWriter writer(512);
    {
        auto root = writer.object();
        writer.kv_str("default_provider", default_provider).comma();
        writer.kv_str("default_model_selection", "manual").comma();
        writer.key("providers");
        auto providers_object = writer.object();
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (i > 0) writer.comma();
            writer.key(names[i]);
            auto provider_object = writer.object();
            const auto& provider = providers.at(names[i]);
            bool has_field = false;
            if (!provider.model.empty()) {
                writer.kv_str("model", provider.model);
                has_field = true;
            }
            if (!provider.auth_type.empty()) {
                if (has_field) writer.comma();
                writer.kv_str("auth_type", provider.auth_type);
                has_field = true;
            }
            if (!provider.api_key.empty()) {
                if (has_field) writer.comma();
                writer.kv_str("api_key", provider.api_key);
            }
        }
    }

    std::string payload = std::move(writer).take();
    payload.push_back('\n');

    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("Failed to open auth overlay for writing.");
        }
        out << payload;
        if (!out) {
            throw std::runtime_error("Failed to write auth overlay.");
        }
    }
    std::filesystem::rename(tmp, path);
}

/**
 * Present a numbered menu and return the chosen index, or nullopt when the
 * user cancels with "q" or the input stream is closed. A closed stream must
 * end the flow: re-prompting a menu that can never be answered would loop.
 */
[[nodiscard]] std::optional<std::size_t> prompt_selection(
    ui::AuthUI& ui,
    std::string_view instructions,
    const std::vector<AuthMenuOption>& options,
    std::string_view prompt_label,
    std::size_t default_index = 0) {
    if (options.empty() || default_index >= options.size()) {
        throw std::logic_error("Authentication selection is not configured correctly.");
    }

    std::string menu(instructions);
    for (std::size_t i = 0; i < options.size(); ++i) {
        menu += "\n  " + std::to_string(i + 1) + ". " + options[i].label;
        if (!options[i].description.empty()) {
            menu += "\n     " + options[i].description;
        }
    }
    menu += "\n  q. Cancel";
    ui.show_instructions(menu);

    const std::string prompt = std::string(prompt_label) + " ["
        + std::to_string(default_index + 1) + "]:";
    for (;;) {
        const std::optional<std::string> entered = ui.prompt_text(prompt);
        if (!entered.has_value()) return std::nullopt;

        const std::string value = core::utils::str::trim_ascii_copy(*entered);
        if (value == "q" || value == "Q") return std::nullopt;
        if (value.empty()) return default_index;

        std::size_t selected = 0;
        const auto [end, error] = std::from_chars(
            value.data(), value.data() + value.size(), selected);
        if (error == std::errc{} && end == value.data() + value.size()
            && selected >= 1 && selected <= options.size()) {
            return selected - 1;
        }
        ui.show_error("Enter a number from 1 to " + std::to_string(options.size())
                      + ", or q to cancel.");
    }
}

void validate_key_for_profile(const ApiKeyLoginProfile& profile,
                              std::string_view api_key) {
    if (profile.accepted_key_prefixes.empty()) return;

    for (const std::string& prefix : profile.accepted_key_prefixes) {
        if (!prefix.empty() && api_key.starts_with(prefix)) {
            return;
        }
    }

    if (!profile.invalid_key_message.empty()) {
        throw std::invalid_argument(profile.invalid_key_message);
    }
    throw std::invalid_argument(
        "The pasted API key is not valid for the selected endpoint.");
}

/// Hints describing the profile a login actually saved.
[[nodiscard]] std::vector<std::string> hints_for_profile(
    const ApiKeyLoginProfile& profile,
    std::string_view env_var,
    std::string_view docs_hint) {
    std::vector<std::string> hints;
    if (profile.default_model.empty()) {
        hints.push_back("Default provider is set to '" + profile.provider_name
                        + "'; Filo will select the newest model from its live catalog.");
    } else {
        hints.push_back("Default provider is set to '" + profile.provider_name
                        + "' with model '" + profile.default_model + "'.");
    }
    if (!env_var.empty()) {
        hints.push_back("For CI or one-off use, you can also export "
                        + std::string(env_var) + ".");
    }
    if (!docs_hint.empty()) {
        hints.push_back(std::string(docs_hint));
    }
    return hints;
}

} // namespace

void save_api_key_profile(std::string_view config_dir,
                          const ApiKeyLoginProfile& profile,
                          std::string_view api_key) {
    auto seeds = profile.additional_provider_seeds;
    seeds.insert(seeds.begin(), ApiKeyProviderSeed{
        .provider_name = profile.provider_name,
        .model = profile.default_model,
    });
    write_api_key_overlay(config_dir, profile.provider_name, seeds, api_key);

    // A /model choice is intentionally loaded after auth_defaults.json. Make
    // this explicit authentication action the current manual selection too,
    // otherwise a stale model-default overlay can silently select a different
    // regional endpoint with an incompatible key.
    std::string selection_error;
    if (!core::config::persist_model_defaults_overlay(
            std::filesystem::path(config_dir),
            profile.provider_name,
            "manual",
            profile.default_model,
            &selection_error)) {
        throw std::runtime_error(
            "Credential was saved, but Filo could not activate this provider: "
            + selection_error);
    }
}

ApiKeyPromptStrategy::ApiKeyPromptStrategy(ApiKeyPromptStrategySpec spec)
    : spec_(std::move(spec)) {}

std::string_view ApiKeyPromptStrategy::login_provider() const noexcept {
    return spec_.login_provider;
}

std::string_view ApiKeyPromptStrategy::display_name() const noexcept {
    return spec_.display_name;
}

std::vector<std::string_view> ApiKeyPromptStrategy::login_aliases() const {
    std::vector<std::string_view> aliases;
    aliases.reserve(spec_.login_aliases.size());
    for (const std::string& alias : spec_.login_aliases) {
        aliases.push_back(alias);
    }
    return aliases;
}

bool ApiKeyPromptStrategy::supports(
    std::string_view /*provider_type*/,
    std::string_view /*auth_type*/) const noexcept {
    return false;
}

std::shared_ptr<ICredentialSource>
ApiKeyPromptStrategy::create_credential_source(
    const core::config::ProviderConfig& /*provider_config*/,
    std::string_view /*config_dir*/) const {
    return nullptr;
}

std::vector<std::string> ApiKeyPromptStrategy::login(
    std::string_view config_dir) const {
    if (spec_.profiles.empty()) {
        throw std::logic_error("API key login has no configured profiles.");
    }

    const std::string instructions = spec_.instructions.empty()
        ? std::string(kDefaultInstructions)
        : spec_.instructions;

    ui::ConsoleAuthUI ui;
    ui.show_header(spec_.display_name + " API Key Login");
    std::size_t selected = 0;
    if (spec_.profiles.size() == 1) {
        ui.show_instructions(instructions);
    } else {
        std::vector<AuthMenuOption> options;
        options.reserve(spec_.profiles.size());
        for (const auto& profile : spec_.profiles) {
            options.push_back(profile.selection);
        }
        const std::optional<std::size_t> choice = prompt_selection(
            ui, instructions, options, "Endpoint", /*default_index=*/0);
        if (!choice.has_value()) {
            throw LoginCancelled{};
        }
        selected = *choice;
    }

    const ApiKeyLoginProfile& profile = spec_.profiles[selected];
    const std::string key = ui.prompt_secret(profile.key_prompt);
    validate_key_for_profile(profile, key);
    save_api_key_profile(config_dir, profile, key);
    ui.show_success(spec_.display_name + " credential saved for '"
                    + profile.provider_name + "'.");
    return hints_for_profile(profile, spec_.env_var, spec_.docs_hint);
}

CompositeAuthStrategy::CompositeAuthStrategy(CompositeAuthStrategySpec spec)
    : spec_(std::move(spec)) {}

std::string_view CompositeAuthStrategy::login_provider() const noexcept {
    return spec_.login_provider;
}

std::string_view CompositeAuthStrategy::display_name() const noexcept {
    return spec_.display_name;
}

std::vector<std::string_view> CompositeAuthStrategy::login_aliases() const {
    std::vector<std::string_view> aliases;
    aliases.reserve(spec_.login_aliases.size());
    for (const std::string& alias : spec_.login_aliases) {
        aliases.push_back(alias);
    }
    return aliases;
}

bool CompositeAuthStrategy::supports(
    std::string_view /*provider_type*/,
    std::string_view /*auth_type*/) const noexcept {
    return false;
}

std::shared_ptr<ICredentialSource>
CompositeAuthStrategy::create_credential_source(
    const core::config::ProviderConfig& /*provider_config*/,
    std::string_view /*config_dir*/) const {
    return nullptr;
}

std::vector<std::string> CompositeAuthStrategy::login(
    std::string_view config_dir) const {
    if (spec_.choices.empty()) {
        throw std::logic_error("Authentication menu has no configured choices.");
    }

    ui::ConsoleAuthUI ui;
    ui.show_header(spec_.display_name + " Login");
    std::vector<AuthMenuOption> options;
    options.reserve(spec_.choices.size());
    for (const auto& choice : spec_.choices) {
        options.push_back(choice.selection);
    }

    for (;;) {
        const std::optional<std::size_t> selected =
            prompt_selection(ui, spec_.instructions, options, "Method",
                             /*default_index=*/0);
        if (!selected.has_value()) {
            throw LoginCancelled{};
        }
        const AuthStrategyChoice& choice = spec_.choices[*selected];
        if (!choice.strategy) {
            throw std::logic_error("Authentication menu has an empty choice.");
        }
        try {
            return choice.strategy->login(config_dir);
        } catch (const LoginCancelled&) {
            throw;
        } catch (const std::exception& error) {
            ui.show_error(error.what());
            ui.show_instructions("Choose a method to try again.");
        }
    }
}

} // namespace core::auth
