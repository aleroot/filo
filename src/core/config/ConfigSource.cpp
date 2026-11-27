#include "ConfigSource.hpp"
#include "ConfigDocument.hpp"
#include "core/auth/FileTokenStore.hpp"
#include "core/auth/IdTokenClaims.hpp"
#include "core/portability/ConfigPortability.hpp"
#include "core/utils/Base64.hpp"
#include "core/utils/Crypto.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/PathUtils.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <pwd.h>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace core::config {
namespace {
namespace fs = std::filesystem;
using core::auth::FileTokenStore;
using core::utils::JsonValue;
using core::portability::ConfigOverlayFiles;

fs::path credential_marker(const fs::path& directory, std::string_view id, std::string_view suffix) {
    return directory / (FileTokenStore::file_name(id) + std::string(suffix));
}

/// Marks a login this state signed in or out on its own.
fs::path selected_marker(const fs::path& directory, std::string_view id) {
    return credential_marker(directory, id, ".selected");
}

/// Marks a login this state hands to the fallback store; see unify_logins.
fs::path inherited_marker(const fs::path& directory, std::string_view id) {
    return credential_marker(directory, id, ".inherited");
}

void write_private(const fs::path& path, std::string_view text) {
    if (fs::is_symlink(fs::symlink_status(path))) {
        throw std::runtime_error("symlink in configuration state: " + path.string());
    }
    std::string error;
    if (!core::utils::atomic_write_file(path, text, &error, core::utils::kPrivateFilePermissions)) {
        throw std::runtime_error(error);
    }
}

std::string read_regular_file(const fs::path& path) {
    if (!fs::is_regular_file(path)) throw std::runtime_error("configuration input must be a regular file: " + path.string());
    const auto text = core::utils::read_file(path);
    if (!text) throw std::runtime_error("cannot read configuration input: " + path.string());
    return *text;
}

void private_directory(const fs::path& path) {
    if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) {
        throw std::runtime_error("cannot create configuration state directory: " + path.string());
    }
    struct stat status{};
    if (::lstat(path.c_str(), &status) != 0 || !S_ISDIR(status.st_mode) || status.st_uid != ::getuid()) {
        throw std::runtime_error("configuration state must be an owned directory, not a symlink: " + path.string());
    }
    fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace);
}

fs::path state_home() {
    if (const char* xdg = std::getenv("XDG_STATE_HOME"); xdg && fs::path(xdg).is_absolute()) return xdg;
    if (const auto home = core::utils::path::home_directory(); !home.empty()) return home / ".local/state";
    if (const auto* user = ::getpwuid(::getuid()); user && user->pw_dir && user->pw_dir[0] != '\0') {
        return fs::path(user->pw_dir) / ".local/state";
    }
    const auto root = fs::temp_directory_path() / ("filo-" + std::to_string(::getuid()));
    private_directory(root);
    return root / "state";
}

ConfigOverlayFiles read_input(const fs::path& path) {
    if (fs::is_regular_file(path)) {
        const auto bundle = core::portability::decode_bundle(read_regular_file(path));
        if (!bundle) throw std::runtime_error(bundle.error());
        return core::portability::config_overlay_files(*bundle);
    }
    if (!fs::is_directory(path)) throw std::runtime_error("configuration must be a bundle file or directory");
    ConfigOverlayFiles files;
    for (const auto name : {"config.json", "auth_defaults.json", "settings.json", "mcp_servers.json",
                            "profile_defaults.json", "model_defaults.json"}) {
        if (auto document = ConfigSource::read_document(path / name)) files.configuration.emplace(name, std::move(*document));
    }
    std::set<std::string> stores;
    for (const auto& entry : fs::directory_iterator(path)) {
        const auto name = entry.path().filename().string();
        const auto id = FileTokenStore::provider_id_from_file_name(name);
        if (id && FileTokenStore::is_valid_provider_id(*id)) {
            if (auto document = ConfigSource::read_document(entry.path())) files.credentials.emplace(name, std::move(*document));
            stores.insert(*id);
        }
    }
    // The client identity of a login travels with its token, exactly as the
    // export carries it: the Antigravity client pair, the Codex installation
    // id, the MiMo key name.
    for (const auto& store : stores) {
        for (const auto& name : core::portability::credential_companion_files(store)) {
            const fs::path file = path / name;
            if (!fs::exists(fs::symlink_status(file))) continue;
            if (name.ends_with(".json")) {
                if (auto document = ConfigSource::read_document(file)) {
                    files.credentials.emplace(name, std::move(*document));
                }
            } else {
                files.credentials.emplace(name, read_regular_file(file));
            }
        }
    }
    // An identity whose login is not here travels on its own: it is what
    // makes the next login possible.
    for (const auto& entry : fs::directory_iterator(path)) {
        const auto name = entry.path().filename().string();
        const auto store = core::portability::store_for_companion(name);
        if (!store || stores.contains(*store)) continue;
        if (!fs::is_regular_file(entry.path())) continue;
        if (name.ends_with(".json")) {
            if (auto document = ConfigSource::read_document(entry.path())) {
                files.credentials.emplace(name, std::move(*document));
            }
        } else {
            files.credentials.emplace(name, read_regular_file(entry.path()));
        }
    }
    // These non-token companions share the token's lifecycle and state store.
    for (const auto& name : core::portability::credential_companion_files("kimi")) {
        if (name.ends_with(".json") || !fs::exists(fs::symlink_status(path / name))) continue;
        files.credentials.emplace(name, read_regular_file(path / name));
    }
    return files;
}

JsonValue canonical_json(const JsonValue& value) {
    if (value.is_object()) {
        auto fields = value.members();
        std::ranges::sort(fields, {}, &JsonValue::Member::first);
        auto result = JsonValue::object();
        for (const auto& [name, content] : fields) result.set(name, canonical_json(content));
        return result;
    }
    if (value.is_array()) {
        auto result = JsonValue::array();
        for (const auto& item : value.items()) result.items().push_back(canonical_json(item));
        return result;
    }
    return value;
}

std::vector<std::string> credential_files(std::string_view id) {
    auto names = core::portability::credential_companion_files(id);
    names.push_back(FileTokenStore::file_name(id));
    return names;
}

ConfigSource::Documents credential_groups(const ConfigSource::Documents& files) {
    // A token and its companions must be replaced together under its refresh
    // lock. Infer ownership using the same file list used by export/import.
    std::map<std::string, std::string> owners;
    for (const auto& [name, _] : files) {
        if (const auto id = FileTokenStore::provider_id_from_file_name(name)) owners.emplace(name, *id);
    }
    const auto tokens = owners;
    for (const auto& [_, id] : tokens) {
        for (const auto& name : core::portability::credential_companion_files(id)) owners[name] = id;
    }
    // A client identity may travel without its login.
    for (const auto& [name, _] : files) {
        if (owners.contains(name)) continue;
        if (const auto store = core::portability::store_for_companion(name)) owners[name] = *store;
    }
    for (const auto& name : core::portability::credential_companion_files("kimi")) owners[name] = "kimi";
    ConfigSource::Documents groups;
    for (const auto& [name, content] : files) {
        const auto owner = owners.find(name);
        if (owner == owners.end()) {
            throw std::runtime_error("credential file without an owner: " + name);
        }
        auto [group, _] = groups.try_emplace(owner->second, JsonValue::object());
        group->second.set(name, content);
    }
    return groups;
}

void reconcile_credentials(const fs::path& directory, const ConfigSource::Documents& files) {
    const auto metadata_path = directory / ".credential-sources.json";
    auto previous = ConfigSource::read_document(metadata_path).value_or(JsonValue::object());
    const auto groups = credential_groups(files);
    std::map<std::string, std::optional<std::string>> changed;
    for (const auto& [id, fingerprint] : previous.members()) {
        if (!FileTokenStore::is_valid_provider_id(id) || !fingerprint.is_string()) {
            throw std::runtime_error("invalid credential source metadata: " + metadata_path.string());
        }
        if (!groups.contains(id)) changed.emplace(id, std::nullopt);
    }
    for (const auto& [id, group] : groups) {
        const auto fingerprint = core::utils::Base64::encode_url(
            core::utils::crypto::sha256(canonical_json(group).dump()));
        const auto* old = previous.find(id);
        if (!old || old->as_string() != fingerprint) changed.emplace(id, fingerprint);
    }
    // The initialization lock serializes source changes; native token locks
    // also serialize token replacement with refresh and credential commits
    // in already-running instances.
    std::vector<std::unique_ptr<core::auth::ITokenStoreLock>> locks;
    for (const auto& [id, _] : changed) locks.push_back(FileTokenStore(directory.string()).acquire_refresh_lock(id));
    const auto replace = [&](const std::string& id, const JsonValue* group) {
        const auto token_path = directory / FileTokenStore::file_name(id);
        const auto marker = selected_marker(directory, id);
        if (group) write_private(marker, "");
        fs::remove(inherited_marker(directory, id));
        // Publish the token last. A failed or interrupted companion update
        // leaves this account unavailable rather than mixing old credentials
        // with the replacement's device or client metadata.
        fs::remove(token_path);
        for (const auto& name : credential_files(id)) {
            if (const auto* content = group ? group->find(name) : nullptr) {
                write_private(directory / name, content->is_string() ? std::string(content->as_string()) : content->dump() + '\n');
            } else {
                fs::remove(directory / name);
            }
        }
        if (!group) fs::remove(marker);
    };
    // Remove old groups first: a previously standalone companion can become
    // part of a newly supplied token group.
    for (const auto& [id, fingerprint] : changed) {
        if (fingerprint) continue;
        replace(id, nullptr);
        previous.erase(id);
        write_private(metadata_path, previous.dump() + '\n');
    }
    for (const auto& [id, fingerprint] : changed) {
        if (!fingerprint) continue;
        replace(id, &groups.at(id));
        previous.set(id, *fingerprint);
        // Commit each credential separately so a later failure cannot reset
        // refreshes or logouts of credentials already reconciled.
        write_private(metadata_path, previous.dump() + '\n');
    }
}

/// Two tokens are one login when they share a refresh token, or when an
/// account identity both carry agrees and nothing that both carry disagrees.
/// Unknown is never "same": a login that cannot be proven shared stays put.
bool same_login(const core::auth::OAuthToken& a, const core::auth::OAuthToken& b) {
    if (a.access_token.empty() || b.access_token.empty()) return false;
    if (!a.refresh_token.empty() && a.refresh_token == b.refresh_token) return true;
    const auto conflict = [](std::string_view x, std::string_view y) {
        return !x.empty() && !y.empty() && x != y;
    };
    if (conflict(a.issuer, b.issuer) || conflict(a.client_id, b.client_id)
        || conflict(a.organization_id, b.organization_id) || conflict(a.team_id, b.team_id)
        || conflict(a.principal_id, b.principal_id)) {
        return false;
    }
    const auto subject = [](const core::auth::OAuthToken& token) {
        return core::auth::parse_id_token_claims(token.access_token).subject;
    };
    const std::array<std::pair<std::string, std::string>, 4> identities{{
        {a.account_id, b.account_id}, {a.user_id, b.user_id}, {a.email, b.email},
        {subject(a), subject(b)}}};
    bool proven = false;
    for (const auto& [x, y] : identities) {
        if (x.empty() || y.empty()) continue;
        if (x != y) return false;
        proven = true;
    }
    return proven;
}

/// One login lives in exactly one store.
///
/// Providers rotate refresh tokens on every refresh (OpenAI answers a second
/// use with `refresh_token_reused`), so two stores holding the same login
/// cannot both survive: whichever refreshes first silently kills the other.
/// Earlier versions seeded config state by copying the fallback login, and a
/// bundle exported from this machine carries a copy too. Every such duplicate
/// is folded back into the fallback, keeping the most recently issued token
/// (the only one of a rotation chain that can still be alive), and the state
/// then reads that store in place.
void unify_logins(const fs::path& state, const fs::path& fallback) {
    std::error_code ec;
    if (!fs::is_directory(fallback, ec) || fs::equivalent(state, fallback, ec)) return;
    std::vector<std::string> ids;
    for (const auto& entry : fs::directory_iterator(state)) {
        if (!entry.is_regular_file()) continue;
        const auto id = FileTokenStore::provider_id_from_file_name(entry.path().filename().string());
        if (id && FileTokenStore::is_valid_provider_id(*id)) ids.push_back(*id);
    }
    for (const auto& id : ids) {
        const auto name = FileTokenStore::file_name(id);
        // Never create the fallback: an absent login has nothing to share.
        if (!fs::exists(fallback / name)) continue;
        FileTokenStore here(state.string());
        FileTokenStore there(fallback.string());
        // Compare before locking: a lock file is a write, and an unrelated
        // login must leave the fallback untouched.
        const auto shared = [&] {
            const auto mine = here.load(id);
            const auto theirs = there.load(id);
            return mine && theirs && same_login(*mine, *theirs);
        };
        if (!shared()) continue;
        // Same order everywhere: state, then fallback.
        const auto here_lock = here.acquire_refresh_lock(id);
        const auto there_lock = there.acquire_refresh_lock(id);
        if (!shared()) continue;
        const auto mine = here.load(id);
        const auto theirs = there.load(id);
        const auto companions = core::portability::credential_companion_files(id);
        if (mine->expires_at > theirs->expires_at) {
            // Companions first: a device id must never pair with another token.
            for (const auto& companion : companions) {
                if (fs::exists(state / companion)) {
                    write_private(fallback / companion, read_regular_file(state / companion));
                }
            }
            write_private(fallback / name, read_regular_file(state / name));
        }
        write_private(inherited_marker(state, id), "");
        fs::remove(state / name);
        fs::remove(selected_marker(state, id));
        for (const auto& companion : companions) fs::remove(state / companion);
    }
}

ConfigOverlayFiles read_validated_input(const fs::path& input) {
    auto files = read_input(input);
    for (const auto& [_, document] : files.configuration) ConfigDocument::validate(document);
    return files;
}

fs::path initialize_state(const fs::path& input, const fs::path& fallback, ConfigOverlayFiles& files) {
    // The canonical source path is the entire identity of mutable state.
    const auto key = core::utils::Base64::encode_url(core::utils::crypto::sha256(input.string()));
    const auto home = fs::weakly_canonical(fs::absolute(state_home()));
    fs::create_directories(home);
    private_directory(home / "filo");
    private_directory(home / "filo/configs");
    const auto directory = home / "filo/configs" / key;
    private_directory(directory);
    std::string error;
    const auto lock_path = directory / ".init.lock";
    if (fs::is_symlink(fs::symlink_status(lock_path))) throw std::runtime_error("symlink in configuration state lock");
    const auto lock = core::utils::InterprocessFileLock::acquire(lock_path, &error);
    if (!lock) throw std::runtime_error(error);
    // Re-read after locking so a waiting opener cannot restore an older source
    // snapshot over credentials installed by another process.
    files = read_validated_input(input);
    reconcile_credentials(directory, files.credentials);
    unify_logins(directory, fallback);
    return directory;
}
} // namespace

ConfigSource::ConfigSource(fs::path input, fs::path state, fs::path fallback, Documents documents,
                           std::vector<std::string> credential_files)
    : input_(std::move(input)), state_(std::move(state)), fallback_(std::move(fallback)),
      documents_(std::move(documents)), credential_files_(std::move(credential_files)),
      directory_input_(fs::is_directory(input_)) {}

std::shared_ptr<const ConfigSource> ConfigSource::open(const fs::path& input, const fs::path& fallback_directory) {
    const auto text = input.string();
    const auto home = core::utils::path::home_directory();
    const auto expanded = !home.empty() && (text == "~" || text.starts_with("~/"))
        ? (text == "~" ? home : home / text.substr(2)) : input;
    if (expanded.empty() || !fs::exists(expanded)) {
        throw std::runtime_error("configuration path does not exist: " + expanded.string());
    }
    const auto path = fs::canonical(expanded);
    const auto fallback = fs::weakly_canonical(fallback_directory);
    // Reject invalid input before creating any runtime state.
    auto files = read_validated_input(path);
    const auto state = initialize_state(path, fallback, files);
    std::vector<std::string> credential_files;
    for (const auto& [name, _] : files.credentials) credential_files.push_back(name);
    return std::shared_ptr<const ConfigSource>(new ConfigSource(
        path, state, fallback, std::move(files.configuration), std::move(credential_files)));
}

std::optional<JsonValue> ConfigSource::read_document(const fs::path& path) {
    if (!fs::exists(fs::symlink_status(path))) return std::nullopt;
    std::string error;
    auto value = JsonValue::parse(read_regular_file(path), &error);
    if (!value) throw std::runtime_error(path.string() + ": " + error);
    if (!value->is_object()) throw std::runtime_error(path.string() + ": expected a configuration object");
    return value;
}

void ConfigSource::verify_available() const {
    if (!(directory_input_ ? fs::is_directory(input_) : fs::is_regular_file(input_))) {
        throw std::runtime_error("configuration source disappeared: " + input_.string());
    }
    if (!fs::is_directory(state_)) throw std::runtime_error("configuration state directory disappeared: " + state_.string());
}

fs::path ConfigSource::workspace_state_directory(const fs::path& workspace) const {
    const auto path = fs::weakly_canonical(fs::absolute(workspace)).string();
    return state_ / "workspaces" / core::utils::Base64::encode_url(core::utils::crypto::sha256(path));
}

fs::path ConfigSource::resolve_file(std::string_view name) const {
    verify_available();
    if (fs::exists(state_ / name)) return state_ / name;
    if (directory_input_ && fs::exists(input_ / name)) return input_ / name;
    return fallback_ / name;
}

void ConfigSource::claim_credential(std::string_view id) const {
    verify_available();
    if (!FileTokenStore::is_valid_provider_id(id)) throw std::runtime_error("invalid credential id");
    const auto marker = selected_marker(state_, id);
    if (!fs::exists(marker)) write_private(marker, "");
    fs::remove(inherited_marker(state_, id));
}

fs::path ConfigSource::credential_directory(std::string_view id) const {
    verify_available();
    if (!FileTokenStore::is_valid_provider_id(id)) throw std::runtime_error("invalid credential id");
    const auto name = FileTokenStore::file_name(id);
    // A login handed to the fallback is read there until this config signs
    // in or out on its own.
    if (fs::exists(inherited_marker(state_, id)) && !fs::exists(state_ / name)) return fallback_;
    // Inherit the fallback login in place, refreshes included. Never copy it:
    // a copied refresh token is a second holder of a single-use secret.
    if (!owns_credential(id)) return fs::exists(fallback_ / name) ? fallback_ : state_;
    if (fs::exists(state_ / name)) claim_credential(id);
    return state_;
}

fs::path ConfigSource::login_directory(std::string_view id) const {
    if (credential_directory(id) == fallback_) return fallback_;
    // A login this config neither supplies nor signed in or out of on its
    // own belongs to the fallback, even when the fallback has none yet.
    return owns_credential(id) ? state_ : fallback_;
}

bool ConfigSource::owns_credential(std::string_view id) const {
    const auto name = FileTokenStore::file_name(id);
    const auto supplied = [&](std::string_view file) { return std::ranges::contains(credential_files_, file); };
    const auto companions = core::portability::credential_companion_files(id);
    return supplied(name) || std::ranges::any_of(companions, supplied) || fs::exists(state_ / name)
        || fs::exists(selected_marker(state_, id));
}

} // namespace core::config
