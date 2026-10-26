#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/cli/ConfigTransfer.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/memory/MemoryStore.hpp"
#include "core/portability/ConfigBundle.hpp"
#include "core/portability/ConfigPortability.hpp"
#include "core/utils/Gzip.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/JsonValue.hpp"
#include "core/utils/PathUtils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

namespace {

namespace fs = std::filesystem;
using core::portability::Bundle;
using core::portability::BundleItem;
using core::portability::ExportOptions;
using core::portability::ImportAction;
using core::portability::ImportOptions;
using core::portability::ItemFilter;
using core::portability::ItemKind;
using core::portability::PortabilityPaths;
using core::utils::JsonValue;

struct TempDir {
    fs::path path;
    explicit TempDir(std::string name) : path(fs::temp_directory_path() / std::move(name)) {
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

/// One simulated machine: a home directory with Filo's config dir inside.
struct Machine {
    TempDir root;
    PortabilityPaths paths;

    explicit Machine(std::string name) : root(std::move(name)) {
        paths.home = (root.path / "home").string();
        paths.config_dir = root.path / "home" / ".config" / "filo";
        paths.memory_file = paths.config_dir / "memory.json";
        fs::create_directories(paths.config_dir);
    }

    void write(std::string_view file, std::string_view content) const {
        std::ofstream(paths.config_dir / file, std::ios::binary) << content;
    }

    [[nodiscard]] std::string read(std::string_view file) const {
        std::ifstream in(paths.config_dir / file, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), {}};
    }

    [[nodiscard]] JsonValue json(std::string_view file) const {
        auto value = JsonValue::parse(read(file));
        REQUIRE(value.has_value());
        return *value;
    }

    [[nodiscard]] std::string project(std::string_view name) const {
        return paths.home + "/src/" + std::string(name);
    }
};

[[nodiscard]] ItemFilter filter(std::vector<std::string> only, std::vector<std::string> except = {}) {
    auto parsed = ItemFilter::parse(only, except);
    REQUIRE(parsed.has_value());
    return *parsed;
}

[[nodiscard]] const BundleItem* find_item(const Bundle& bundle, ItemKind kind, std::string_view name) {
    const auto it = std::ranges::find_if(bundle.items, [&](const BundleItem& item) {
        return item.kind == kind && item.name == name;
    });
    return it == bundle.items.end() ? nullptr : &*it;
}

void seed_source(const Machine& machine) {
    machine.write("config.json", R"({
  "default_provider": "claude",
  "auto_compact_threshold": 80,
  "providers": {
    "claude": {"model": "claude-sonnet-4-6"},
    "qwen-token-plan": {"model": "qwen3.8-max", "api_key": "sk-config-secret"}
  },
  "mcp_servers": [
    {"name": "linear", "transport": "http", "url": "https://mcp.linear.app/mcp",
     "headers": {"Authorization": "Bearer mcp-secret"}}
  ]
})");
    machine.write("auth_defaults.json", R"({
  "default_provider": "claude",
  "providers": {
    "claude": {"model": "claude-sonnet-4-6", "auth_type": "oauth_claude"},
    "qwen-token-plan": {"api_key": "sk-overlay-secret"}
  }
})");
    machine.write("oauth_claude.json", R"({"access_token": "claude-access", "refresh_token": "claude-refresh", "expires_at": 1893456000})");
    machine.write("oauth_kimi.json", R"({"access_token": "kimi-access", "device_id": "dev-1"})");
    machine.write("kimi_device_id", "dev-1");
    machine.write("oauth_claude.json.lock", "");
    // An MCP login refreshes through endpoints + client registration kept in a
    // second token-store file, which must travel with it.
    machine.write("oauth_mcp_linear.json", R"({"access_token": "linear-access"})");
    machine.write("oauth_mcp_linear_meta.json",
                  R"({"token_endpoint": "https://mcp.linear.app/token", "client_id": "dcr-1"})");

    const core::memory::MemoryStore store(machine.paths.memory_file);
    auto state = store.load();
    state.entries.push_back({.id = "m1", .content = "Filo builds with ninja.", .scope = "project",
                             .source = "agent", .project_root = machine.project("filo")});
    state.entries.push_back({.id = "m2", .content = "Prefer short replies.", .scope = "global"});
    state.entries.push_back({.id = "m3", .content = "Scratch note.", .scope = "session",
                             .project_root = machine.project("filo"), .session_id = "s1"});
    state.entries.push_back({.id = "m4", .content = "Old fact.", .scope = "project",
                             .archived = true, .project_root = machine.project("filo")});
    REQUIRE(store.save(state));
}

[[nodiscard]] Bundle round_trip(const Bundle& bundle) {
    auto parsed = core::portability::parse_bundle(core::portability::serialize_bundle(bundle));
    REQUIRE(parsed.has_value());
    return *parsed;
}

} // namespace

TEST_CASE("JsonValue keeps member order, number values and escapes", "[portability][json]") {
    const auto value = JsonValue::parse(R"({"b": 1.50, "a": [true, null, "x\"y"], "c": {}})");
    REQUIRE(value.has_value());
    CHECK(value->dump(0) == R"({"b":1.5,"a":[true,null,"x\"y"],"c":{}})");

    JsonValue copy = *value;
    copy.set("a", "replaced");
    copy.set("d", JsonValue::integer(-7));
    CHECK(copy.erase("c"));
    CHECK_FALSE(copy.erase("missing"));
    CHECK(copy.dump(0) == R"({"b":1.5,"a":"replaced","d":-7})");
    CHECK(copy.find("d")->as_integer() == -7);
    CHECK(JsonValue::parse(copy.dump(2)) == copy);
    CHECK_FALSE(JsonValue::parse("{broken").has_value());
    // Duplicate keys resolve like simdjson lookups (the rest of Filo): first wins.
    CHECK(JsonValue::parse(R"({"k": 1, "j": 2, "k": 3})")->dump(0) == R"({"k":1,"j":2})");
}

TEST_CASE("JsonValue resolves and stores RFC 6901 pointers", "[portability][json]") {
    auto doc = *JsonValue::parse(R"({"a/b": {"m~n": [10, 20]}, "": 1})");
    const std::string escaped = "/" + JsonValue::pointer_token("a/b") + "/" + JsonValue::pointer_token("m~n");
    CHECK(escaped == "/a~1b/m~0n");
    CHECK(doc.at_pointer(escaped + "/1")->as_integer() == 20);
    CHECK(doc.at_pointer("/")->as_integer() == 1);       // the empty key
    CHECK(doc.at_pointer("") == &doc);                   // the whole document
    CHECK(doc.at_pointer(escaped + "/2") == nullptr);
    CHECK(doc.at_pointer("no-slash") == nullptr);
    CHECK(doc.at_pointer("/a~2b") == nullptr);           // invalid escape

    REQUIRE(doc.set_at_pointer("/x/y~1z", "v") != nullptr);
    CHECK(doc.find("x")->find("y/z")->as_string() == "v");
    CHECK(doc.set_at_pointer("bad", "v") == nullptr);
}

TEST_CASE("User paths fold into and out of a given home", "[portability][paths]") {
    using namespace core::utils::path;
    CHECK(relative_within("/home/al/src", "/home/al") == fs::path("src"));
    CHECK(relative_within("/home/al", "/home/al/") == fs::path("."));
    CHECK_FALSE(relative_within("/home/alice", "/home/al").has_value());
    CHECK_FALSE(relative_within("/home/al/../bob", "/home/al").has_value());
    CHECK_FALSE(relative_within("relative", "/home/al").has_value());

    CHECK(abbreviate_user_path("/home/al/src/filo", "/home/al/") == "~/src/filo");
    CHECK(abbreviate_user_path("/home/al", "/home/al") == "~");
    CHECK(abbreviate_user_path("/srv/x", "/home/al") == "/srv/x");
    CHECK(expand_user_path("~/src", "/home/bo") == fs::path("/home/bo/src"));
    CHECK(expand_user_path("~other/src", "/home/bo") == fs::path("~other/src"));
}

TEST_CASE("Selectors accept kinds, kind:pattern and bare names", "[portability][selector]") {
    using core::portability::parse_selector;

    const auto kind_only = parse_selector("Providers");
    REQUIRE(kind_only.has_value());
    CHECK(kind_only->kind == ItemKind::Provider);
    CHECK(kind_only->matches(ItemKind::Provider, "anything"));
    CHECK_FALSE(kind_only->matches(ItemKind::Credential, "anything"));

    const auto singular = parse_selector("memory:~/src/*");
    REQUIRE(singular.has_value());
    CHECK(singular->kind == ItemKind::Memory);
    CHECK(singular->matches(ItemKind::Memory, "~/src/filo/sub"));
    CHECK_FALSE(singular->matches(ItemKind::Memory, "global"));

    const auto bare = parse_selector("claude*");
    REQUIRE(bare.has_value());
    CHECK_FALSE(bare->kind.has_value());
    CHECK(bare->matches(ItemKind::Provider, "claude-thinking"));
    CHECK(bare->matches(ItemKind::Credential, "claude"));

    const auto unknown = parse_selector("providerz:claude");
    REQUIRE_FALSE(unknown.has_value());
    CHECK_THAT(unknown.error(), Catch::Matchers::ContainsSubstring("providers"));

    const auto combined = filter({"providers", "credentials"}, {"provider:qwen*"});
    CHECK(combined.matches(ItemKind::Provider, "claude"));
    CHECK_FALSE(combined.matches(ItemKind::Provider, "qwen-token-plan"));
    CHECK_FALSE(combined.matches(ItemKind::Memory, "global"));

    CHECK(parse_selector("a?c*")->matches(ItemKind::Setting, "abcdef"));
    CHECK_FALSE(parse_selector("a?c")->matches(ItemKind::Setting, "abcd"));
}

TEST_CASE("Export withholds secrets unless asked", "[portability][export]") {
    Machine source("filo_portability_export");
    seed_source(source);

    const auto result = core::portability::export_bundle(source.paths, {});
    const auto& bundle = result.bundle;
    const std::string text = core::portability::serialize_bundle(bundle);

    for (const auto secret : {"sk-config-secret", "sk-overlay-secret", "mcp-secret",
                              "claude-access", "kimi-access"}) {
        CHECK_THAT(text, !Catch::Matchers::ContainsSubstring(secret));
    }
    CHECK_FALSE(bundle.contains_secrets());
    CHECK(result.withheld == std::vector<std::string>{
        "credentials:claude", "credentials:kimi", "credentials:mcp_linear"});

    const auto* qwen = find_item(bundle, ItemKind::Provider, "qwen-token-plan");
    REQUIRE(qwen != nullptr);
    CHECK(qwen->redacted == std::vector<std::string>{
        "/config.json/api_key", "/auth_defaults.json/api_key"});
    CHECK(qwen->data.find("config.json")->find("model")->as_string() == "qwen3.8-max");

    const auto* linear = find_item(bundle, ItemKind::Mcp, "linear");
    REQUIRE(linear != nullptr);
    CHECK(linear->redacted == std::vector<std::string>{"/config.json/headers"});

    const auto* setting = find_item(bundle, ItemKind::Setting, "default_provider");
    REQUIRE(setting != nullptr);
    CHECK(setting->data.members().size() == 2);  // config.json + auth_defaults.json
    CHECK(find_item(bundle, ItemKind::Setting, "providers") == nullptr);

    // Project memories are stored relative to home; archived and session
    // memories stay behind.
    const auto* project = find_item(bundle, ItemKind::Memory, "~/src/filo");
    REQUIRE(project != nullptr);
    REQUIRE(project->data.find("entries")->items().size() == 1);
    CHECK(project->data.find("entries")->items()[0].find("project_root")->as_string() == "~/src/filo");
    REQUIRE(find_item(bundle, ItemKind::Memory, "global") != nullptr);
    CHECK_THAT(text, !Catch::Matchers::ContainsSubstring("Scratch note."));
    CHECK_THAT(text, !Catch::Matchers::ContainsSubstring("Old fact."));

    // Kinds come out in table order.
    CHECK(std::ranges::is_sorted(bundle.items, {}, [](const BundleItem& item) {
        return static_cast<int>(item.kind);
    }));
}

TEST_CASE("A full export moves logins, providers and memories to another machine",
          "[portability][import]") {
    Machine source("filo_portability_source");
    Machine target("filo_portability_target");
    seed_source(source);

    const auto exported = core::portability::export_bundle(
        source.paths, {.include_secrets = true});
    CHECK(exported.withheld.empty());
    const auto bundle = round_trip(exported.bundle);
    CHECK(bundle.contains_secrets());

    const auto result = core::portability::import_bundle(bundle, target.paths, {});
    REQUIRE(result.ok());
    CHECK_FALSE(result.backup.has_value());  // nothing pre-existed

    CHECK(target.json("oauth_claude.json").find("refresh_token")->as_string() == "claude-refresh");
    CHECK(target.read("kimi_device_id") == "dev-1");
    CHECK(target.json("oauth_mcp_linear_meta.json").find("client_id")->as_string() == "dcr-1");
    CHECK(find_item(bundle, ItemKind::Credential, "mcp_linear_meta") == nullptr);
    CHECK((fs::status(target.paths.config_dir / "oauth_claude.json").permissions()
           & fs::perms::group_all) == fs::perms::none);
    const auto config = target.json("config.json");
    CHECK(config.find("providers")->find("qwen-token-plan")->find("api_key")->as_string()
          == "sk-config-secret");
    CHECK(config.find("mcp_servers")->items().front().find("headers") != nullptr);
    CHECK(target.json("auth_defaults.json").find("providers")->find("claude")
              ->find("auth_type")->as_string() == "oauth_claude");

    // Memory follows the home directory, without --map-path.
    const auto memories = core::memory::MemoryStore(target.paths.memory_file).list();
    REQUIRE(memories.size() == 2);
    CHECK(std::ranges::any_of(memories, [&](const auto& entry) {
        return entry.content == "Filo builds with ninja." && entry.project_root == target.project("filo")
            && entry.source == "agent";
    }));

    // Importing the same bundle again changes nothing and needs no backup.
    const auto again = core::portability::import_bundle(bundle, target.paths, {});
    REQUIRE(again.ok());
    CHECK(std::ranges::all_of(again.items, [](const auto& item) {
        return item.action == ImportAction::Unchanged;
    }));
    CHECK_FALSE(again.backup.has_value());
    CHECK(core::memory::MemoryStore(target.paths.memory_file).list().size() == 2);
}

TEST_CASE("Importing a redacted provider keeps this machine's API key", "[portability][import]") {
    Machine source("filo_portability_redacted_source");
    Machine target("filo_portability_redacted_target");
    seed_source(source);
    target.write("config.json", R"({"providers": {"qwen-token-plan": {"model": "old", "api_key": "sk-local"}}})");

    const auto bundle = round_trip(core::portability::export_bundle(
        source.paths, {.filter = filter({"provider:qwen*"})}).bundle);
    const auto result = core::portability::import_bundle(bundle, target.paths, {});
    REQUIRE(result.ok());
    REQUIRE(result.items.size() == 1);
    CHECK(result.items[0].action == ImportAction::Updated);
    // auth_defaults.json had no key on this machine; that is reported.
    CHECK_THAT(result.items[0].detail, Catch::Matchers::ContainsSubstring("auth_defaults.json/api_key"));

    const auto config = target.json("config.json");
    const auto* provider = config.find("providers")->find("qwen-token-plan");
    CHECK(provider->find("model")->as_string() == "qwen3.8-max");
    CHECK(provider->find("api_key")->as_string() == "sk-local");

    // The previous definition was saved and can be restored.
    REQUIRE(result.backup.has_value());
    CHECK((fs::status(*result.backup).permissions() & fs::perms::others_all) == fs::perms::none);
    std::ifstream backup_in(*result.backup);
    const auto backup = core::portability::parse_bundle(
        std::string(std::istreambuf_iterator<char>(backup_in), {}));
    REQUIRE(backup.has_value());
    REQUIRE(core::portability::import_bundle(*backup, target.paths, {}).ok());
    CHECK(target.json("config.json").find("providers")->find("qwen-token-plan")
              ->find("model")->as_string() == "old");
}

TEST_CASE("Import filters select items by name across kinds", "[portability][import]") {
    Machine source("filo_portability_filter_source");
    Machine target("filo_portability_filter_target");
    seed_source(source);
    const auto bundle = core::portability::export_bundle(
        source.paths, {.include_secrets = true}).bundle;

    const auto result = core::portability::import_bundle(
        bundle, target.paths, {.filter = filter({"claude"})});
    REQUIRE(result.ok());
    REQUIRE(result.items.size() == 2);
    CHECK(result.items[0].kind == ItemKind::Provider);
    CHECK(result.items[1].kind == ItemKind::Credential);
    CHECK(fs::exists(target.paths.config_dir / "oauth_claude.json"));
    CHECK_FALSE(fs::exists(target.paths.config_dir / "oauth_kimi.json"));
    CHECK_FALSE(fs::exists(target.paths.memory_file));
    // A config.json created by the import starts from the shipped defaults,
    // and settings were not selected, so the default provider is the shipped one.
    const auto defaults = JsonValue::parse(core::config::default_user_config_json());
    REQUIRE(defaults.has_value());
    const auto config = target.json("config.json");
    CHECK(*config.find("default_provider") == *defaults->find("default_provider"));
    CHECK(config.find("subagents") != nullptr);
    CHECK(config.find("providers")->members().size() >= defaults->find("providers")->members().size());
    CHECK(config.find("providers")->find("claude")->find("model")->as_string() == "claude-sonnet-4-6");
}

TEST_CASE("Dry-run import reports the plan and writes nothing", "[portability][import]") {
    Machine source("filo_portability_dry_source");
    Machine target("filo_portability_dry_target");
    seed_source(source);
    const auto bundle = core::portability::export_bundle(
        source.paths, {.include_secrets = true}).bundle;

    const auto result = core::portability::import_bundle(bundle, target.paths, {.dry_run = true});
    REQUIRE(result.ok());
    CHECK(std::ranges::none_of(result.items, [](const auto& item) {
        return item.action == ImportAction::Failed || item.action == ImportAction::Skipped;
    }));
    CHECK(std::ranges::any_of(result.items, [](const auto& item) {
        return item.kind == ItemKind::Credential && item.action == ImportAction::Created;
    }));
    // Only advisory lock files may appear.
    for (const auto& entry : fs::directory_iterator(target.paths.config_dir)) {
        CHECK(entry.path().extension() == ".lock");
    }
    CHECK_FALSE(fs::exists(target.paths.memory_file));
}

TEST_CASE("An untrusted bundle cannot write outside the token store", "[portability][import]") {
    Machine target("filo_portability_untrusted");
    Bundle bundle;
    bundle.items.push_back({.kind = ItemKind::Provider, .name = "claude",
                            .data = *JsonValue::parse(R"({"config.json": {"model": "x"}})")});
    bundle.items.push_back({.kind = ItemKind::Credential, .name = "../escape",
                            .data = *JsonValue::parse(R"({"oauth_../escape.json": {"access_token": "t"}})"),
                            .secret = true});
    bundle.items.push_back({.kind = ItemKind::Credential, .name = "claude",
                            .data = *JsonValue::parse(
                                R"({"oauth_claude.json": {"access_token": "t"}, "config.json": "{}"})"),
                            .secret = true});
    bundle.items.push_back({.kind = ItemKind::Setting, .name = "x",
                            .data = *JsonValue::parse(R"({"../../outside.json": 1})")});

    const auto result = core::portability::import_bundle(round_trip(bundle), target.paths, {});
    CHECK_FALSE(result.ok());
    CHECK_FALSE(result.write_started);
    REQUIRE(result.items.size() == 4);
    CHECK(result.items[0].action == ImportAction::Skipped);
    CHECK(result.items[1].action == ImportAction::Failed);
    CHECK(result.items[2].action == ImportAction::Failed);
    CHECK(result.items[3].action == ImportAction::Failed);
    // The valid provider was not applied either: validation is all-or-nothing.
    CHECK(fs::is_empty(target.paths.config_dir));
}

TEST_CASE("Memory import maps project paths and skips facts already present",
          "[portability][memory]") {
    Machine source("filo_portability_memory_source");
    Machine target("filo_portability_memory_target");
    const core::memory::MemoryStore source_store(source.paths.memory_file);
    auto state = source_store.load();
    state.entries.push_back({.id = "m1", .content = "Outside home.", .scope = "project",
                             .project_root = "/work/filo"});
    state.entries.push_back({.id = "m2", .content = "Inside home.", .scope = "project",
                             .project_root = source.project("filo")});
    REQUIRE(source_store.save(state));

    const core::memory::MemoryStore target_store(target.paths.memory_file);
    auto existing = target_store.load();
    existing.entries.push_back({.id = "m7", .content = "inside   HOME.", .scope = "project",
                                .project_root = target.paths.home + "/code/filo"});
    REQUIRE(target_store.save(existing));

    const auto bundle = round_trip(core::portability::export_bundle(
        source.paths, {.filter = filter({"memories"})}).bundle);
    REQUIRE(bundle.items.size() == 2);

    auto mapping_from_source = core::portability::parse_path_mapping(source.paths.home + "/src=~/code");
    auto mapping_absolute = core::portability::parse_path_mapping("/work=/srv/work");
    REQUIRE(mapping_from_source.has_value());
    REQUIRE(mapping_absolute.has_value());
    const auto result = core::portability::import_bundle(
        bundle, target.paths, {.path_mappings = {*mapping_from_source, *mapping_absolute}});
    REQUIRE(result.ok());

    const auto entries = target_store.list();
    REQUIRE(entries.size() == 2);  // "Inside home." matched the existing fact
    CHECK(std::ranges::any_of(entries, [](const auto& entry) {
        return entry.content == "Outside home." && entry.project_root == "/srv/work/filo"
            && entry.id == "m8";
    }));
    CHECK_FALSE(core::portability::parse_path_mapping("no-equals").has_value());
}

TEST_CASE("Memory import reports automatic notes archived to respect the limit",
          "[portability][memory]") {
    Machine target("filo_portability_memory_limit");
    const core::memory::MemoryStore store(target.paths.memory_file);
    auto state = store.load();
    state.settings.max_active_entries = 2;
    state.entries.push_back({.id = "m1", .content = "Local note.", .scope = "global",
                             .source = "agent"});
    REQUIRE(store.save(state));

    std::vector<core::memory::MemoryEntry> incoming{
        {.content = "Imported one.", .scope = "global", .source = "agent"},
        {.content = "Imported two.", .scope = "global", .source = "agent"},
    };
    const auto preview = store.import_entries(incoming, /*dry_run=*/true);
    CHECK(preview.added == 2);
    CHECK(preview.archived == 1);
    CHECK(store.list().size() == 1);  // dry run wrote nothing

    const auto outcome = store.import_entries(std::move(incoming));
    REQUIRE(outcome.ok);
    CHECK(outcome.archived == 1);
    CHECK(store.list().size() == 2);
}

TEST_CASE("Bundles from a newer Filo or another tool are rejected", "[portability][bundle]") {
    CHECK_FALSE(core::portability::parse_bundle(R"({"format": "other", "version": 1, "items": []})"));
    const auto newer = core::portability::parse_bundle(
        R"({"format": "filo-config-bundle", "version": 99, "items": []})");
    REQUIRE_FALSE(newer.has_value());
    CHECK_THAT(newer.error(), Catch::Matchers::ContainsSubstring("newer"));
    const auto unknown_kind = core::portability::parse_bundle(
        R"({"format": "filo-config-bundle", "version": 1, "items": [{"kind": "x", "name": "y", "data": {}}]})");
    REQUIRE_FALSE(unknown_kind.has_value());
    // A credential is a secret even when the file says otherwise.
    const auto credential = core::portability::parse_bundle(
        R"({"format": "filo-config-bundle", "version": 1, "items": [{"kind": "credential", "name": "y", "secret": false, "data": {}}]})");
    REQUIRE(credential.has_value());
    CHECK(credential->contains_secrets());
}

TEST_CASE("Bundles are stored as plain JSON or gzip and read back from either", "[portability][bundle][gzip]") {
    using core::portability::BundleEncoding;
    Machine source("filo_portability_encoding");
    seed_source(source);
    const auto bundle = core::portability::export_bundle(source.paths, {.include_secrets = true}).bundle;

    CHECK(core::portability::bundle_encoding_for("setup.json.gz") == BundleEncoding::Gzip);
    CHECK(core::portability::bundle_encoding_for("SETUP.JSON.GZ") == BundleEncoding::Gzip);
    CHECK(core::portability::bundle_encoding_for("setup.json") == BundleEncoding::Json);
    CHECK(core::portability::bundle_encoding_for("gz") == BundleEncoding::Json);

    const auto plain = core::portability::encode_bundle(bundle, BundleEncoding::Json);
    const auto packed = core::portability::encode_bundle(bundle, BundleEncoding::Gzip);
    CHECK(plain == core::portability::serialize_bundle(bundle));
    CHECK(core::portability::detect_bundle_encoding(plain) == BundleEncoding::Json);
    CHECK(core::portability::detect_bundle_encoding(packed) == BundleEncoding::Gzip);
    CHECK(packed.size() < plain.size());

    const auto from_plain = core::portability::decode_bundle(plain);
    const auto from_packed = core::portability::decode_bundle(packed);
    REQUIRE(from_plain.has_value());
    REQUIRE(from_packed.has_value());
    // Same document either way: re-serializing gives identical text.
    CHECK(core::portability::serialize_bundle(*from_packed) == plain);
    CHECK(core::portability::serialize_bundle(*from_plain) == plain);

    const auto damaged = core::portability::decode_bundle(packed.substr(0, packed.size() - 8));
    REQUIRE_FALSE(damaged.has_value());
    CHECK_THAT(damaged.error(), Catch::Matchers::ContainsSubstring("compressed bundle is damaged"));
    // A compressed document that is not a bundle fails like its plain form.
    const auto foreign = core::portability::decode_bundle(core::utils::gzip::compress(R"({"format":"x"})"));
    REQUIRE_FALSE(foreign.has_value());
    CHECK_THAT(foreign.error(), Catch::Matchers::ContainsSubstring("not a Filo configuration bundle"));
}

TEST_CASE("The export command compresses .gz targets and import reads them", "[portability][cli][gzip]") {
    Machine source("filo_portability_cli_gzip_source");
    Machine target("filo_portability_cli_gzip_target");
    seed_source(source);
    std::istringstream in;
    std::ostringstream out;
    std::ostringstream err;
    const auto archive = (source.root.path / "claude.json.gz").string();

    core::cli::ConfigTransferRequest request{
        .export_target = archive, .only = {"claude"}, .include_secrets = true};
    REQUIRE(core::cli::run_config_transfer(request, source.paths, in, out, err) == 0);
    CHECK_THAT(out.str(), Catch::Matchers::ContainsSubstring("gzip-compressed"));
    const auto bytes = core::utils::read_file(archive);
    REQUIRE(bytes.has_value());
    CHECK(core::utils::gzip::is_compressed(*bytes));
    CHECK((fs::status(archive).permissions() & (fs::perms::group_all | fs::perms::others_all))
          == fs::perms::none);

    // Import from the file and from standard input, as written.
    std::ostringstream imported;
    core::cli::ConfigTransferRequest from_file{.import_source = archive};
    REQUIRE(core::cli::run_config_transfer(from_file, target.paths, in, imported, err) == 0);
    CHECK(target.json("oauth_claude.json").find("refresh_token")->as_string() == "claude-refresh");
    CHECK(target.json("auth_defaults.json").find("providers")->find("claude")
              ->find("auth_type")->as_string() == "oauth_claude");
    std::istringstream piped(*bytes);
    std::ostringstream again;
    core::cli::ConfigTransferRequest from_stdin{.import_source = "-"};
    REQUIRE(core::cli::run_config_transfer(from_stdin, target.paths, piped, again, err) == 0);
    CHECK_THAT(again.str(), Catch::Matchers::ContainsSubstring("unchanged"));

    // --compress applies to any name and to stdout, but never to a terminal.
    std::ostringstream stdout_bytes;
    std::ostringstream report;
    core::cli::ConfigTransferRequest piped_export{.export_target = "-", .compress = true};
    REQUIRE(core::cli::run_config_transfer(piped_export, source.paths, in, stdout_bytes, report) == 0);
    CHECK(core::utils::gzip::is_compressed(stdout_bytes.str()));
    CHECK(core::portability::decode_bundle(stdout_bytes.str()).has_value());
    std::ostringstream terminal;
    piped_export.stdout_is_terminal = true;
    CHECK(core::cli::run_config_transfer(piped_export, source.paths, in, terminal, report) == 2);
    CHECK(terminal.str().empty());
    CHECK_THAT(report.str(), Catch::Matchers::ContainsSubstring("compressed bundle to the terminal"));

    // Without --compress or a .gz name the file stays plain JSON.
    const auto plain = (source.root.path / "setup.json").string();
    core::cli::ConfigTransferRequest plain_export{.export_target = plain};
    REQUIRE(core::cli::run_config_transfer(plain_export, source.paths, in, out, err) == 0);
    CHECK(core::utils::read_file(plain)->starts_with("{"));
}

TEST_CASE("The export command never prints credentials to a terminal", "[portability][cli]") {
    Machine source("filo_portability_cli");
    seed_source(source);
    std::istringstream in;
    std::ostringstream out;
    std::ostringstream err;

    core::cli::ConfigTransferRequest request{
        .export_target = "-", .include_secrets = true, .stdout_is_terminal = true};
    CHECK(core::cli::run_config_transfer(request, source.paths, in, out, err) == 2);
    CHECK(out.str().empty());

    request.stdout_is_terminal = false;
    err.str("");
    REQUIRE(core::cli::run_config_transfer(request, source.paths, in, out, err) == 0);
    CHECK_THAT(out.str(), Catch::Matchers::ContainsSubstring("claude-refresh"));
    CHECK_THAT(err.str(), Catch::Matchers::ContainsSubstring("Exported"));

    // Without secrets the report lists what was withheld.
    std::ostringstream listing;
    core::cli::ConfigTransferRequest dry{.export_target = "", .dry_run = true};
    REQUIRE(core::cli::run_config_transfer(dry, source.paths, in, listing, err) == 0);
    CHECK_THAT(listing.str(), Catch::Matchers::ContainsSubstring("Would export"));
    CHECK_THAT(listing.str(), Catch::Matchers::ContainsSubstring("credentials:claude"));

    // Round trip through stdin.
    Machine target("filo_portability_cli_target");
    std::istringstream bundle_in(out.str());
    std::ostringstream import_out;
    core::cli::ConfigTransferRequest import_request{.import_source = "-", .only = {"credentials"}};
    REQUIRE(core::cli::run_config_transfer(import_request, target.paths, bundle_in, import_out, err) == 0);
    CHECK(fs::exists(target.paths.config_dir / "oauth_kimi.json"));
    CHECK_THAT(import_out.str(), Catch::Matchers::ContainsSubstring("create"));
}
