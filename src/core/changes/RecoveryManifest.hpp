#pragma once

#include "ObjectStore.hpp"
#include "../utils/PathUtils.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace core::changes {

struct RecoveryEntry {
    RecoveryImage before;
    RecoveryImage after;
    bool after_known = false;
    /// FileStamp::settled_identity of the file when it held `before`/`after`;
    /// 0 when unknown. A file still carrying it holds those bytes, so it is
    /// neither read nor hashed again — in this turn or the next.
    std::uint64_t before_stamp = 0;
    std::uint64_t after_stamp = 0;

    /// The image the file was last seen to hold, and its stamp then.
    [[nodiscard]] const RecoveryImage& latest() const noexcept { return after_known ? after : before; }
    [[nodiscard]] std::uint64_t latest_stamp() const noexcept { return after_known ? after_stamp : before_stamp; }
    bool operator==(const RecoveryEntry&) const = default;
};

enum class RecoveryPhase { Recording, Complete, Interrupted, Reverted };

using RecoveryEntries = std::map<std::filesystem::path, RecoveryEntry, core::utils::path::NativeLess>;

struct RecoveryManifest {
    std::filesystem::path workspace;
    RecoveryPhase phase = RecoveryPhase::Recording;
    bool partial = false;
    bool ambiguous_ownership = false;
    bool inventory_complete = false;
    std::vector<std::filesystem::path> roots;
    std::set<std::filesystem::path, core::utils::path::NativeLess> uncaptured;
    std::vector<int> groups;
    RecoveryEntries entries;
};

/// The durable JSON form of a manifest, and of a change to one.
namespace manifest_json {

/// Bounds a decoded manifest may not exceed.
struct Limits {
    std::size_t files;
    std::size_t roots = 64;
};

[[nodiscard]] std::string encode(const RecoveryManifest& manifest);
/// What turns @p from into @p to, naming only what differs; nullopt when
/// nothing does.
[[nodiscard]] std::optional<std::string> encode_change(const RecoveryManifest& from, const RecoveryManifest& to);
/// A manifest of @p workspace; fails on a schema, bound or workspace mismatch.
[[nodiscard]] std::expected<RecoveryManifest, std::string> decode(
    std::string_view json, const std::filesystem::path& workspace, Limits limits);
/// Applies a change encode_change() wrote.
[[nodiscard]] std::expected<void, std::string> apply_change(
    std::string_view json, RecoveryManifest& manifest, Limits limits);

} // namespace manifest_json

} // namespace core::changes
