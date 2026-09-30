#include "TurnChangeTracker.hpp"

#include "../tools/ToolDiffUtils.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>
#include <set>
#include <span>
#include <string_view>
#include <utility>

namespace core::changes {
namespace {

namespace fs = std::filesystem;
using core::tools::detail::kFnvOffsetBasis;
using core::tools::detail::kMaxToolDiffInputBytes;

/// What one retained line hash costs, so both the allowance and the check that
/// spends it agree on the price.
constexpr std::size_t kBytesPerLineHash = sizeof(std::uint64_t);

/// Additions and deletions of one change, when they could be established.
using Counts = std::optional<std::pair<std::size_t, std::size_t>>;

std::uint64_t fnv1a(std::string_view bytes,
                    std::uint64_t seed = kFnvOffsetBasis) {
    std::uint64_t hash = seed;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string canonical_string(const fs::path& path) {
    std::error_code ec;
    fs::path canonical = fs::weakly_canonical(path, ec);
    if (ec) {
        canonical = path.lexically_normal();
    }
    std::string key = canonical.generic_string();
    while (key.size() > 1 && key.ends_with('/')) {
        key.pop_back();
    }
    return key;
}

/// Lines in `text`, counted the way a unified diff counts them: a final line
/// without a terminator is still a line.
std::size_t count_lines(std::string_view text) {
    std::size_t lines = static_cast<std::size_t>(std::ranges::count(text, '\n'));
    if (!text.empty() && text.back() != '\n') {
        ++lines;
    }
    return lines;
}

} // namespace

TurnChangeTracker::TurnChangeTracker(fs::path display_root)
    : TurnChangeTracker(std::move(display_root), Limits{}) {}

TurnChangeTracker::TurnChangeTracker(fs::path display_root, Limits limits)
    : display_root_(canonical_string(display_root)), limits_(limits) {}

TurnChangeTracker::Stat TurnChangeTracker::stat_of(const fs::path& path) {
    std::error_code ec;
    const auto status = fs::status(path, ec);
    if (ec || !fs::exists(status)) {
        return {};  // Absent, or gone before it could be examined.
    }
    if (!fs::is_regular_file(status)) {
        return {.exists = true};
    }
    Stat stat{.exists = true, .regular = true};
    stat.size = fs::file_size(path, ec);
    if (!ec) {
        std::error_code time_ec;
        const auto modified = static_cast<long long>(
            fs::last_write_time(path, time_ec).time_since_epoch().count());
        stat.stamp = fnv1a(std::to_string(stat.size) + ':' + std::to_string(modified));
    }
    return stat;
}

std::size_t TurnChangeTracker::remaining_bytes_unlocked() const {
    // A single file may overshoot the bound: the check reserves one copy while
    // a path seen for the first time is retained twice, as baseline and as
    // current. Saturating keeps that from wrapping into an endless budget.
    return tracked_bytes_ < limits_.max_tracked_bytes
        ? limits_.max_tracked_bytes - tracked_bytes_
        : 0;
}

std::size_t TurnChangeTracker::remaining_hash_bytes_unlocked() const {
    return hashed_bytes_ < limits_.max_hashed_bytes
        ? limits_.max_hashed_bytes - hashed_bytes_
        : 0;
}

TurnChangeTracker::Snapshot TurnChangeTracker::read_unlocked(const fs::path& path,
                                                            const Reclaimable& reclaimable) {
    const auto stat = stat_of(path);
    if (!stat.exists) {
        return {};
    }

    Snapshot snapshot{.state = Snapshot::State::Opaque, .stamp = stat.stamp};
    if (!stat.regular || stat.stamp == 0) {
        // Not a regular file, or its size and mtime could not be read: nothing
        // to diff, and no cheap way to notice a change either.
        return snapshot;
    }

    const auto size = stat.size;
    const bool oversized = size > kMaxToolDiffInputBytes;
    if (oversized || size > remaining_bytes_unlocked() + reclaimable.text) {
        // The bytes are not kept, so this change cannot be diffed: one file is
        // beyond any diff Filo shows, or the turn has retained all it may. One
        // bounded look still says how many lines the file held and whether it
        // was text at all, so the summary can report the size of a change it
        // cannot show, and name a binary file as binary rather than as a
        // casualty of the budget.
        const auto scan = scan_without_retaining(path);
        snapshot.fingerprint = snapshot.stamp;
        if (scan && !scan->text) {
            // A binary file is not read to the end, so its head cannot identify
            // it: size and mtime stay the only cheap witness of a change.
            snapshot.opaque_as = FileChangeContent::Binary;
            return snapshot;
        }
        snapshot.opaque_as = oversized ? FileChangeContent::TooLarge
                                       : FileChangeContent::BudgetSpent;
        if (scan) {
            snapshot.lines = scan->lines;
            // The bytes were read to count them, so they can also identify the
            // file: a rewrite that changes nothing is then not a change, which
            // a size-and-mtime stamp would call one.
            snapshot.fingerprint = scan->fingerprint;
            // Line hashes are retained memory like text, and charged to an
            // allowance of their own: they are what this file is counted from
            // once its bytes are gone, so a turn that has spent its text budget
            // is exactly the turn that still needs them.
            if (scan->line_hashes.size() * kBytesPerLineHash
                <= remaining_hash_bytes_unlocked() + reclaimable.hashes) {
                snapshot.line_hashes = std::move(scan->line_hashes);
            }
        }
        // Only a spent turn budget is a gap in the summary; an oversized file
        // describes itself.
        partial_enumeration_ |= !oversized;
        return snapshot;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return snapshot;
    }
    std::string bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    if (!core::tools::detail::is_text_like_for_diff(bytes)) {
        snapshot.fingerprint = fnv1a(bytes);
        return snapshot;
    }
    const auto lines = count_lines(bytes);
    return {.state = Snapshot::State::Text,
            .text = std::move(bytes),
            .lines = lines,
            .stamp = snapshot.stamp};
}

std::optional<TurnChangeTracker::Scan> TurnChangeTracker::scan_without_retaining(
    const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    Scan scan;
    std::array<char, 64 * 1024> buffer{};
    std::uint64_t seen = 0;
    std::uint64_t hash = kFnvOffsetBasis;
    bool trailing_text = false;  // A last line without a terminator.
    bool first_chunk = true;
    bool hashing = true;  // Until the file shows more lines than one keeps.
    core::tools::detail::LineHasher hasher;
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = static_cast<std::size_t>(in.gcount());
        if (count == 0) {
            break;
        }
        seen += count;
        if (seen > kMaxCountedFileBytes) {
            return std::nullopt;
        }
        const std::string_view chunk(buffer.data(), count);
        // Text or binary is decided by the head of the file, exactly where the
        // retained path decides it; reading on only counts lines and identifies
        // the content they came from.
        if (first_chunk && !core::tools::detail::is_text_like_for_diff(chunk)) {
            scan.text = false;
            return scan;
        }
        first_chunk = false;
        scan.lines += static_cast<std::size_t>(std::ranges::count(chunk, '\n'));
        hash = fnv1a(chunk, hash);
        trailing_text = chunk.back() != '\n';
        if (hashing) {
            hasher.feed(chunk);
            if (hasher.hashes.size() >= kMaxHashedLines) {
                // More lines than one summary hashes. Hashes of only the first
                // lines would count only part of a change, so they are given up
                // whole.
                hashing = false;
                hasher.hashes.clear();
            }
        }
    }
    scan.lines += trailing_text ? 1 : 0;
    scan.fingerprint = hash;
    if (hashing) {
        hasher.finish();
        scan.line_hashes = std::move(hasher.hashes);
    }
    return scan;
}

TurnChangeTracker::Reclaimable TurnChangeTracker::reclaimable_for_unlocked(
    const Key& key) const {
    const auto known = current_.find(key);
    return known == current_.end()
        ? Reclaimable{}
        : Reclaimable{.text = known->second.text.size(), .hashes = hash_bytes(known->second)};
}

std::size_t TurnChangeTracker::hash_bytes(const Snapshot& snapshot) noexcept {
    return snapshot.line_hashes ? snapshot.line_hashes->size() * kBytesPerLineHash : 0;
}

void TurnChangeTracker::charge_unlocked(const Snapshot& snapshot) noexcept {
    tracked_bytes_ += snapshot.text.size();
    hashed_bytes_ += hash_bytes(snapshot);
}

void TurnChangeTracker::release_unlocked(const Snapshot& snapshot) noexcept {
    tracked_bytes_ -= snapshot.text.size();
    hashed_bytes_ -= hash_bytes(snapshot);
}

void TurnChangeTracker::store_current_unlocked(const Key& key, Snapshot snapshot) {
    if (const auto known = current_.find(key); known != current_.end()) {
        release_unlocked(known->second);  // Superseded content is released.
    }
    charge_unlocked(snapshot);
    current_[key] = std::move(snapshot);
}

TurnChangeTracker::Key TurnChangeTracker::key_for(const fs::path& path) const {
    return canonical_string(path);
}

TurnChangeTracker::Expansion TurnChangeTracker::expand_unlocked(const Key& root) const {
    Expansion expansion;
    std::error_code ec;
    if (fs::is_directory(root, ec)) {
        auto it = fs::recursive_directory_iterator(
            root, fs::directory_options::skip_permission_denied, ec);
        for (; !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (it->is_directory(ec) && it->path().filename() == ".git") {
                it.disable_recursion_pending();
            } else if (it->is_regular_file(ec)) {
                if (expansion.keys.size() >= limits_.max_files_per_directory) {
                    // More files exist than one summary enumerates. Stop, and
                    // let the caller report the summary as partial rather than
                    // presenting a short list as a complete one.
                    expansion.complete = false;
                    break;
                }
                expansion.keys.push_back(it->path().lexically_normal().generic_string());
            }
        }
    } else {
        expansion.keys.push_back(root);
    }

    // Files seen earlier under a directory that is now gone or changed.
    const std::string prefix = root + '/';
    for (auto it = current_.lower_bound(prefix);
         it != current_.end() && it->first.starts_with(prefix);
         ++it) {
        expansion.keys.push_back(it->first);
    }
    std::ranges::sort(expansion.keys);
    expansion.keys.erase(
        std::ranges::unique(expansion.keys).begin(), expansion.keys.end());
    return expansion;
}

void TurnChangeTracker::reconcile_known_unlocked() {
    std::vector<Key> keys;
    keys.reserve(current_.size());
    for (const auto& [key, snapshot] : current_) {
        keys.push_back(key);
    }
    for (const auto& key : keys) {
        // Metadata only: a path whose size and mtime did not move is not read.
        if (stat_of(key).stamp == current_.at(key).stamp) {
            continue;
        }
        store_current_unlocked(
            key, read_unlocked(key, reclaimable_for_unlocked(key)));
    }
}

void TurnChangeTracker::observe(const MutationScope& scope, Observation when) {
    std::lock_guard lock(mutex_);
    if (scope.unbounded) {
        // The tool could have written anywhere. Everything the turn already
        // knows is brought up to date; what it created from nothing cannot be
        // discovered, so the summary is flagged — and names the tool — rather
        // than trusted as whole.
        unscoped_mutations_ = true;
        note_unscoped_tool_unlocked(scope.tool);
        reconcile_known_unlocked();
    }
    for (const auto& path : scope.paths) {
        const Key root = key_for(path);
        const auto expansion = expand_unlocked(root);
        partial_enumeration_ |= !expansion.complete;
        for (const auto& key : expansion.keys) {
            Snapshot snapshot = read_unlocked(key, reclaimable_for_unlocked(key));
            if (!baseline_.contains(key)) {
                // A path met for the first time after the mutation was not seen
                // before it, so what it held then is unknown: an empty baseline
                // reports it as the change it is, and lets a move pair it with
                // the origin it came from. Before the mutation, what is on disk
                // is the pre-image.
                Snapshot baseline = when == Observation::After ? Snapshot{} : snapshot;
                charge_unlocked(baseline);
                baseline_.emplace(key, std::move(baseline));
            }
            store_current_unlocked(key, std::move(snapshot));
        }
    }
    for (const auto& [source, destination] : scope.moves) {
        record_move_unlocked(key_for(source), key_for(destination));
    }
}

void TurnChangeTracker::note_unscoped_mutation(std::string tool) {
    observe(MutationScope{.unbounded = true, .tool = std::move(tool)}, Observation::After);
}

void TurnChangeTracker::settle() {
    std::lock_guard lock(mutex_);
    if (!unscoped_mutations_) {
        return;
    }
    reconcile_known_unlocked();
}

void TurnChangeTracker::note_unscoped_tool_unlocked(std::string_view tool) {
    if (tool.empty() || unscoped_tools_.size() == kMaxUnscopedToolNames) {
        return;
    }
    if (std::ranges::find(unscoped_tools_, tool) == unscoped_tools_.end()) {
        unscoped_tools_.emplace_back(tool);
    }
}

void TurnChangeTracker::record_move_unlocked(const Key& source, const Key& destination) {
    const auto link = [this](const Key& from, const Key& to) {
        const auto from_now = current_.find(from);
        const auto to_now = current_.find(to);
        if (from_now == current_.end() || to_now == current_.end()
            || from_now->second.state != Snapshot::State::Missing
            || to_now->second.state == Snapshot::State::Missing) {
            return;  // The move did not happen (yet).
        }
        Key origin = from;
        if (const auto chained = origin_.find(from); chained != origin_.end()) {
            origin = chained->second;
            origin_.erase(chained);
        }
        origin_[to] = std::move(origin);
    };

    link(source, destination);
    const std::string prefix = source + '/';
    for (auto it = current_.lower_bound(prefix);
         it != current_.end() && it->first.starts_with(prefix);
         ++it) {
        link(it->first, destination + it->first.substr(source.size()));
    }
}

std::string TurnChangeTracker::display_path(const Key& key) const {
    const fs::path relative = fs::path(key).lexically_relative(display_root_);
    const std::string text = relative.generic_string();
    if (text.empty() || text == "." || text.starts_with("..")) {
        return key;
    }
    return text;
}

FileChangeContent TurnChangeTracker::no_diff_reason(const Snapshot& before,
                                                    const Snapshot& after) {
    const auto reason = [](const Snapshot& snapshot) {
        return snapshot.state == Snapshot::State::Opaque ? snapshot.opaque_as
                                                         : FileChangeContent::Text;
    };
    const auto left = reason(before);
    const auto right = reason(after);
    if (left == FileChangeContent::Text && right == FileChangeContent::Text) {
        return FileChangeContent::Text;
    }
    // Naming a file binary tells the reader more than naming the budget that
    // ran out, so the more specific reason wins.
    for (const auto candidate : {FileChangeContent::Binary, FileChangeContent::TooLarge}) {
        if (left == candidate || right == candidate) {
            return candidate;
        }
    }
    return FileChangeContent::BudgetSpent;
}

Counts TurnChangeTracker::counted_without_diff(const Snapshot& before,
                                              const Snapshot& after,
                                              core::tools::detail::WorkBudget& budget) {
    // The hashes one side can be counted from: the ones a read kept when it
    // could not keep the bytes, or the ones its retained text still yields.
    // Borrowed where they already exist, so counting a file the turn could not
    // afford does not copy what it dropped.
    const auto hashes_of = [](const Snapshot& snapshot,
                              std::vector<std::uint64_t>& owned)
        -> std::optional<std::span<const std::uint64_t>> {
        switch (snapshot.state) {
            case Snapshot::State::Text:
                owned = core::tools::detail::hash_text_lines(snapshot.text);
                return std::span<const std::uint64_t>{owned};
            case Snapshot::State::Opaque:
                if (snapshot.line_hashes) {
                    return std::span<const std::uint64_t>{*snapshot.line_hashes};
                }
                return std::nullopt;
            case Snapshot::State::Missing:
                return std::nullopt;
        }
        return std::nullopt;
    };
    std::vector<std::uint64_t> before_owned;
    std::vector<std::uint64_t> after_owned;
    const auto lhs = hashes_of(before, before_owned);
    const auto rhs = hashes_of(after, after_owned);
    if (!lhs || !rhs) {
        return std::nullopt;
    }
    return core::tools::detail::count_line_edits(*lhs, *rhs, budget);
}

FileChange TurnChangeTracker::describe(FileChangeKind kind,
                                       const Key& from,
                                       const Key& to,
                                       ReportBudget& budget) const {
    const Snapshot& before = baseline_.at(from);
    const Snapshot& after = current_.at(to);

    FileChange change{.kind = kind, .path = display_path(to)};
    if (kind == FileChangeKind::Renamed) {
        change.previous_path = display_path(from);
    }

    // A change with no diff still has an exact size when a file simply appeared
    // or disappeared: its lines were counted even though its bytes were not
    // kept. A modification has no such shortcut, so it takes whatever the
    // caller could establish, and reports no counts rather than guess.
    const auto undiffed = [&](FileChangeContent reason, Counts counts) {
        change.content = reason;
        if (kind == FileChangeKind::Added) {
            change.added = after.lines.value_or(0);
        } else if (kind == FileChangeKind::Deleted) {
            change.deleted = before.lines.value_or(0);
        } else if (counts) {
            change.added = counts->first;
            change.deleted = counts->second;
        }
        return change;
    };
    // What the per-line hashes both sides kept can still be searched for: the
    // numbers a change no diff was built for. One file gets one search's worth
    // of the report's allowance, so a change beyond the bound of a single
    // search costs that bound and no more, and the files after it are left an
    // allowance to be counted from.
    const auto hashed_counts = [&] {
        const auto allowance = std::min(core::tools::detail::kMaxMyersWorkUnits,
                                        budget.count_work.remaining());
        core::tools::detail::WorkBudget search(allowance);
        auto counts = counted_without_diff(before, after, search);
        budget.count_work.charge(allowance - search.remaining());
        return counts;
    };

    const auto opaque = [](const Snapshot& snapshot) {
        return snapshot.state == Snapshot::State::Opaque;
    };
    if (opaque(before) || opaque(after)) {
        return undiffed(no_diff_reason(before, after), hashed_counts());
    }
    if (before.text == after.text) {
        return change;  // A pure rename.
    }
    if (budget.diff_bytes == 0) {
        return undiffed(FileChangeContent::BudgetSpent, hashed_counts());
    }
    auto outcome = core::tools::detail::build_unified_diff_detailed(
        change.path, before.text, after.text);
    if (!outcome.has_diff()) {
        // Beyond what one diff may search or emit. The search may still have
        // measured the change before its text proved unaffordable. Where it did
        // not, the hashes both sides kept are searched on the report's own
        // allowance — and that search charges a line what the diff charges a
        // byte, so a change too reshaped to diff is often still cheap to count.
        return undiffed(FileChangeContent::TooLarge,
                        outcome.counts ? outcome.counts : hashed_counts());
    }
    if (outcome.diff->size() > budget.diff_bytes) {
        // This one file would spend the whole turn's remaining detail budget.
        // The search that built it already proved the numbers, so they survive
        // the text being dropped.
        return undiffed(FileChangeContent::BudgetSpent, outcome.counts);
    }
    budget.diff_bytes -= outcome.diff->size();
    // A diff that fit the turn's budget was measured to build it, so its
    // numbers came back with it.
    change.added = outcome.counts->first;
    change.deleted = outcome.counts->second;
    change.diff = std::move(*outcome.diff);
    return change;
}

TurnChanges TurnChangeTracker::changes() const {
    std::lock_guard lock(mutex_);
    TurnChanges result;
    result.partial_enumeration = partial_enumeration_;
    result.unscoped_mutations = unscoped_mutations_;
    result.unscoped_tools = unscoped_tools_;
    ReportBudget budget{.diff_bytes = limits_.max_diff_bytes,
                        .count_work = core::tools::detail::WorkBudget(limits_.max_count_work)};
    std::set<Key> reported;

    const auto missing = [](const Snapshot& snapshot) {
        return snapshot.state == Snapshot::State::Missing;
    };
    for (const auto& [destination, origin] : origin_) {
        if (destination == origin
            || !missing(current_.at(origin)) || missing(current_.at(destination))
            || missing(baseline_.at(origin)) || !missing(baseline_.at(destination))) {
            continue;
        }
        result.files.push_back(
            describe(FileChangeKind::Renamed, origin, destination, budget));
        reported.insert(origin);
        reported.insert(destination);
    }

    for (const auto& [key, before] : baseline_) {
        const Snapshot& after = current_.at(key);
        if (reported.contains(key) || before.same_content_as(after)) {
            continue;
        }
        const auto kind = missing(before) ? FileChangeKind::Added
            : missing(after)              ? FileChangeKind::Deleted
                                          : FileChangeKind::Modified;
        result.files.push_back(describe(kind, key, key, budget));
    }

    std::ranges::sort(result.files, {}, &FileChange::path);
    // Detail given up at report time is a gap in the summary, not in a file.
    result.partial_enumeration |= std::ranges::any_of(
        result.files, [](const FileChange& change) {
            return change.content == FileChangeContent::BudgetSpent;
        });
    return result;
}

} // namespace core::changes
