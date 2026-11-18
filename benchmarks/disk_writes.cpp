// Offline benchmark: how many bytes Filo's own persistence writes to disk while
// an agent works. It drives the same recovery checkpoint, turn tracker and
// execution journal the agent uses, on a synthetic workspace in a temporary
// directory, and reads the bytes this process wrote from the kernel:
// proc_pid_rusage on macOS, /proc/self/io on Linux. Nothing outside the
// temporary directory is touched.
//
//   filo_disk_write_benchmark [files] [file_kib] [sessions] [turns] [tools] [cold_resume] [small_edits]
// Set cold_resume to 1 to reopen the session store each turn, or small_edits
// to 1 to overwrite a few bytes in place rather than truncating each file.
//
// Defaults model a working day: a ~2,000-file workspace, several sessions of a
// few turns, about twenty tool calls a turn, half of them shell commands; each
// turn starts with a prompt that history keeps and memory recalls notes for.

#include "core/changes/RecoveryStore.hpp"
#include "core/changes/TurnChangeTracker.hpp"
#include "core/changes/TurnRecovery.hpp"
#include "core/session/ExecutionJournal.hpp"
#include "core/session/SessionStore.hpp"
#include "core/history/PromptHistoryStore.hpp"
#include "core/memory/MemoryStore.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <libproc.h>
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;

namespace {

/// Where Filo's own files go; settle() flushes nothing else.
fs::path g_data;

/// Flushes the delayed writes of every file under the data directory, so they
/// are charged now. Not sync(): that flushes every process's dirty pages and
/// charges them to this one.
void settle() {
    std::error_code ec;
    for (fs::recursive_directory_iterator it(g_data, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const int fd = ::open(it->path().c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        ::fsync(fd);
        ::close(fd);
    }
}

/// Bytes this process has written to storage. Delayed writes are charged when
/// they reach the device.
std::optional<std::uint64_t> bytes_written(bool flush = true) {
    if (flush) settle();
#if defined(__APPLE__)
    rusage_info_v2 info{};
    if (proc_pid_rusage(::getpid(), RUSAGE_INFO_V2, reinterpret_cast<rusage_info_t*>(&info)) != 0)
        return std::nullopt;
    return info.ri_diskio_byteswritten;
#else
    std::ifstream io("/proc/self/io");
    std::string key;
    std::uint64_t value = 0;
    while (io >> key >> value)
        if (key == "write_bytes:") return value;
    return std::nullopt;
#endif
}

std::uint64_t tree_bytes(const fs::path& root, std::size_t& files) {
    std::uint64_t total = 0;
    files = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        ++files;
        total += it->file_size(ec);
    }
    return total;
}

/// Writes a workspace file durably, so its cost is charged now and can be
/// told apart from what Filo writes.
void put(const fs::path& path, std::string_view bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return;
    if (::write(fd, bytes.data(), bytes.size()) < 0) std::cerr << "write failed\n";
    ::fsync(fd);
    ::close(fd);
}

std::string mib(std::uint64_t bytes) { return std::format("{:.1f} MiB", static_cast<double>(bytes) / (1024.0 * 1024.0)); }

std::size_t argument(int argc, char** argv, int index, std::size_t fallback) {
    return argc > index ? static_cast<std::size_t>(std::strtoull(argv[index], nullptr, 10)) : fallback;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t files = argument(argc, argv, 1, 2000);
    const std::size_t file_kib = argument(argc, argv, 2, 16);
    const std::size_t sessions = argument(argc, argv, 3, 4);
    const std::size_t turns = argument(argc, argv, 4, 4);
    const std::size_t tools = argument(argc, argv, 5, 20);

    const bool cold_resume = argument(argc, argv, 6, 0) != 0;
    const bool small_edits = argument(argc, argv, 7, 0) != 0;

    const auto base = fs::temp_directory_path() / std::format("filo-disk-writes-{}", ::getpid());
    const auto workspace = base / "workspace";
    g_data = base / "data";
    const auto sessions_dir = g_data / "sessions";
    fs::create_directories(workspace / "src");
    fs::create_directories(sessions_dir);
    std::mt19937_64 random(42);
    // Tool output reads like text, not like one repeated byte: it must not
    // compress better than a real conversation does.
    const auto text = [&](std::size_t bytes) {
        static constexpr std::string_view kWords[] = {"the", "file", "return", "const", "auto", "std::string",
            "error", "value", "if", "for", "path", "turn", "session", "write", "read", "{", "}", ";", "\\n", "0x1f"};
        std::string out;
        while (out.size() < bytes) out.append(kWords[random() % std::size(kWords)]).push_back(' ');
        return out;
    };
    std::vector<fs::path> paths;
    for (std::size_t i = 0; i < files; ++i) {
        paths.push_back(workspace / "src" / std::format("file-{:05}.cpp", i));
        std::string bytes(file_kib * 1024, static_cast<char>('a' + i % 26));
        bytes += std::to_string(i);
        put(paths.back(), bytes);
    }
    // Memory as a long-used store holds it: a hundred active notes and the
    // archived originals of revisions, about 800 KB. Prompt history with a
    // couple of thousand prompts.
    core::memory::MemoryStore memory(g_data / "config" / "memory.json");
    {
        core::memory::MemoryState state;
        for (std::size_t i = 0; i < 1500; ++i)
            state.entries.push_back({.id = std::format("m{}", i), .content = text(480),
                                     .created_at = "2026-01-01T00:00:00Z", .archived = i >= 100});
        fs::create_directories(g_data / "config");
        if (!memory.save(state)) std::cerr << "memory setup failed\n";
    }
    core::history::PromptHistoryStore history(g_data / "history.json");
    for (std::size_t i = 0; i < 2000; ++i) history.add(text(120));
    if (!history.save()) std::cerr << "history setup failed\n";

    // A repository at rest: identities are trusted only once files settle.
    std::this_thread::sleep_for(core::utils::FileStamp::kSettleTime + std::chrono::seconds(1));

    const auto start = bytes_written();
    if (!start) {
        std::cerr << "This platform does not report per-process disk writes.\n";
        return 1;
    }

    core::session::ExecutionJournal journal(core::session::ExecutionJournal::root_in(sessions_dir));
    const core::session::SessionStore store(sessions_dir);
    const std::string payload = std::format(R"({{"role":"tool","content":"{}"}})", text(2048));
    std::size_t edits = 0, shells = 0, appends = 0, saves = 0;
    std::uint64_t recovery_written = 0, journal_written = 0, session_written = 0, edits_written = 0, prompt_written = 0;
    // The agent's own edits are the work, not overhead: they are measured
    // apart and left out of the recovery figure.
    const auto edit = [&](const fs::path& target, std::string_view bytes) {
        const auto before = *bytes_written(false);
        if (small_edits) {
            std::fstream file(target, std::ios::binary | std::ios::in | std::ios::out);
            if (!file.write(bytes.data(), static_cast<std::streamsize>(bytes.size())))
                throw std::runtime_error("in-place edit failed");
        } else {
            put(target, bytes);
        }
        edits_written += *bytes_written(false) - before;
    };
    for (std::size_t session = 0; session < sessions; ++session) {
        const auto id = std::format("bench{:03}", session);
        const auto root = core::changes::RecoveryStore::session_root_in(sessions_dir, id);
        core::session::SessionData data{.session_id = std::format("{:08x}", session), .working_dir = workspace.string()};
        data.created_at = data.last_active_at = core::session::SessionStore::now_iso8601();
        data.messages.push_back({.role = "user", .content = "Make it faster."});
        for (std::size_t turn = 0; turn < turns; ++turn) {
            // The prompt that starts the turn: history keeps it, and the
            // memories it carries are marked recalled.
            const auto before_prompt = *bytes_written();
            if (!history.append_and_save(text(200))) std::cerr << "history append failed\n";
            std::vector<std::string> recalled;
            for (int i = 0; i < 8; ++i) recalled.push_back(std::format("m{}", random() % 100));
            if (!memory.record_prompt_recall(recalled)) std::cerr << "memory recall failed\n";
            prompt_written += *bytes_written() - before_prompt;

            const auto before_turn = *bytes_written();
            const auto edits_before = edits_written;
            core::changes::TurnChangeTracker tracker(workspace);
            tracker.set_workspace_roots({workspace});
            tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(
                core::changes::RecoveryStore::new_directory(root), workspace));
            for (std::size_t tool = 0; tool < tools; ++tool) {
                const auto& target = paths[random() % paths.size()];
                if (tool % 2 == 0) {
                    ++shells;
                    tracker.track(core::changes::MutationScope{.unbounded = true, .tool = "run_terminal_command"}, [&] {
                        tracker.note_process_group(static_cast<int>(10'000 + shells));
                        // Most shell commands only read; one in four writes a file.
                        if (tool % 4 == 0) edit(target, std::format("shell edit {}\n", shells));
                        return 0;
                    });
                } else {
                    ++edits;
                    tracker.track(core::changes::MutationScope{.paths = {target}}, [&] {
                        edit(target, std::format("edit {}\n", edits));
                        return 0;
                    });
                }
            }
            static_cast<void>(tracker.changes());
            tracker.finish_recovery();
            recovery_written += *bytes_written() - before_turn;
            recovery_written -= edits_written - edits_before;

            const auto before_journal = *bytes_written();
            for (std::size_t event = 0; event < tools * 2; ++event, ++appends)
                static_cast<void>(journal.append(id, "agent", "", "message", payload));
            if (!journal.append(id, "agent", "", "turn_end", "{}")) return 1;
            ++appends;
            journal_written += *bytes_written() - before_journal;

            // What the conversation gained this turn, saved the way the TUI
            // saves it when a turn ends.
            const auto before_save = *bytes_written();
            for (std::size_t tool = 0; tool < tools; ++tool) {
                data.messages.push_back({.role = "assistant", .content = text(512)});
                data.messages.push_back({.role = "tool", .content = text(4096), .tool_call_id = std::format("call_{}", tool)});
            }
            data.messages.push_back({.role = "user", .content = text(256)});
            ++data.stats.turn_count;
            data.last_active_at = core::session::SessionStore::now_iso8601();
            if (cold_resume) {
                const core::session::SessionStore resumed(sessions_dir);
                if (turn && !resumed.load_by_id(data.session_id)) return 1;
                if (!resumed.save(data)) return 1;
            } else if (!store.save(data)) {
                return 1;
            }
            ++saves;
            session_written += *bytes_written() - before_save;
        }
    }
    const auto total = *bytes_written() - *start;

    std::size_t recovery_files = 0;
    const auto recovery_bytes = tree_bytes(core::changes::RecoveryStore::root_in(sessions_dir), recovery_files);
    std::size_t workspace_files = 0;
    const auto workspace_bytes = tree_bytes(workspace, workspace_files);
    std::cout << std::format(
        "workspace        {} files, {}\n"
        "activity         {} sessions x {} turns x {} tools ({} shell, {} edits), {} journal appends\n"
        "written          {} total: recovery {}, journal {}, sessions {} ({} saves), prompt+memory {}, agent edits {}\n"
        "recovery at rest {} files, {}\n",
        workspace_files, mib(workspace_bytes), sessions, turns, tools, shells, edits, appends,
        mib(total), mib(recovery_written), mib(journal_written), mib(session_written), saves, mib(prompt_written), mib(edits_written), recovery_files, mib(recovery_bytes));
    std::error_code ec;
    fs::remove_all(base, ec);
    return 0;
}
