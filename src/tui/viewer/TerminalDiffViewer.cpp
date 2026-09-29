#include "TerminalDiffViewer.hpp"

#include "core/tools/shell/ShellUtils.hpp"
#include "tui/CommandLookup.hpp"

#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

namespace tui::viewer {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kFilePrefix = "filo-diff-view-";
constexpr std::string_view kFileSuffix = ".patch";
constexpr std::string_view kDefaultPager = "less";
constexpr std::string_view kLastResortPager = "more";

// The patch a pager reads, in a private file that lives exactly as long as the
// comparison. The session id in its name keeps concurrent Filo instances apart;
// owner-only permissions keep everybody else out.
class PatchScratch {
public:
    [[nodiscard]] static std::expected<PatchScratch, std::string> create(
        std::string_view patch,
        std::string_view session_id) {
        std::error_code ec;
        const fs::path root = fs::temp_directory_path(ec);
        if (ec) {
            return std::unexpected(std::format(
                "Could not locate the temporary directory: {}", ec.message()));
        }
        const fs::path file = root / std::format("{}{}{}", kFilePrefix, session_id, kFileSuffix);

        {
            std::ofstream output(file, std::ios::binary | std::ios::trunc);
            if (!output) {
                return std::unexpected("Could not write the patch for the pager.");
            }
            output << patch;
            if (!output) {
                fs::remove(file, ec);
                return std::unexpected("Could not write the patch for the pager.");
            }
        }

        fs::permissions(
            file,
            fs::perms::owner_read | fs::perms::owner_write,
            fs::perm_options::replace,
            ec);
        if (ec) {
            fs::remove(file, ec);
            return std::unexpected("Could not secure the patch for the pager.");
        }
        return PatchScratch(file);
    }

    ~PatchScratch() {
        if (file_.empty()) {
            return;
        }
        std::error_code ec;
        fs::remove(file_, ec);
    }

    PatchScratch(PatchScratch&& other) noexcept : file_(std::exchange(other.file_, {})) {}
    PatchScratch& operator=(PatchScratch&& other) noexcept {
        if (this != &other) {
            std::error_code ec;
            fs::remove(file_, ec);
            file_ = std::exchange(other.file_, {});
        }
        return *this;
    }

    PatchScratch(const PatchScratch&) = delete;
    PatchScratch& operator=(const PatchScratch&) = delete;

    [[nodiscard]] const fs::path& file() const noexcept { return file_; }

private:
    explicit PatchScratch(fs::path file) : file_(std::move(file)) {}

    fs::path file_;
};

[[nodiscard]] std::string_view environment_pager() {
    if (const char* value = std::getenv("PAGER"); value != nullptr && *value != '\0') {
        return value;
    }
    return command_is_executable(kDefaultPager) ? kDefaultPager : kLastResortPager;
}

} // namespace

ViewerDescriptor terminal_diff_viewer_descriptor() noexcept {
    return ViewerDescriptor{
        .id = "terminal",
        .label = "Terminal",
        .description = "Page the diff in $PAGER (less, delta, ...).",
        .launch = ViewerLaunch::Terminal,
    };
}

std::string resolve_pager_command(std::string_view configured_command) {
    return configured_command.empty()
        ? std::string(environment_pager())
        : std::string(configured_command);
}

std::string build_pager_invocation(std::string_view command, std::string_view file_path) {
    // The patch arrives on standard input, the one channel every pager and diff
    // highlighter reads (`git diff | delta`). A path argument is not: delta,
    // for one, reads positional arguments as files to compare. The
    // command runs in a subshell so a pipeline such as `diff-so-fancy | less`
    // feeds the patch to its first stage rather than to its last.
    std::string invocation = "(";
    invocation += command;
    invocation += ") < '";
    invocation += core::tools::detail::shell_single_quote(file_path);
    invocation += "'";
    return invocation;
}

ViewerDescriptor TerminalDiffViewer::descriptor() const noexcept {
    return terminal_diff_viewer_descriptor();
}

ViewResult TerminalDiffViewer::view(const ViewContext& context) {
    if (!context.with_terminal) {
        return ViewResult::failed("No terminal is available for the pager.");
    }
    if (context.cancellation.stop_requested()) {
        return ViewResult::cancelled();
    }

    auto scratch = PatchScratch::create(context.patch, context.session_id);
    if (!scratch) {
        return ViewResult::failed(std::move(scratch.error()));
    }

    const std::string invocation = build_pager_invocation(
        resolve_pager_command(command_),
        scratch->file().string());

    int status = -1;
    context.with_terminal([&invocation, &status]() { status = std::system(invocation.c_str()); });

    if (const int exit_code = tui::child_exit_code(status); exit_code != 0) {
        return ViewResult::failed(std::format("The pager exited with status {}.", exit_code));
    }
    return ViewResult::shown();
}

} // namespace tui::viewer
