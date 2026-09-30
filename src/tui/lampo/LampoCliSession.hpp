#pragma once

// Lampo is a native macOS App Store application; its protocols exist only in
// Apple builds. Every other platform simply never sees a Lampo backend, and no
// catalog ever offers one.
#if defined(__APPLE__)

#include <chrono>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>

namespace tui::lampo {

/// Lampo's bundle identifier. One value for every protocol, because they all
/// reach the same application.
[[nodiscard]] std::string_view bundle_identifier() noexcept;

/// The client name Filo announces itself with. Display-only: Lampo shows it as
/// the title of a session it did not start itself.
[[nodiscard]] std::string_view client_name() noexcept;

/// One of Lampo's command-line protocols.
///
/// Every protocol has the same shape — a private session directory holding
/// exactly one payload file, announced on a private pasteboard — and differs
/// only in these four strings. Filo is one client of a public protocol; Lampo
/// knows nothing about Filo.
struct CliProtocol {
    std::string_view directory_prefix;   ///< Session directory name prefix.
    std::string_view pasteboard_prefix;  ///< Private pasteboard name prefix.
    std::string_view pasteboard_type;    ///< Property-list type on that pasteboard.
    std::string_view file_name;          ///< The one payload file Lampo reads.
    std::string_view reply = {};         ///< The `reply` value sent with the request.
};

// The protocols Lampo publishes, declared together because they are one wire
// contract with one application: same shape, same version, same channel rules,
// and a backend only picks the one it speaks. The values must match Lampo's
// PrompterCli.swift and ComparerCli.swift exactly; a drift here is invisible
// until a session is silently treated as an ordinary document open.

/// Editing a prompt draft in Lampo's Prompter.
inline constexpr CliProtocol kPromptEditProtocol{
    .directory_prefix = "lampo-prompter-edit-v1-",
    .pasteboard_prefix = "alessio.pollero.Lampo.prompter-edit.",
    .pasteboard_type = "alessio.pollero.Lampo.prompter-edit.v1",
    .file_name = "prompt.md",
};

/// Showing a unified diff in Lampo's Comparer.
inline constexpr CliProtocol kComparerProtocol{
    .directory_prefix = "lampo-comparer-view-v1-",
    .pasteboard_prefix = "alessio.pollero.Lampo.comparer-view.",
    .pasteboard_type = "alessio.pollero.Lampo.comparer-view.v1",
    .file_name = "changes.patch",
    .reply = "comments",
};

/// The states a Lampo CLI protocol speaks. `request` is the client's; the rest
/// are Lampo's answers, and which of them a protocol uses is that protocol's
/// business.
namespace state {
inline constexpr std::string_view request = "request";
inline constexpr std::string_view editing = "editing";
inline constexpr std::string_view viewing = "viewing";
/// A comparer session whose comments go back to the client, in place of
/// `viewing`. Lampo stays in it until the review window closes.
inline constexpr std::string_view annotating = "annotating";
inline constexpr std::string_view saved = "saved";
inline constexpr std::string_view cancelled = "cancelled";
inline constexpr std::string_view failed = "failed";
} // namespace state

/// One message Lampo published on this session's channel.
struct CliMessage {
    std::string state;
    std::string error;  ///< Set only when Lampo reported a failure reason.
};

/// The client half of one Lampo CLI session.
///
/// The session owns the whole exchange: it creates and secures the private
/// directory, writes the payload, announces the session, hands the payload to
/// Launch Services, and tears everything down on destruction — including on the
/// cancelled and failed paths. A backend therefore only has to write the state
/// machine its own protocol defines.
class CliSession {
public:
    /// Creates a session whose payload is `content`, and announces it.
    ///
    /// The announcement happens here rather than at launch because Lampo only
    /// adopts an open it can match to a live request; a payload that arrives
    /// unannounced is treated as an ordinary document.
    [[nodiscard]] static std::expected<CliSession, std::string> create(
        const CliProtocol& protocol,
        std::string_view session_id,
        std::string_view content);

    /// Creates a session whose payload is a copy of `source`, for a backend
    /// that already keeps its payload in a scratch file and has no reason to
    /// read it back. Announces the session like create() does.
    [[nodiscard]] static std::expected<CliSession, std::string> create_from_file(
        const CliProtocol& protocol,
        std::string_view session_id,
        const std::filesystem::path& source);

    ~CliSession();

    CliSession(CliSession&& other) noexcept;
    CliSession& operator=(CliSession&& other) noexcept;

    CliSession(const CliSession&) = delete;
    CliSession& operator=(const CliSession&) = delete;

    /// The payload file Lampo is asked to open.
    [[nodiscard]] const std::filesystem::path& file() const noexcept { return file_; }

    /// Asks Launch Services to open the payload with Lampo, and remembers the
    /// exact instance that answered so its liveness can be checked afterwards.
    [[nodiscard]] std::expected<void, std::string> launch(std::stop_token cancellation);

    /// The next message Lampo published, or nullopt when nothing new has
    /// arrived. Callers poll; the interval is theirs to choose.
    [[nodiscard]] std::optional<CliMessage> poll();

    /// Tells Lampo this client is gone, so a session waiting on it can stop.
    /// Safe to call more than once, and only ever published by the client.
    void cancel();

    /// True while the Lampo instance that accepted the launch is still running.
    /// False before launch() succeeded, which reads as "gone" on purpose.
    [[nodiscard]] bool app_running() const;

    /// Sleeps for `interval`, returning as soon as cancellation is requested.
    static void idle(std::stop_token cancellation, std::chrono::milliseconds interval);

private:
    /// Bridges NSWorkspace's asynchronous launch result to this session. Held
    /// by shared pointer because the completion handler outlives a session that
    /// gave up waiting for it.
    struct Launch;

    CliSession() = default;

    /// Creates and secures the session directory, leaving the payload unwritten
    /// and the session unannounced.
    [[nodiscard]] static std::expected<CliSession, std::string> begin(
        const CliProtocol& protocol,
        std::string_view session_id);
    [[nodiscard]] std::expected<void, std::string> write_text(std::string_view content);
    [[nodiscard]] std::expected<void, std::string> write_copy_of(
        const std::filesystem::path& source);
    /// Publishes the request and records the change count, so the client's own
    /// announcement is never mistaken for Lampo's answer.
    void announce();
    void publish(std::string_view state, std::string_view error = {});
    [[nodiscard]] std::expected<void, std::string> secure_payload();
    void discard() noexcept;

    std::filesystem::path directory_;
    std::filesystem::path file_;
    std::string session_id_;
    std::string pasteboard_name_;
    std::string pasteboard_type_;
    std::string reply_;
    long observed_change_count_ = 0;
    std::shared_ptr<Launch> launch_;
};

} // namespace tui::lampo

#endif // defined(__APPLE__)
