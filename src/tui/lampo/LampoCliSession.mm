#include "LampoCliSession.hpp"

#import <AppKit/AppKit.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <format>
#include <fstream>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>

#include <signal.h>

namespace tui::lampo {
namespace {

constexpr std::string_view kBundleIdentifier = "alessio.pollero.Lampo";
constexpr std::string_view kClientName = "Filo";

/// Every message on every Lampo channel speaks this version. A message that
/// does not is not part of the exchange, whichever session it landed on.
constexpr std::string_view kProtocolVersion = "1";

constexpr auto kPollInterval = std::chrono::milliseconds(100);
/// Long enough for a cold launch of a sandboxed GUI app, short enough that a
/// user who never sees a window is told so instead of being left waiting.
constexpr auto kLaunchTimeout = std::chrono::seconds(15);
/// Cancellation is checked between slices, so a session ends promptly without
/// the caller having to pick a short poll interval.
constexpr auto kStopCheckSlice = std::chrono::milliseconds(20);

[[nodiscard]] NSString* to_ns(std::string_view value) {
    return [[NSString alloc] initWithBytes:value.data()
                                    length:static_cast<NSUInteger>(value.size())
                                  encoding:NSUTF8StringEncoding];
}

[[nodiscard]] std::string to_std(NSString* value) {
    return value == nil ? std::string{} : std::string(value.UTF8String);
}

/// The named pasteboard, freshly addressed. Lampo resolves the same board from
/// its own side the same way, so no instance has to be kept alive here.
[[nodiscard]] NSPasteboard* pasteboard(std::string_view name) {
    return [NSPasteboard pasteboardWithName:NSPasteboardName(to_ns(name))];
}

} // namespace

/// Bridges NSWorkspace's asynchronous launch result to the waiting session and
/// retains the exact Lampo instance that accepted the open.
struct CliSession::Launch {
    void complete(NSRunningApplication* application, NSError* error) {
        {
            const std::lock_guard lock(mutex);
            if (completed) {
                return;
            }
            completed = true;
            process_identifier = application.processIdentifier;
            if (error != nil) {
                failure = to_std(error.localizedDescription);
            } else if (application == nil) {
                failure = "Lampo did not return a running application.";
            }
        }
        completion.notify_all();
    }

    [[nodiscard]] std::expected<void, std::string> wait(std::stop_token cancellation) {
        const auto deadline = std::chrono::steady_clock::now() + kLaunchTimeout;
        std::unique_lock lock(mutex);

        while (!completed) {
            if (cancellation.stop_requested()) {
                return std::unexpected("Opening Lampo was cancelled.");
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return std::unexpected("Timed out while opening Lampo.");
            }
            completion.wait_for(lock, kPollInterval);
        }

        if (!failure.empty()) {
            return std::unexpected(failure);
        }
        return {};
    }

    [[nodiscard]] bool is_running() const {
        if (process_identifier <= 0) {
            return false;
        }
        if (::kill(process_identifier, 0) == 0) {
            return true;
        }
        return errno == EPERM;
    }

    std::mutex mutex;
    std::condition_variable completion;
    bool completed = false;
    pid_t process_identifier = -1;
    std::string failure;
};

std::string_view bundle_identifier() noexcept {
    return kBundleIdentifier;
}

std::string_view client_name() noexcept {
    return kClientName;
}

std::expected<CliSession, std::string> CliSession::create(
    const CliProtocol& protocol,
    std::string_view session_id,
    std::string_view content) {
    auto session = begin(protocol, session_id);
    if (!session) {
        return std::unexpected(std::move(session.error()));
    }
    if (auto written = session->write_text(content); !written) {
        session->discard();
        return std::unexpected(std::move(written.error()));
    }
    session->announce();
    return session;
}

std::expected<CliSession, std::string> CliSession::create_from_file(
    const CliProtocol& protocol,
    std::string_view session_id,
    const std::filesystem::path& source) {
    auto session = begin(protocol, session_id);
    if (!session) {
        return std::unexpected(std::move(session.error()));
    }
    if (auto written = session->write_copy_of(source); !written) {
        session->discard();
        return std::unexpected(std::move(written.error()));
    }
    session->announce();
    return session;
}

CliSession::~CliSession() {
    discard();
}

CliSession::CliSession(CliSession&& other) noexcept {
    *this = std::move(other);
}

CliSession& CliSession::operator=(CliSession&& other) noexcept {
    if (this != &other) {
        discard();
        directory_ = std::exchange(other.directory_, {});
        file_ = std::exchange(other.file_, {});
        session_id_ = std::exchange(other.session_id_, {});
        pasteboard_name_ = std::exchange(other.pasteboard_name_, {});
        pasteboard_type_ = std::exchange(other.pasteboard_type_, {});
        reply_ = std::exchange(other.reply_, {});
        observed_change_count_ = std::exchange(other.observed_change_count_, 0);
        launch_ = std::move(other.launch_);
    }
    return *this;
}

std::expected<CliSession, std::string> CliSession::begin(
    const CliProtocol& protocol,
    std::string_view session_id) {
    std::error_code ec;
    const std::filesystem::path temp_root = std::filesystem::temp_directory_path(ec);
    if (ec) {
        return std::unexpected(
            std::format("Could not locate the temporary directory: {}", ec.message()));
    }

    const std::filesystem::path directory =
        temp_root / (std::string(protocol.directory_prefix) + std::string(session_id));
    if (!std::filesystem::create_directory(directory, ec)) {
        return std::unexpected(
            std::format("Could not create the Lampo session: {}", ec.message()));
    }

    std::filesystem::permissions(
        directory,
        std::filesystem::perms::owner_all,
        std::filesystem::perm_options::replace,
        ec);
    if (ec) {
        std::filesystem::remove_all(directory, ec);
        return std::unexpected("Could not secure the Lampo session.");
    }

    CliSession session;
    session.directory_ = std::move(directory);
    session.file_ = session.directory_ / protocol.file_name;
    session.session_id_ = session_id;
    session.pasteboard_name_ =
        std::string(protocol.pasteboard_prefix) + std::string(session_id);
    session.pasteboard_type_ = protocol.pasteboard_type;
    session.reply_ = protocol.reply;
    return session;
}

std::expected<void, std::string> CliSession::write_text(std::string_view content) {
    std::ofstream output(file_, std::ios::binary | std::ios::trunc);
    if (!output) {
        return std::unexpected("Could not write the Lampo payload.");
    }
    output << content;
    if (!output) {
        return std::unexpected("Could not write the Lampo payload.");
    }
    output.close();
    return secure_payload();
}

std::expected<void, std::string> CliSession::write_copy_of(const std::filesystem::path& source) {
    std::error_code ec;
    if (!std::filesystem::copy_file(source, file_, std::filesystem::copy_options::none, ec)) {
        return std::unexpected(
            std::format("Could not prepare the Lampo payload: {}", ec.message()));
    }
    return secure_payload();
}

std::expected<void, std::string> CliSession::secure_payload() {
    std::error_code ec;
    std::filesystem::permissions(
        file_,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace,
        ec);
    if (ec) {
        return std::unexpected("Could not secure the Lampo payload.");
    }
    return {};
}

void CliSession::announce() {
    publish(state::request);
    @autoreleasepool {
        observed_change_count_ = static_cast<long>(pasteboard(pasteboard_name_).changeCount);
    }
}

void CliSession::publish(std::string_view published_state, std::string_view error) {
    @autoreleasepool {
        NSPasteboard* board = pasteboard(pasteboard_name_);
        NSMutableDictionary<NSString*, NSString*>* message = [@{
            @"version": to_ns(kProtocolVersion),
            @"session": to_ns(session_id_),
            @"state": to_ns(published_state),
        } mutableCopy];
        if (!error.empty()) {
            message[@"error"] = to_ns(error);
        }
        // A client names itself when it asks, which is the only message Lampo
        // reads a client name from.
        if (published_state == state::request) {
            message[@"client"] = to_ns(kClientName);
            if (!reply_.empty()) {
                message[@"reply"] = to_ns(reply_);
            }
        }

        [board clearContents];
        [board setPropertyList:message forType:NSPasteboardType(to_ns(pasteboard_type_))];
    }
}

std::expected<void, std::string> CliSession::launch(std::stop_token cancellation) {
    @autoreleasepool {
        NSURL* application = [[NSWorkspace sharedWorkspace]
            URLForApplicationWithBundleIdentifier:to_ns(kBundleIdentifier)];
        if (application == nil) {
            return std::unexpected(
                "Lampo is not installed. Install Lampo, or choose another option in /settings.");
        }

        NSWorkspaceOpenConfiguration* configuration = [NSWorkspaceOpenConfiguration configuration];
        configuration.activates = YES;
        configuration.addsToRecentItems = NO;
        configuration.createsNewApplicationInstance = NO;
        configuration.allowsRunningApplicationSubstitution = NO;

        // Shared with the completion handler, which can outlive a session that
        // stopped waiting for it.
        auto launch = std::make_shared<Launch>();
        launch_ = launch;
        [[NSWorkspace sharedWorkspace]
                        openURLs:@[[NSURL fileURLWithPath:to_ns(file_.string())]]
            withApplicationAtURL:application
                   configuration:configuration
               completionHandler:^(NSRunningApplication* running_application, NSError* error) {
                   @autoreleasepool {
                       launch->complete(running_application, error);
                   }
               }];

        return launch->wait(std::move(cancellation));
    }
}

std::optional<CliMessage> CliSession::poll() {
    @autoreleasepool {
        NSPasteboard* board = pasteboard(pasteboard_name_);
        const long current = static_cast<long>(board.changeCount);
        if (current == observed_change_count_) {
            return std::nullopt;
        }
        observed_change_count_ = current;

        id value = [board propertyListForType:NSPasteboardType(to_ns(pasteboard_type_))];
        if (![value isKindOfClass:NSDictionary.class]) {
            return std::nullopt;
        }
        NSDictionary* message = static_cast<NSDictionary*>(value);
        NSString* version = message[@"version"];
        NSString* session = message[@"session"];
        NSString* published_state = message[@"state"];
        if (![version isKindOfClass:NSString.class]
            || ![session isKindOfClass:NSString.class]
            || ![published_state isKindOfClass:NSString.class]
            || ![version isEqualToString:to_ns(kProtocolVersion)]
            || ![session isEqualToString:to_ns(session_id_)]) {
            return std::nullopt;
        }

        CliMessage result{.state = to_std(published_state)};
        if (NSString* error = message[@"error"]; [error isKindOfClass:NSString.class]) {
            result.error = to_std(error);
        }
        return result;
    }
}

void CliSession::cancel() {
    publish(state::cancelled);
}

bool CliSession::app_running() const {
    return launch_ != nullptr && launch_->is_running();
}

void CliSession::idle(std::stop_token cancellation, std::chrono::milliseconds interval) {
    for (auto remaining = interval;
         remaining.count() > 0 && !cancellation.stop_requested();
         remaining -= kStopCheckSlice) {
        std::this_thread::sleep_for(std::min(remaining, kStopCheckSlice));
    }
}

void CliSession::discard() noexcept {
    if (!pasteboard_name_.empty()) {
        @autoreleasepool {
            [pasteboard(pasteboard_name_) releaseGlobally];
        }
        pasteboard_name_.clear();
    }
    if (!directory_.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(directory_, ec);
        directory_.clear();
        file_.clear();
    }
    launch_.reset();
}

} // namespace tui::lampo
