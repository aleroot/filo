#include "LampoPromptEditor.hpp"

#import <AppKit/AppKit.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>

#include <signal.h>

// Lampo's Prompter CLI protocol, v1.
//
// Filo publishes a request on a private, per-session pasteboard and opens the
// draft with Lampo; Lampo answers on the same pasteboard with `editing`, then
// `saved` / `cancelled` / `failed`. Filo is one client of that public protocol —
// Lampo knows nothing about Filo.
namespace tui::editor {
namespace {

constexpr std::string_view kBundleIdentifier = "alessio.pollero.Lampo";
constexpr std::string_view kDirectoryPrefix = "lampo-prompter-edit-v1-";
constexpr std::string_view kPromptFileName = "prompt.md";
constexpr std::string_view kPasteboardNamePrefix = "alessio.pollero.Lampo.prompter-edit.";
constexpr std::string_view kPasteboardType = "alessio.pollero.Lampo.prompter-edit.v1";
constexpr std::string_view kClientName = "Filo";

constexpr auto kPollInterval = std::chrono::milliseconds(100);
constexpr auto kLaunchTimeout = std::chrono::seconds(15);
// If Lampo never acknowledges, the user is stuck staring at an overlay; fail
// loudly instead and point them back at /settings.
constexpr auto kAcknowledgementTimeout = std::chrono::seconds(15);

[[nodiscard]] NSString* to_ns(std::string_view value) {
    return [[NSString alloc] initWithBytes:value.data()
                                    length:static_cast<NSUInteger>(value.size())
                                  encoding:NSUTF8StringEncoding];
}

[[nodiscard]] std::string to_std(NSString* value) {
    return value == nil ? std::string{} : std::string(value.UTF8String);
}

// Adapts Filo's editor-neutral scratch buffer to Lampo's versioned file shape.
// Keeping the envelope here prevents the generic prompt-editor path from
// depending on one backend's transport protocol.
class LampoEditBuffer {
public:
    [[nodiscard]] static std::expected<std::unique_ptr<LampoEditBuffer>, std::string> create(
        const std::filesystem::path& source,
        std::string_view session_id) {
        std::error_code ec;
        const std::filesystem::path temp_root = std::filesystem::temp_directory_path(ec);
        if (ec) {
            return std::unexpected(
                std::format("Could not locate the temporary directory: {}", ec.message()));
        }

        const std::filesystem::path directory =
            temp_root / (std::string(kDirectoryPrefix) + std::string(session_id));
        if (!std::filesystem::create_directory(directory, ec)) {
            return std::unexpected(
                std::format("Could not create the Lampo edit session: {}", ec.message()));
        }

        std::filesystem::permissions(
            directory,
            std::filesystem::perms::owner_all,
            std::filesystem::perm_options::replace,
            ec);
        if (ec) {
            std::filesystem::remove_all(directory, ec);
            return std::unexpected("Could not secure the Lampo edit session.");
        }

        const std::filesystem::path file = directory / kPromptFileName;
        if (!std::filesystem::copy_file(source, file, std::filesystem::copy_options::none, ec)) {
            std::filesystem::remove_all(directory, ec);
            return std::unexpected(
                std::format("Could not prepare the Lampo prompt: {}", ec.message()));
        }

        std::filesystem::permissions(
            file,
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace,
            ec);
        if (ec) {
            std::filesystem::remove_all(directory, ec);
            return std::unexpected("Could not secure the Lampo prompt.");
        }

        return std::unique_ptr<LampoEditBuffer>(
            new LampoEditBuffer(source, directory, file));
    }

    ~LampoEditBuffer() {
        std::error_code ec;
        std::filesystem::remove_all(directory_, ec);
    }

    LampoEditBuffer(const LampoEditBuffer&) = delete;
    LampoEditBuffer& operator=(const LampoEditBuffer&) = delete;

    [[nodiscard]] const std::filesystem::path& file() const noexcept { return file_; }

    [[nodiscard]] std::string commit() const {
        std::error_code ec;
        if (!std::filesystem::copy_file(
                file_,
                source_,
                std::filesystem::copy_options::overwrite_existing,
                ec)) {
            return std::format("Could not restore the prompt from Lampo: {}", ec.message());
        }
        return {};
    }

private:
    LampoEditBuffer(
        std::filesystem::path source,
        std::filesystem::path directory,
        std::filesystem::path file)
        : source_(std::move(source)),
          directory_(std::move(directory)),
          file_(std::move(file)) {}

    std::filesystem::path source_;
    std::filesystem::path directory_;
    std::filesystem::path file_;
};

void publish(NSPasteboard* pasteboard,
             NSString* session_id,
             NSString* state,
             NSString* error = nil,
             NSString* client = nil) {
    NSMutableDictionary<NSString*, NSString*>* message = [@{
        @"version": @"1",
        @"session": session_id,
        @"state": state,
    } mutableCopy];
    if (error.length > 0) {
        message[@"error"] = error;
    }
    if (client.length > 0) {
        message[@"client"] = client;
    }

    [pasteboard clearContents];
    [pasteboard setPropertyList:message forType:NSPasteboardType(to_ns(kPasteboardType))];
}

// Owns the private pasteboard for one edit so it is released on every exit
// path, including the cancelled and failed ones.
class PasteboardChannel {
public:
    explicit PasteboardChannel(std::string_view session_id)
        : session_id_(to_ns(session_id)),
          pasteboard_([NSPasteboard pasteboardWithName:NSPasteboardName(to_ns(
              std::string(kPasteboardNamePrefix) + std::string(session_id)))]) {}

    ~PasteboardChannel() { [pasteboard_ releaseGlobally]; }

    PasteboardChannel(const PasteboardChannel&) = delete;
    PasteboardChannel& operator=(const PasteboardChannel&) = delete;

    [[nodiscard]] NSPasteboard* pasteboard() const noexcept { return pasteboard_; }
    [[nodiscard]] NSString* session_id() const noexcept { return session_id_; }
    [[nodiscard]] NSString* name() const noexcept { return pasteboard_.name; }

    [[nodiscard]] long change_count() const noexcept {
        return static_cast<long>(pasteboard_.changeCount);
    }

    void request() { publish(pasteboard_, session_id_, @"request", nil, to_ns(kClientName)); }
    void cancel() { publish(pasteboard_, session_id_, @"cancelled"); }

    // Returns the state reported by Lampo, or nil when nothing addressed to
    // this session has arrived yet.
    [[nodiscard]] NSDictionary* poll(long& observed_change_count) const {
        const long current = change_count();
        if (current == observed_change_count) {
            return nil;
        }
        observed_change_count = current;

        id value = [pasteboard_ propertyListForType:NSPasteboardType(to_ns(kPasteboardType))];
        if (![value isKindOfClass:NSDictionary.class]) {
            return nil;
        }
        NSDictionary* message = static_cast<NSDictionary*>(value);
        NSString* version = message[@"version"];
        NSString* session = message[@"session"];
        NSString* state = message[@"state"];
        if (![version isKindOfClass:NSString.class]
            || ![session isKindOfClass:NSString.class]
            || ![state isKindOfClass:NSString.class]
            || ![version isEqualToString:@"1"]
            || ![session isEqualToString:session_id_]) {
            return nil;
        }
        return message;
    }

private:
    NSString* session_id_ = nil;
    NSPasteboard* pasteboard_ = nil;
};

// Bridges NSWorkspace's asynchronous launch result to the editor worker and
// retains the exact Lampo instance that owns this session.
class ApplicationLaunch {
public:
    void complete(NSRunningApplication* application, NSError* error) {
        {
            const std::lock_guard lock(mutex_);
            if (completed_) {
                return;
            }
            completed_ = true;
            process_identifier_ = application.processIdentifier;
            if (error != nil) {
                error_ = to_std(error.localizedDescription);
            } else if (application == nil) {
                error_ = "Lampo did not return a running application.";
            }
        }
        completion_.notify_all();
    }

    [[nodiscard]] std::expected<void, std::string> wait(std::stop_token cancellation) {
        const auto deadline = std::chrono::steady_clock::now() + kLaunchTimeout;
        std::unique_lock lock(mutex_);

        while (!completed_) {
            if (cancellation.stop_requested()) {
                return std::unexpected("Opening Lampo was cancelled.");
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return std::unexpected("Timed out while opening Lampo.");
            }
            completion_.wait_for(lock, kPollInterval);
        }

        if (!error_.empty()) {
            return std::unexpected(error_);
        }
        return {};
    }

    [[nodiscard]] bool is_running() const {
        if (process_identifier_ <= 0) {
            return false;
        }
        if (::kill(process_identifier_, 0) == 0) {
            return true;
        }
        return errno == EPERM;
    }

private:
    std::mutex mutex_;
    std::condition_variable completion_;
    bool completed_ = false;
    pid_t process_identifier_ = -1;
    std::string error_;
};

class LampoPromptEditor final : public PromptEditor {
public:
    [[nodiscard]] EditorDescriptor descriptor() const noexcept override {
        return lampo_prompt_editor_descriptor();
    }

    [[nodiscard]] EditResult edit(const EditContext& context) override {
        @autoreleasepool {
            auto buffer = LampoEditBuffer::create(context.file, context.session_id);
            if (!buffer) {
                return EditResult::failed(std::move(buffer.error()));
            }

            PasteboardChannel channel(context.session_id);
            channel.request();
            long observed_change_count = channel.change_count();

            context.report_progress(EditProgress::Launching);
            auto application = launch((*buffer)->file(), context.cancellation);
            if (!application) {
                channel.cancel();
                if (context.cancellation.stop_requested()) {
                    return EditResult::cancelled();
                }
                return EditResult::failed(std::move(application.error()));
            }

            EditResult result =
                await_completion(channel, context, observed_change_count, **application);
            if (result.status() == EditStatus::Saved) {
                if (auto error = (*buffer)->commit(); !error.empty()) {
                    return EditResult::failed(std::move(error));
                }
            }
            return result;
        }
    }

private:
    // Asks the workspace to open the draft with Lampo and retains the exact app
    // instance selected by Launch Services.
    [[nodiscard]] static std::expected<std::shared_ptr<ApplicationLaunch>, std::string> launch(
        const std::filesystem::path& file,
        std::stop_token cancellation) {
        NSURL* application = [[NSWorkspace sharedWorkspace]
            URLForApplicationWithBundleIdentifier:to_ns(kBundleIdentifier)];
        if (application == nil) {
            return std::unexpected(
                "Lampo is not installed. Choose System in /settings, or install Lampo.");
        }

        NSWorkspaceOpenConfiguration* configuration = [NSWorkspaceOpenConfiguration configuration];
        configuration.activates = YES;
        configuration.addsToRecentItems = NO;
        configuration.createsNewApplicationInstance = NO;
        configuration.allowsRunningApplicationSubstitution = NO;

        auto launch = std::make_shared<ApplicationLaunch>();
        [[NSWorkspace sharedWorkspace]
                        openURLs:@[[NSURL fileURLWithPath:to_ns(file.string())]]
            withApplicationAtURL:application
                   configuration:configuration
               completionHandler:^(NSRunningApplication* running_application, NSError* error) {
                   launch->complete(running_application, error);
               }];

        if (auto result = launch->wait(cancellation); !result) {
            return std::unexpected(std::move(result.error()));
        }
        return launch;
    }

    [[nodiscard]] static EditResult await_completion(
        PasteboardChannel& channel,
        const EditContext& context,
        long observed_change_count,
        const ApplicationLaunch& application) {
        bool acknowledged = false;
        const auto deadline = std::chrono::steady_clock::now() + kAcknowledgementTimeout;

        while (!context.cancellation.stop_requested()) {
            @autoreleasepool {
                if (NSDictionary* message = channel.poll(observed_change_count); message != nil) {
                    NSString* state = message[@"state"];

                    if ([state isEqualToString:@"editing"]) {
                        acknowledged = true;
                        context.report_progress(EditProgress::Editing);
                    } else if ([state isEqualToString:@"saved"]) {
                        return EditResult::saved();
                    } else if ([state isEqualToString:@"cancelled"]) {
                        return EditResult::cancelled();
                    } else if ([state isEqualToString:@"failed"]) {
                        NSString* error = message[@"error"];
                        return EditResult::failed(
                            error.length > 0
                                ? to_std(error)
                                : "Lampo could not complete the edit.");
                    }
                }

                // Quitting the exact Lampo instance at any point cancels the edit.
                if (!application.is_running()) {
                    return EditResult::cancelled();
                }
                if (!acknowledged && std::chrono::steady_clock::now() >= deadline) {
                    channel.cancel();
                    return EditResult::failed(
                        "Lampo did not acknowledge the editing session. "
                        "Update Lampo, or choose System in /settings.");
                }
            }
            std::this_thread::sleep_for(kPollInterval);
        }

        channel.cancel();
        return EditResult::cancelled();
    }
};

} // namespace

EditorDescriptor lampo_prompt_editor_descriptor() noexcept {
    return EditorDescriptor{
        .id = "lampo",
        .label = "Lampo",
        .description = "Edit the draft in the Lampo macOS app.",
        .launch = LaunchMode::Detached,
    };
}

std::unique_ptr<PromptEditor> make_lampo_prompt_editor() {
    return std::make_unique<LampoPromptEditor>();
}

} // namespace tui::editor
