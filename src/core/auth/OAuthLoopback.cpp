#include "OAuthLoopback.hpp"

#include "core/utils/StringUtils.hpp"

#include <httplib.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace core::auth {
namespace {

[[nodiscard]] std::string normalize_redirect_host(std::string_view bind_host,
                                                  std::string_view redirect_host) {
    std::string host = core::utils::str::trim_ascii_copy(
        redirect_host.empty() ? bind_host : redirect_host);
    if (host.empty()
        || host == "0.0.0.0"
        || host == "::"
        || host == "[::]"
        || host == "::0") {
        host = "127.0.0.1";
    }
    if (host.find(':') != std::string::npos
        && !(host.starts_with('[') && host.ends_with(']'))) {
        host = "[" + host + "]";
    }
    return host;
}

[[nodiscard]] std::string normalize_callback_path(std::string_view path) {
    if (path.empty()) return "/callback";
    if (path.front() == '/') return std::string(path);
    return std::string("/") + std::string(path);
}

[[nodiscard]] std::string url_decode_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        const char c = value[i];
        if (c == '+') {
            out.push_back(' ');
            continue;
        }
        if (c == '%' && i + 2 < value.size()) {
            const auto hex = value.substr(i + 1, 2);
            char* end = nullptr;
            const long byte = std::strtol(std::string(hex).c_str(), &end, 16);
            if (end && *end == '\0') {
                out.push_back(static_cast<char>(byte));
                i += 2;
                continue;
            }
        }
        out.push_back(c);
    }
    return out;
}

} // namespace

struct OAuthLoopbackServer::Impl {
    OAuthLoopbackOptions options;
    httplib::Server server;
    std::thread thread;
    int port = -1;
    std::string redirect_uri;
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    OAuthLoopbackResult result;
    bool started = false;
};

OAuthLoopbackServer::OAuthLoopbackServer(OAuthLoopbackOptions options)
    : impl_(std::make_unique<Impl>()) {
    impl_->options = std::move(options);
    const std::string path = normalize_callback_path(impl_->options.callback_path);

    const auto handler = [this](const httplib::Request& req, httplib::Response& res) {
        const std::string state = req.get_param_value("state");
        if (impl_->options.expected_state.has_value()
            && state != *impl_->options.expected_state) {
            res.status = 400;
            res.set_content(
                "Login failed: OAuth state mismatch - possible CSRF attempt",
                "text/plain");
            std::lock_guard lock(impl_->mutex);
            impl_->result.error = "OAuth state mismatch - possible CSRF attempt";
            impl_->done = true;
            impl_->cv.notify_one();
            return;
        }

        const std::string err = req.get_param_value("error");
        if (!err.empty()) {
            const std::string desc = req.get_param_value("error_description");
            res.set_content("Login failed: " + err + " — " + desc, "text/plain");
            std::lock_guard lock(impl_->mutex);
            impl_->result.error = err + (desc.empty() ? "" : (": " + desc));
            impl_->done = true;
            impl_->cv.notify_one();
            return;
        }

        const std::string code = req.get_param_value("code");
        std::string validation_error;
        if (code.empty()) {
            validation_error = "authorization callback did not include a code";
        }

        if (!validation_error.empty()) {
            res.status = 400;
            res.set_content("Login failed: " + validation_error, "text/plain");
            std::lock_guard lock(impl_->mutex);
            impl_->result.error = std::move(validation_error);
            impl_->done = true;
            impl_->cv.notify_one();
            return;
        }

        res.set_content(impl_->options.success_html, "text/html");
        std::lock_guard lock(impl_->mutex);
        impl_->result.code = code;
        impl_->result.state = state;
        impl_->done = true;
        impl_->cv.notify_one();
    };

    impl_->server.Get(path, handler);
    for (const auto& extra : impl_->options.extra_paths) {
        impl_->server.Get(normalize_callback_path(extra), handler);
    }

    if (impl_->options.fixed_port.has_value()) {
        const int candidate = *impl_->options.fixed_port;
        if (candidate == 0) {
            impl_->port = impl_->server.bind_to_any_port(impl_->options.bind_host);
        } else if (impl_->server.bind_to_port(impl_->options.bind_host, candidate)) {
            impl_->port = candidate;
        } else {
            throw OAuthLoopbackBindError(
                "OAuth loopback: could not bind "
                + impl_->options.bind_host + ":" + std::to_string(candidate));
        }
    } else if (!impl_->options.candidate_ports.empty()) {
        for (const int candidate : impl_->options.candidate_ports) {
            if (candidate <= 0 || candidate > 65535) continue;
            if (impl_->server.bind_to_port(impl_->options.bind_host, candidate)) {
                impl_->port = candidate;
                break;
            }
        }
    } else {
        for (int candidate = impl_->options.port_start;
             candidate <= impl_->options.port_end;
             ++candidate) {
            if (impl_->server.bind_to_port(impl_->options.bind_host, candidate)) {
                impl_->port = candidate;
                break;
            }
        }
    }

    if (impl_->port < 0) {
        throw OAuthLoopbackBindError("OAuth loopback: no free port available");
    }

    const std::string host =
        normalize_redirect_host(impl_->options.bind_host, impl_->options.redirect_host);
    impl_->redirect_uri =
        "http://" + host + ":" + std::to_string(impl_->port) + path;
}

OAuthLoopbackServer::~OAuthLoopbackServer() {
    stop();
}

int OAuthLoopbackServer::port() const noexcept {
    return impl_ ? impl_->port : -1;
}

const std::string& OAuthLoopbackServer::redirect_uri() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->redirect_uri : empty;
}

void OAuthLoopbackServer::start() {
    if (!impl_ || impl_->started) return;
    impl_->started = true;
    impl_->thread = std::thread([this]() { impl_->server.listen_after_bind(); });
    for (int i = 0; i < 200 && !impl_->server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!impl_->server.is_running()) {
        stop();
        throw std::runtime_error("OAuth loopback: callback server failed to start");
    }
}

OAuthLoopbackResult OAuthLoopbackServer::wait() {
    if (!impl_) {
        return {.timed_out = true};
    }
    if (!impl_->started) {
        start();
    }

    bool received = false;
    {
        std::unique_lock lock(impl_->mutex);
        received = impl_->cv.wait_for(
            lock, impl_->options.timeout, [&] { return impl_->done; });
    }
    stop();

    std::lock_guard lock(impl_->mutex);
    if (!received) {
        impl_->result.timed_out = true;
    }
    return impl_->result;
}

void OAuthLoopbackServer::stop() noexcept {
    if (!impl_) return;
    try {
        impl_->server.stop();
    } catch (...) {
    }
    if (impl_->thread.joinable()) {
        try {
            impl_->thread.join();
        } catch (...) {
        }
    }
    impl_->started = false;
}

OAuthManualAuthInput parse_oauth_manual_auth_input(std::string_view raw_input) {
    const std::string input = core::utils::str::trim_ascii_copy(raw_input);
    if (input.empty()) {
        throw std::runtime_error("No authorization code provided");
    }

    const bool looks_like_url =
        input.starts_with("http://") || input.starts_with("https://")
        || input.find("code=") != std::string::npos;
    if (!looks_like_url) {
        return {.code = input, .state = std::nullopt};
    }

    std::string_view query = input;
    const auto qmark = input.find('?');
    if (qmark != std::string::npos) {
        query = std::string_view(input).substr(qmark + 1);
    }
    if (query.empty()) {
        throw std::runtime_error("Callback URL does not contain query parameters");
    }

    OAuthManualAuthInput parsed{};
    std::size_t pos = 0;
    while (pos < query.size()) {
        const auto amp = query.find('&', pos);
        const auto piece = query.substr(
            pos, amp == std::string_view::npos ? std::string_view::npos : amp - pos);
        const auto eq = piece.find('=');
        const auto key = url_decode_component(
            eq == std::string_view::npos ? piece : piece.substr(0, eq));
        const auto value = url_decode_component(
            eq == std::string_view::npos ? std::string_view{} : piece.substr(eq + 1));
        if (key == "code") parsed.code = value;
        else if (key == "state") parsed.state = value;
        else if (key == "error" && !value.empty()) {
            throw std::runtime_error("Authorization error in callback: " + value);
        }
        if (amp == std::string_view::npos) break;
        pos = amp + 1;
    }

    if (parsed.code.empty()) {
        throw std::runtime_error("Callback URL did not include an authorization code");
    }
    return parsed;
}

bool complete_oauth_loopback_with_manual_fallback(
    OAuthLoopbackResult& result,
    std::string_view expected_state,
    ui::AuthUI* ui,
    std::string_view prompt_label) {
    if (!result.error.empty()) {
        return false;
    }
    if (!result.code.empty()) {
        if (!result.state.empty() && !expected_state.empty()
            && result.state != expected_state) {
            result.error = "state mismatch (possible CSRF)";
            return false;
        }
        return true;
    }
    if (!ui) {
        return false;
    }

    ui->show_instructions(
        "Automatic browser callback was not received. Paste the full redirect "
        "URL or just the authorization code.");
    try {
        const auto manual = parse_oauth_manual_auth_input(ui->prompt_secret(std::string(prompt_label)));
        result.code = manual.code;
        if (manual.state.has_value()) {
            result.state = *manual.state;
        } else {
            result.state = std::string(expected_state);
        }
        result.timed_out = false;
        result.error.clear();
        if (!result.state.empty() && !expected_state.empty()
            && result.state != expected_state) {
            result.error = "state mismatch (possible CSRF)";
            return false;
        }
        return !result.code.empty();
    } catch (const std::exception& e) {
        result.error = e.what();
        return false;
    }
}

} // namespace core::auth
