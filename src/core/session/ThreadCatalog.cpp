#include "ThreadCatalog.hpp"

#include "core/llm/Models.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <charconv>
#include <ctime>
#include <format>
#include <string>

namespace core::session {
namespace {

[[nodiscard]] int parse_int_field(std::string_view token) {
    int value = 0;
    const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size()) {
        return -1;
    }
    return value;
}

} // namespace

std::string_view to_string(ThreadRecencyGroup group) noexcept {
    switch (group) {
    case ThreadRecencyGroup::Today:     return "Today";
    case ThreadRecencyGroup::Yesterday: return "Yesterday";
    case ThreadRecencyGroup::LastWeek:  return "Last week";
    case ThreadRecencyGroup::LastMonth: return "Last month";
    case ThreadRecencyGroup::Older:     return "Older";
    }
    return "Older";
}

std::string collapse_preview(std::string_view text, std::size_t max_chars) {
    std::string preview =
        core::utils::str::collapse_ascii_whitespace_copy(text);
    if (preview.size() <= max_chars) {
        return preview;
    }
    if (max_chars > 3) {
        preview.resize(max_chars - 3);
        preview += "...";
    } else {
        preview.resize(max_chars);
    }
    return preview;
}

std::string first_user_message_preview(
    const std::vector<core::llm::Message>& messages,
    std::size_t max_chars) {
    for (const auto& message : messages) {
        if (message.role != "user" || message.synthetic) {
            continue;
        }
        const std::string display = core::llm::message_text_for_display(message);
        auto preview = collapse_preview(display, max_chars);
        if (!preview.empty()) {
            return preview;
        }
    }
    return {};
}

std::string thread_display_title(const SessionInfo& info) {
    if (!info.name.empty()) {
        return info.name;
    }
    if (!info.preview.empty()) {
        return info.preview;
    }
    if (!info.session_id.empty()) {
        return std::format("thread {}", info.session_id);
    }
    return "untitled thread";
}

std::optional<std::chrono::system_clock::time_point>
parse_iso8601_utc(std::string_view iso) {
    // Accept "YYYY-MM-DDTHH:MM:SSZ" and the space-separated form used in UI.
    if (iso.size() < 19) {
        return std::nullopt;
    }

    const int year  = parse_int_field(iso.substr(0, 4));
    const int month = parse_int_field(iso.substr(5, 2));
    const int day   = parse_int_field(iso.substr(8, 2));
    const int hour  = parse_int_field(iso.substr(11, 2));
    const int min   = parse_int_field(iso.substr(14, 2));
    const int sec   = parse_int_field(iso.substr(17, 2));
    if (year < 1970 || month < 1 || month > 12 || day < 1 || day > 31
        || hour < 0 || hour > 23 || min < 0 || min > 59 || sec < 0 || sec > 60) {
        return std::nullopt;
    }

    std::tm tm{};
    tm.tm_year  = year - 1900;
    tm.tm_mon   = month - 1;
    tm.tm_mday  = day;
    tm.tm_hour  = hour;
    tm.tm_min   = min;
    tm.tm_sec   = sec;
    tm.tm_isdst = 0;

#if defined(_WIN32)
    const auto tt = _mkgmtime(&tm);
#else
    const auto tt = timegm(&tm);
#endif
    if (tt == static_cast<std::time_t>(-1)) {
        return std::nullopt;
    }
    return std::chrono::system_clock::from_time_t(tt);
}

std::string format_relative_time(
    std::string_view iso_timestamp,
    std::chrono::system_clock::time_point now) {
    const auto tp = parse_iso8601_utc(iso_timestamp);
    if (!tp.has_value()) {
        if (iso_timestamp.empty()) {
            return "—";
        }
        // Fall back to a trimmed raw timestamp.
        std::string raw{iso_timestamp};
        if (raw.size() > 10 && raw[10] == 'T') {
            raw[10] = ' ';
        }
        if (raw.size() > 16) {
            raw.resize(16);
        }
        return raw;
    }

    const auto delta = now - *tp;
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(delta).count();
    if (secs < 0) {
        return "just now";
    }
    if (secs < 45) {
        return "just now";
    }
    if (secs < 90) {
        return "1m ago";
    }
    if (secs < 3600) {
        return std::format("{}m ago", secs / 60);
    }
    if (secs < 90 * 60) {
        return "1h ago";
    }
    if (secs < 24 * 3600) {
        return std::format("{}h ago", secs / 3600);
    }
    if (secs < 48 * 3600) {
        return "yesterday";
    }
    if (secs < 30 * 24 * 3600) {
        return std::format("{}d ago", secs / (24 * 3600));
    }

    const auto tt = std::chrono::system_clock::to_time_t(*tp);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    return std::format("{:04d}-{:02d}-{:02d}",
                       tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
}

ThreadRecencyGroup thread_recency_group(
    std::string_view iso_timestamp,
    std::chrono::system_clock::time_point now) {
    const auto tp = parse_iso8601_utc(iso_timestamp);
    if (!tp.has_value()) {
        return ThreadRecencyGroup::Older;
    }
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(now - *tp).count();
    if (secs < 0) {
        return ThreadRecencyGroup::Today;
    }
    if (secs < 24 * 3600) {
        return ThreadRecencyGroup::Today;
    }
    if (secs < 48 * 3600) {
        return ThreadRecencyGroup::Yesterday;
    }
    if (secs < 7 * 24 * 3600) {
        return ThreadRecencyGroup::LastWeek;
    }
    if (secs < 30 * 24 * 3600) {
        return ThreadRecencyGroup::LastMonth;
    }
    return ThreadRecencyGroup::Older;
}

std::vector<SessionInfo> filter_threads(
    const std::vector<SessionInfo>& sessions,
    std::string_view query) {
    if (query.empty()) {
        return sessions;
    }

    std::vector<SessionInfo> filtered;
    filtered.reserve(sessions.size());
    for (const auto& session : sessions) {
        if (core::utils::str::contains_case_insensitive(session.session_id, query)
            || core::utils::str::contains_case_insensitive(session.name, query)
            || core::utils::str::contains_case_insensitive(session.preview, query)
            || core::utils::str::contains_case_insensitive(session.working_dir, query)
            || core::utils::str::contains_case_insensitive(session.provider, query)
            || core::utils::str::contains_case_insensitive(session.model, query)
            || core::utils::str::contains_case_insensitive(session.mode, query)) {
            filtered.push_back(session);
        }
    }
    return filtered;
}

void order_active_threads(std::vector<SessionInfo>& threads,
                          std::string_view main_session_id) {
    std::ranges::sort(threads, [main_session_id](const SessionInfo& lhs,
                                                 const SessionInfo& rhs) {
        const bool lhs_is_main = lhs.session_id == main_session_id;
        const bool rhs_is_main = rhs.session_id == main_session_id;
        if (lhs_is_main != rhs_is_main) {
            return lhs_is_main;
        }
        if (lhs.last_active_at != rhs.last_active_at) {
            return lhs.last_active_at > rhs.last_active_at;
        }
        return lhs.session_id < rhs.session_id;
    });
}

std::vector<ThreadGroup> group_threads_by_recency(
    const std::vector<SessionInfo>& sessions,
    std::chrono::system_clock::time_point now) {
    std::vector<ThreadGroup> groups;
    groups.reserve(5);

    auto ensure_group = [&](ThreadRecencyGroup kind) -> ThreadGroup& {
        for (auto& group : groups) {
            if (group.group == kind) {
                return group;
            }
        }
        groups.push_back(ThreadGroup{
            .group = kind,
            .label = std::string{to_string(kind)},
            .indices = {},
        });
        return groups.back();
    };

    for (std::size_t i = 0; i < sessions.size(); ++i) {
        const auto& session = sessions[i];
        const std::string_view stamp = session.last_active_at.empty()
            ? std::string_view{session.created_at}
            : std::string_view{session.last_active_at};
        auto& group = ensure_group(thread_recency_group(stamp, now));
        group.indices.push_back(i);
    }
    return groups;
}

bool is_current_thread(const SessionInfo& info,
                       std::string_view current_session_id) noexcept {
    return !current_session_id.empty() && info.session_id == current_session_id;
}

} // namespace core::session
