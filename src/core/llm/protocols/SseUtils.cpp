#include "SseUtils.hpp"
#include "../../utils/StringUtils.hpp"

namespace core::llm::protocols::sse {

bool parse_envelope(std::string_view raw_event,
                    EnvelopeView& out,
                    std::string& data_scratch,
                    bool allow_unframed_payload) {
    out = {};
    data_scratch.clear();

    std::string_view first_data_line;
    bool saw_data = false;

    std::size_t start = 0;
    while (start <= raw_event.size()) {
        const std::size_t end =
            (start < raw_event.size()) ? raw_event.find('\n', start) : std::string_view::npos;

        std::string_view line = (end == std::string_view::npos)
            ? raw_event.substr(start)
            : raw_event.substr(start, end - start);

        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

        if (line.starts_with("event:")) {
            std::string_view event_part = line.substr(6);
            if (!event_part.empty() && event_part.front() == ' ') event_part.remove_prefix(1);
            out.event = core::utils::str::trim_ascii_view(event_part);
        } else if (line.starts_with("data:")) {
            std::string_view data_part = line.substr(5);
            if (!data_part.empty() && data_part.front() == ' ') data_part.remove_prefix(1);
            if (!saw_data) {
                first_data_line = data_part;
                out.data = first_data_line;
                saw_data = true;
            } else if (data_scratch.empty()) {
                data_scratch.reserve(first_data_line.size() + 1 + data_part.size());
                data_scratch.append(first_data_line);
                data_scratch.push_back('\n');
                data_scratch.append(data_part);
                out.data = data_scratch;
            } else {
                data_scratch.push_back('\n');
                data_scratch.append(data_part);
                out.data = data_scratch;
            }
        }

        if (end == std::string_view::npos) break;
        start = end + 1;
    }

    if (saw_data) {
        out.data = core::utils::str::trim_ascii_view(out.data);
        return !out.data.empty();
    }

    if (!allow_unframed_payload) return false;

    const std::string_view trimmed = core::utils::str::trim_ascii_view(raw_event);
    if (trimmed == "[DONE]" || (!trimmed.empty() && (trimmed.front() == '{' || trimmed.front() == '['))) {
        out.data = trimmed;
        return true;
    }
    return false;
}

bool is_done_sentinel(std::string_view payload) noexcept {
    return core::utils::str::trim_ascii_view(payload) == "[DONE]";
}

bool parse_event_payload(std::string_view raw_event,
                         ParsedEventView& out,
                         std::string& data_scratch,
                         bool allow_unframed_payload) {
    out = {};

    EnvelopeView envelope;
    if (!parse_envelope(raw_event, envelope, data_scratch, allow_unframed_payload)) return false;

    out.event   = envelope.event;
    out.data    = envelope.data;
    // parse_envelope already returns trimmed data, so this avoids a second trim.
    out.is_done = (out.data == "[DONE]");
    return true;
}

bool parse_event_payload(std::string_view raw_event,
                         ParsedEventView& out,
                         bool allow_unframed_payload) {
    thread_local std::string data_scratch;
    return parse_event_payload(raw_event, out, data_scratch, allow_unframed_payload);
}

} // namespace core::llm::protocols::sse
