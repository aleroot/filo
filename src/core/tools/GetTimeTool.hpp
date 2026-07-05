#pragma once

#include "Tool.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonWriter.hpp"

#include <chrono>
#include <ctime>
#include <format>
#include <string>

namespace core::tools {

class GetTimeTool : public Tool {
public:
    ToolDefinition get_definition() const override {
        return {
            .name        = std::string(names::kGetCurrentTime),
            .title       = "Get Current Time",
            .description =
                "Returns the user's current local date, time, and UTC offset. Use it "
                "only when a task depends on the current clock or calendar context. "
                "Fields: 'time' is HH:MM:SS.microseconds, 'date' is YYYY-MM-DD, and "
                "'timezone' is a numeric offset such as +0400.",
            .output_schema =
                R"({"type":"object","properties":{"time":{"type":"string","description":"Local time formatted as HH:MM:SS.microseconds."},"date":{"type":"string","description":"Local date formatted as YYYY-MM-DD."},"timezone":{"type":"string","description":"Local UTC offset formatted as +HHMM or -HHMM."}},"required":["time","date","timezone"],"additionalProperties":false})",
            .annotations = { .read_only_hint = true, .idempotent_hint = false },
        };
    }

    std::string execute([[maybe_unused]] const std::string& json_args, [[maybe_unused]] const core::context::SessionContext& context) override {
        auto now = std::chrono::system_clock::now();
        const auto seconds = std::chrono::floor<std::chrono::seconds>(now);
        const auto fractional =
            std::chrono::duration_cast<std::chrono::microseconds>(now - seconds).count();

        const auto tt = std::chrono::system_clock::to_time_t(seconds);
        std::tm local_tm{};
#if defined(_WIN32)
        localtime_s(&local_tm, &tt);
#else
        localtime_r(&tt, &local_tm);
#endif

        char date_buf[sizeof("YYYY-MM-DD")] = {};
        std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", &local_tm);

        char timezone_offset[16] = {};
        std::strftime(timezone_offset, sizeof(timezone_offset), "%z", &local_tm);
        const std::string timezone = timezone_offset[0] == '\0' ? "local" : timezone_offset;

        core::utils::JsonWriter w(128);
        {
            auto _obj = w.object();
            w.kv_str("time", std::format(
                "{:02d}:{:02d}:{:02d}.{:06d}",
                local_tm.tm_hour,
                local_tm.tm_min,
                local_tm.tm_sec,
                fractional));
            w.comma().kv_str("date", date_buf);
            w.comma().kv_str("timezone", timezone);
        }
        return std::move(w).take();
    }
};

} // namespace core::tools
