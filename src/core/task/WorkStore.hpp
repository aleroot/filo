#pragma once

#include "../context/SessionContext.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace core::task {

struct WorkItem {
    static constexpr int kVersion = 1;

    int version = kVersion;
    std::string work_id;
    std::string title;
    std::string status;
    std::string worker;
    std::string mode;
    std::string provider;
    std::string model;
    std::string working_dir;
    std::string session_id;
    std::string created_at;
    std::string updated_at;
    std::string summary;
    std::string result;
    std::string handoff_summary;
    int32_t steps = 0;
    int32_t tool_calls = 0;
    int32_t failed_tool_calls = 0;
    std::vector<std::string> files_touched;
    std::vector<std::string> commands_run;
};

struct WorkInfo {
    std::string work_id;
    std::string title;
    std::string status;
    std::string worker;
    std::string provider;
    std::string model;
    std::string working_dir;
    std::string updated_at;
};

class WorkStore {
public:
    explicit WorkStore(std::filesystem::path root_dir);

    bool save(const WorkItem& item, std::string* error = nullptr) const;
    [[nodiscard]] std::optional<WorkItem> load(std::string_view work_id) const;
    [[nodiscard]] std::vector<WorkInfo> list() const;

    [[nodiscard]] const std::filesystem::path& root_dir() const noexcept { return root_dir_; }
    [[nodiscard]] std::filesystem::path compute_path(std::string_view work_id) const;

    [[nodiscard]] static std::filesystem::path default_work_dir();
    [[nodiscard]] static std::filesystem::path
    default_work_dir(const core::context::SessionContext& context);
    [[nodiscard]] static std::string generate_work_id();
    [[nodiscard]] static bool is_valid_work_id(std::string_view work_id) noexcept;

private:
    [[nodiscard]] static std::string to_json(const WorkItem& item);
    [[nodiscard]] static std::optional<WorkItem> from_json(std::string_view json);
    bool ensure_dir(std::string* error = nullptr) const;

    std::filesystem::path root_dir_;
};

} // namespace core::task
