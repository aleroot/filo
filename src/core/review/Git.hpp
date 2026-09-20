#pragma once

#include "Types.hpp"

#include <expected>
#include <optional>
#include <string>
#include <string_view>

namespace core::review {

class GitClient {
public:
    [[nodiscard]] static bool is_repository(std::string* error_detail = nullptr);

    [[nodiscard]] static std::optional<std::string> rev_parse(std::string_view ref);

    [[nodiscard]] static std::optional<std::string>
    merge_base_with_head(std::string_view branch);

    [[nodiscard]] static std::string uncommitted_base_ref();

    [[nodiscard]] static std::expected<GitSnapshot, std::string>
    collect_uncommitted();

    [[nodiscard]] static std::expected<GitSnapshot, std::string>
    collect_staged();

    [[nodiscard]] static std::expected<GitSnapshot, std::string>
    collect_against_ref(std::string_view ref);

    [[nodiscard]] static std::expected<GitSnapshot, std::string>
    collect_commit(std::string_view sha);
};

} // namespace core::review
