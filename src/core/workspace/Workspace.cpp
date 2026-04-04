#include "Workspace.hpp"
#include <algorithm>

namespace core::workspace {

void Workspace::initialize(std::filesystem::path primary, std::vector<std::filesystem::path> additional, bool enforce) {
    std::error_code ec;
    primary_ = std::filesystem::weakly_canonical(primary, ec);
    if (ec) primary_ = std::filesystem::absolute(primary, ec);

    additional_.clear();
    for (const auto& dir : additional) {
        auto abs_dir = std::filesystem::weakly_canonical(dir, ec);
        if (ec) abs_dir = std::filesystem::absolute(dir, ec);
        additional_.push_back(std::move(abs_dir));
    }
    enforce_ = enforce;
}

bool Workspace::is_path_allowed(const std::filesystem::path& target_path) const {
    if (!enforce_ || primary_.empty()) {
        return true; // Not initialized or disabled
    }

    std::error_code ec;
    auto abs_target = std::filesystem::weakly_canonical(target_path, ec);
    if (ec) {
        abs_target = std::filesystem::absolute(target_path, ec);
    }

    auto is_subpath = [](const std::filesystem::path& root, const std::filesystem::path& target) {
        const auto r = root.lexically_normal();
        const auto t = target.lexically_normal();

        auto r_it = r.begin();
        auto t_it = t.begin();
        while (r_it != r.end() && t_it != t.end()) {
            if (*r_it != *t_it) {
                return false;
            }
            ++r_it;
            ++t_it;
        }
        return r_it == r.end();
    };

    if (is_subpath(primary_, abs_target)) {
        return true;
    }

    for (const auto& add : additional_) {
        if (is_subpath(add, abs_target)) {
            return true;
        }
    }

    return false;
}

} // namespace core::workspace
