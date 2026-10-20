#pragma once

namespace core::utils::process {

// Whether a process group may still run code, and so still write. A group of
// zombies does not: an orphan waits as one until a reaper collects it, and a
// container's init may never do that. Permission errors, invalid identifiers
// and members that cannot be examined cannot establish quiescence.
[[nodiscard]] bool group_may_exist(int group) noexcept;

} // namespace core::utils::process
