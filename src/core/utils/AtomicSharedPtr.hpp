#pragma once

// ---------------------------------------------------------------------------
// Lock-free shared_ptr snapshot slots.
//
// The publish/subscribe snapshots in this tree (registry state, provider usage
// and rate-limit info) are written under a mutex by one thread and read
// lock-free by others.  C++20 deprecated the free
// std::atomic_{load,store}_explicit(shared_ptr*) overloads in favour of
// std::atomic<std::shared_ptr<T>>, and libstdc++ warns loudly about them, but
// libc++ (Apple clang) still has no atomic<shared_ptr> specialisation — its
// std::atomic<T> hard-errors on non-trivially-copyable T.
//
// So the choice has to stay compile-time.  Everything is funnelled through this
// header: one alias for the storage slot plus acquire/release accessors, with
// the unavoidable deprecated fallback suppressed in exactly one place.
// ---------------------------------------------------------------------------

#include <atomic>
#include <memory>
#include <utility>
#include <version>

namespace core::utils {

#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
#define FILO_HAS_ATOMIC_SHARED_PTR 1
#else
#define FILO_HAS_ATOMIC_SHARED_PTR 0
#endif

#if FILO_HAS_ATOMIC_SHARED_PTR
/// Storage for a shared_ptr snapshot that is published and consumed atomically.
template <typename T>
using AtomicSharedPtr = std::atomic<std::shared_ptr<T>>;
#else
template <typename T>
using AtomicSharedPtr = std::shared_ptr<T>;
#endif

/// Read the currently published snapshot (acquire ordering).
template <typename T>
[[nodiscard]] std::shared_ptr<T> atomic_load_acquire(
    const AtomicSharedPtr<T>& slot) noexcept {
#if FILO_HAS_ATOMIC_SHARED_PTR
    return slot.load(std::memory_order_acquire);
#else
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    return std::atomic_load_explicit(&slot, std::memory_order_acquire);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
}

/// Publish a new snapshot (release ordering).
template <typename T>
void atomic_store_release(AtomicSharedPtr<T>& slot,
                          std::shared_ptr<T> snapshot) noexcept {
#if FILO_HAS_ATOMIC_SHARED_PTR
    slot.store(std::move(snapshot), std::memory_order_release);
#else
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    std::atomic_store_explicit(&slot, std::move(snapshot), std::memory_order_release);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
}

} // namespace core::utils
