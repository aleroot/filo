#pragma once

#include <cstdint>
#include <format>
#include <random>
#include <string>

namespace core::utils {

[[nodiscard]] inline std::string random_uuid_v4() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<std::uint32_t> dist(0, 0xffffffffU);

    const std::uint32_t a = dist(rng);
    const std::uint32_t b = dist(rng);
    const std::uint32_t c = (dist(rng) & 0x0fffU) | 0x4000U;
    const std::uint32_t d = (dist(rng) & 0x3fffU) | 0x8000U;
    const std::uint32_t e = dist(rng);
    const std::uint32_t f = dist(rng);

    return std::format(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:04x}{:08x}",
        a,
        b & 0xffffU,
        c & 0xffffU,
        d & 0xffffU,
        e & 0xffffU,
        f);
}

} // namespace core::utils
