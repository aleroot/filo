#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace core::utils {

class Base64 final {
public:
    Base64() = delete;

    [[nodiscard]] static std::string encode(std::string_view input);
    [[nodiscard]] static std::string encode(std::span<const std::uint8_t> input);

    [[nodiscard]] static std::string encode_url(std::span<const std::uint8_t> input,
                                                bool include_padding = false);

    [[nodiscard]] static std::optional<std::string> decode(std::string_view input);
    [[nodiscard]] static std::optional<std::string> decode_url(std::string_view input);
};

} // namespace core::utils
