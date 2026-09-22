#pragma once

/**
 * @file Crypto.hpp
 * @brief Self-contained cryptographic primitives: SHA-256, X25519, AES-256-GCM.
 *
 * Filo links no OpenSSL, no libsodium, no system crypto library at all. The
 * binary has to build wherever a C++23 toolchain exists, without hunting for
 * platform packages and without inheriting a third-party ABI on every
 * refresh. Everything here is therefore written from scratch in portable
 * C++23 — the same choice already made by the PKCE helper in
 * `core/auth/OAuthPkce.cpp`, whose private SHA-256 this module replaces.
 *
 * These implementations are tuned for reviewability against the published
 * test vectors (FIPS 180-4, RFC 7748 §5.2/§6.1, NIST SP 800-38D), not for
 * throughput. The X25519 ladder uses a masked conditional swap and the GCM
 * tag comparison accumulates XOR differences, so neither branches on secret
 * data; nothing here defends against physical side channels.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::utils::crypto {

/** FIPS 180-4 SHA-256. */
[[nodiscard]] std::array<std::uint8_t, 32> sha256(std::span<const std::uint8_t> data);
[[nodiscard]] std::array<std::uint8_t, 32> sha256(std::string_view data);

/** RFC 7748 X25519 scalar multiplication. Returns the 32-byte shared secret. */
[[nodiscard]] std::array<std::uint8_t, 32> x25519(
    std::span<const std::uint8_t, 32> scalar,
    std::span<const std::uint8_t, 32> u_coordinate);

/** X25519 public key for `scalar` (scalar * basepoint 9). */
[[nodiscard]] std::array<std::uint8_t, 32> x25519_public_key(
    std::span<const std::uint8_t, 32> scalar);

/** Generate a clamped X25519 private scalar from a CSPRNG. */
[[nodiscard]] std::array<std::uint8_t, 32> x25519_generate_private_key();

/**
 * AES-256-GCM authenticated decryption.
 * Returns std::nullopt when the authentication tag does not verify.
 */
[[nodiscard]] std::optional<std::string> aes_256_gcm_decrypt(
    std::span<const std::uint8_t, 32> key,
    std::span<const std::uint8_t> nonce,
    std::span<const std::uint8_t> ciphertext,
    std::span<const std::uint8_t, 16> tag,
    std::span<const std::uint8_t> aad = {});

/** Cryptographically unpredictable bytes. */
[[nodiscard]] std::vector<std::uint8_t> random_bytes(std::size_t count);

} // namespace core::utils::crypto
