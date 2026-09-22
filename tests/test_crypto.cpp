#include <catch2/catch_test_macros.hpp>

#include "core/utils/Crypto.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using namespace core::utils::crypto;

namespace {

[[nodiscard]] std::vector<std::uint8_t> from_hex(std::string_view hex) {
    const auto nibble = [](char c) -> std::uint8_t {
        if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(c - 'a' + 10);
        return static_cast<std::uint8_t>(c - 'A' + 10);
    };
    std::vector<std::uint8_t> out;
    out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        out.push_back(static_cast<std::uint8_t>((nibble(hex[i]) << 4)
                                                | nibble(hex[i + 1])));
    }
    return out;
}

template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> array_from_hex(std::string_view hex) {
    const auto bytes = from_hex(hex);
    std::array<std::uint8_t, N> out{};
    for (std::size_t i = 0; i < N && i < bytes.size(); ++i) out[i] = bytes[i];
    return out;
}

[[nodiscard]] std::string to_hex(std::span<const std::uint8_t> bytes) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const auto byte : bytes) {
        out.push_back(kDigits[byte >> 4]);
        out.push_back(kDigits[byte & 0x0f]);
    }
    return out;
}

template <std::size_t N>
[[nodiscard]] std::string to_hex(const std::array<std::uint8_t, N>& bytes) {
    return to_hex(std::span<const std::uint8_t>(bytes.data(), bytes.size()));
}

[[nodiscard]] std::string to_hex(const std::string& bytes) {
    return to_hex(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// SHA-256 — FIPS 180-4
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SHA-256 matches the FIPS 180-4 vectors", "[crypto][sha256]") {
    CHECK(to_hex(sha256(std::string_view{""}))
          == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(to_hex(sha256(std::string_view{"abc"}))
          == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // 56 bytes — forces the two-block padding path.
    CHECK(to_hex(sha256(std::string_view{
              "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"}))
          == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST_CASE("SHA-256 span and string_view overloads agree", "[crypto][sha256]") {
    const std::vector<std::uint8_t> bytes{'a', 'b', 'c'};
    CHECK(to_hex(sha256(std::span<const std::uint8_t>(bytes)))
          == to_hex(sha256(std::string_view{"abc"})));
}

// ─────────────────────────────────────────────────────────────────────────────
// X25519 — RFC 7748
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("X25519 matches the RFC 7748 section 5.2 vectors", "[crypto][x25519]") {
    CHECK(to_hex(x25519(
              array_from_hex<32>(
                  "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4"),
              array_from_hex<32>(
                  "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c")))
          == "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552");

    CHECK(to_hex(x25519(
              array_from_hex<32>(
                  "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d"),
              array_from_hex<32>(
                  "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493")))
          == "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957");
}

TEST_CASE("X25519 reproduces the RFC 7748 section 6.1 exchange",
          "[crypto][x25519]") {
    const auto alice_private = array_from_hex<32>(
        "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
    const auto bob_private = array_from_hex<32>(
        "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");

    const auto alice_public = x25519_public_key(alice_private);
    const auto bob_public = x25519_public_key(bob_private);

    CHECK(to_hex(alice_public)
          == "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
    CHECK(to_hex(bob_public)
          == "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f");

    // Both sides must derive the same secret.
    CHECK(to_hex(x25519(alice_private, bob_public))
          == "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
    CHECK(to_hex(x25519(bob_private, alice_public))
          == "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
}

TEST_CASE("Generated X25519 private keys are clamped and usable",
          "[crypto][x25519]") {
    const auto private_key = x25519_generate_private_key();
    CHECK((private_key[0] & 0x07) == 0);
    CHECK((private_key[31] & 0x80) == 0);
    CHECK((private_key[31] & 0x40) == 0x40);

    // A fresh pair still agrees with a fixed counterparty.
    const auto peer_private = array_from_hex<32>(
        "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
    const auto peer_public = x25519_public_key(peer_private);
    CHECK(to_hex(x25519(private_key, peer_public))
          == to_hex(x25519(peer_private, x25519_public_key(private_key))));
}

// ─────────────────────────────────────────────────────────────────────────────
// AES-256-GCM — NIST SP 800-38D
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AES-256-GCM decrypts the SP 800-38D zero-key cases",
          "[crypto][aes_gcm]") {
    const auto key = array_from_hex<32>(
        "0000000000000000000000000000000000000000000000000000000000000000");
    const auto iv = from_hex("000000000000000000000000");

    SECTION("case 13 — empty plaintext and AAD") {
        const auto plaintext = aes_256_gcm_decrypt(
            key, iv, {}, array_from_hex<16>("530f8afbc74536b9a963b4f1c4cb738b"));
        REQUIRE(plaintext.has_value());
        CHECK(plaintext->empty());
    }

    SECTION("case 14 — 16 zero bytes") {
        const auto ciphertext = from_hex("cea7403d4d606b6e074ec5d3baf39d18");
        const auto plaintext = aes_256_gcm_decrypt(
            key, iv, ciphertext,
            array_from_hex<16>("d0d1c8a799996bf0265b98b5d48ab919"));
        REQUIRE(plaintext.has_value());
        CHECK(to_hex(*plaintext) == "00000000000000000000000000000000");
    }
}

TEST_CASE("AES-256-GCM handles AAD and a 12-byte IV (SP 800-38D case 16)",
          "[crypto][aes_gcm]") {
    const auto plaintext = aes_256_gcm_decrypt(
        array_from_hex<32>(
            "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308"),
        from_hex("cafebabefacedbaddecaf888"),
        from_hex("522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"
                 "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662"),
        array_from_hex<16>("76fc6ece0f4e1768cddf8853bb2d551b"),
        from_hex("feedfacedeadbeeffeedfacedeadbeefabaddad2"));

    REQUIRE(plaintext.has_value());
    CHECK(to_hex(*plaintext)
          == "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
             "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39");
}

TEST_CASE("AES-256-GCM handles a short IV via the GHASH-derived J0 path",
          "[crypto][aes_gcm]") {
    // SP 800-38D case 17 uses an 8-byte IV, which does not take the
    // IV || 0x00000001 shortcut.
    const auto plaintext = aes_256_gcm_decrypt(
        array_from_hex<32>(
            "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308"),
        from_hex("cafebabefacedbad"),
        from_hex("c3762df1ca787d32ae47c13bf19844cbaf1ae14d0b976afac52ff7d79bba9de0"
                 "feb582d33934a4f0954cc2363bc73f7862ac430e64abe499f47c9b1f"),
        array_from_hex<16>("3a337dbf46a792c45e454913fe2ea8f2"),
        from_hex("feedfacedeadbeeffeedfacedeadbeefabaddad2"));

    REQUIRE(plaintext.has_value());
    CHECK(to_hex(*plaintext)
          == "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
             "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39");
}

TEST_CASE("AES-256-GCM rejects anything that fails authentication",
          "[crypto][aes_gcm]") {
    const auto key = array_from_hex<32>(
        "0000000000000000000000000000000000000000000000000000000000000000");
    const auto iv = from_hex("000000000000000000000000");
    const auto ciphertext = from_hex("cea7403d4d606b6e074ec5d3baf39d18");
    const auto tag = array_from_hex<16>("d0d1c8a799996bf0265b98b5d48ab919");

    SECTION("corrupted tag") {
        auto bad_tag = tag;
        bad_tag[0] ^= 0x01;
        CHECK_FALSE(
            aes_256_gcm_decrypt(key, iv, ciphertext, bad_tag).has_value());
    }

    SECTION("corrupted ciphertext") {
        auto bad_ciphertext = ciphertext;
        bad_ciphertext[0] ^= 0x01;
        CHECK_FALSE(
            aes_256_gcm_decrypt(key, iv, bad_ciphertext, tag).has_value());
    }

    SECTION("wrong key") {
        auto wrong_key = key;
        wrong_key[0] ^= 0x01;
        CHECK_FALSE(
            aes_256_gcm_decrypt(wrong_key, iv, ciphertext, tag).has_value());
    }

    SECTION("unexpected AAD") {
        CHECK_FALSE(aes_256_gcm_decrypt(key, iv, ciphertext, tag,
                                        from_hex("00112233"))
                        .has_value());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// random_bytes
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("random_bytes returns fresh, correctly sized output", "[crypto][random]") {
    CHECK(random_bytes(0).empty());

    const auto first = random_bytes(32);
    const auto second = random_bytes(32);
    REQUIRE(first.size() == 32);
    REQUIRE(second.size() == 32);
    CHECK(first != second);

    // An all-zero draw would indicate a dead entropy source.
    CHECK(std::ranges::any_of(first, [](std::uint8_t b) { return b != 0; }));
}
