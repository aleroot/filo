#include "Crypto.hpp"

#include <algorithm>
#include <chrono>
#include <random>

namespace core::utils::crypto {
namespace {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

[[nodiscard]] std::uint32_t rotate_right(std::uint32_t value,
                                          std::uint32_t count) noexcept {
    return (value >> count) | (value << (32u - count));
}

[[nodiscard]] std::array<std::uint8_t, 32> sha256_bytes(
    std::span<const std::uint8_t> message) {
    std::array<std::uint32_t, 8> hash{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    std::vector<std::uint8_t> data(message.begin(), message.end());
    const std::uint64_t bit_length = static_cast<std::uint64_t>(message.size()) * 8u;
    data.push_back(0x80u);
    while (data.size() % 64u != 56u) data.push_back(0x00u);
    for (int i = 7; i >= 0; --i) {
        data.push_back(static_cast<std::uint8_t>((bit_length >> (i * 8)) & 0xffu));
    }

    for (std::size_t offset = 0; offset < data.size(); offset += 64u) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] = (std::uint32_t(data[offset + i * 4u]) << 24u)
                     | (std::uint32_t(data[offset + i * 4u + 1u]) << 16u)
                     | (std::uint32_t(data[offset + i * 4u + 2u]) << 8u)
                     | std::uint32_t(data[offset + i * 4u + 3u]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const auto s0 = rotate_right(words[i - 15], 7u)
                          ^ rotate_right(words[i - 15], 18u)
                          ^ (words[i - 15] >> 3u);
            const auto s1 = rotate_right(words[i - 2], 17u)
                          ^ rotate_right(words[i - 2], 19u)
                          ^ (words[i - 2] >> 10u);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        auto a = hash[0]; auto b = hash[1]; auto c = hash[2]; auto d = hash[3];
        auto e = hash[4]; auto f = hash[5]; auto g = hash[6]; auto h = hash[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const auto upper_sigma1 = rotate_right(e, 6u)
                                    ^ rotate_right(e, 11u)
                                    ^ rotate_right(e, 25u);
            const auto choose = (e & f) ^ (~e & g);
            const auto temp1 = h + upper_sigma1 + choose
                             + kSha256RoundConstants[i] + words[i];
            const auto upper_sigma0 = rotate_right(a, 2u)
                                    ^ rotate_right(a, 13u)
                                    ^ rotate_right(a, 22u);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = upper_sigma0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
        hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
    }

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t i = 0; i < hash.size(); ++i) {
        digest[i * 4u] = static_cast<std::uint8_t>((hash[i] >> 24u) & 0xffu);
        digest[i * 4u + 1u] = static_cast<std::uint8_t>((hash[i] >> 16u) & 0xffu);
        digest[i * 4u + 2u] = static_cast<std::uint8_t>((hash[i] >> 8u) & 0xffu);
        digest[i * 4u + 3u] = static_cast<std::uint8_t>(hash[i] & 0xffu);
    }
    return digest;
}

// ---------------------------------------------------------------------------
// GF(2^255 - 19) field arithmetic for X25519 (RFC 7748 §5)
//
// Field elements use the standard 10-limb representation with the alternating
// 25.5-bit radix: value = sum(limb[i] * 2^offset[i]). Limbs are *signed* so
// subtraction never needs a borrow pass, and the whole file stays on 64-bit
// integer arithmetic (no 128-bit extension types, which are not portable).
// ---------------------------------------------------------------------------

using FieldElement = std::array<std::int64_t, 10>;

constexpr std::array<int, 10> kLimbBits{26, 25, 26, 25, 26, 25, 26, 25, 26, 25};
constexpr std::array<int, 10> kLimbOffset{0, 26, 51, 77, 102, 128, 153, 179, 204, 230};

[[nodiscard]] FieldElement fe_zero() noexcept {
    return FieldElement{};
}

[[nodiscard]] FieldElement fe_one() noexcept {
    FieldElement out{};
    out[0] = 1;
    return out;
}

/**
 * Reduce each limb back into roughly +/-2^(bits-1) using round-to-nearest
 * carries. Two sweeps are enough: the first can leave a large carry folded
 * into limb 0 (2^255 == 19 mod p), the second flattens it.
 */
void fe_carry(FieldElement& h) noexcept {
    for (int sweep = 0; sweep < 2; ++sweep) {
        for (std::size_t i = 0; i < h.size(); ++i) {
            const int bits = kLimbBits[i];
            const std::int64_t carry =
                (h[i] + (std::int64_t{1} << (bits - 1))) >> bits;
            h[i] -= carry << bits;
            if (i == h.size() - 1) {
                h[0] += 19 * carry;
            } else {
                h[i + 1] += carry;
            }
        }
    }
}

[[nodiscard]] FieldElement fe_add(const FieldElement& f,
                                   const FieldElement& g) noexcept {
    FieldElement h{};
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = f[i] + g[i];
    return h;
}

[[nodiscard]] FieldElement fe_sub(const FieldElement& f,
                                   const FieldElement& g) noexcept {
    FieldElement h{};
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = f[i] - g[i];
    return h;
}

/**
 * Schoolbook multiply in the 25.5-bit radix. Two odd-indexed limbs multiply
 * to a weight one bit above the destination limb (offset[i] + offset[j] ==
 * offset[i + j] + 1 exactly when i and j are both odd), hence the doubling.
 * Everything at or above limb 10 folds down by 19, since 2^255 == 19 mod p.
 */
[[nodiscard]] FieldElement fe_mul(const FieldElement& f,
                                   const FieldElement& g) noexcept {
    std::array<std::int64_t, 19> product{};
    for (std::size_t i = 0; i < f.size(); ++i) {
        for (std::size_t j = 0; j < g.size(); ++j) {
            std::int64_t term = f[i] * g[j];
            if ((i % 2u) == 1u && (j % 2u) == 1u) term *= 2;
            product[i + j] += term;
        }
    }

    FieldElement h{};
    for (std::size_t i = 0; i < h.size(); ++i) {
        h[i] = product[i];
        if (i + 10u < product.size()) h[i] += 19 * product[i + 10u];
    }
    fe_carry(h);
    return h;
}

[[nodiscard]] FieldElement fe_square(const FieldElement& f) noexcept {
    return fe_mul(f, f);
}

/** Multiply by a small positive constant (the ladder needs a24 = 121665). */
[[nodiscard]] FieldElement fe_mul_small(const FieldElement& f,
                                         std::int64_t scalar) noexcept {
    FieldElement h{};
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = f[i] * scalar;
    fe_carry(h);
    return h;
}

/** Exponentiate by p-2 = 2^255 - 21, i.e. every bit set except bits 2 and 4. */
[[nodiscard]] FieldElement fe_invert(const FieldElement& z) noexcept {
    FieldElement out = z;
    for (int bit = 253; bit >= 0; --bit) {
        out = fe_square(out);
        if (bit != 2 && bit != 4) out = fe_mul(out, z);
    }
    return out;
}

/** Branch-free conditional swap: exchanges f and g when `swap` is 1. */
void fe_conditional_swap(FieldElement& f, FieldElement& g,
                         unsigned swap) noexcept {
    const std::uint64_t mask = 0u - static_cast<std::uint64_t>(swap & 1u);
    for (std::size_t i = 0; i < f.size(); ++i) {
        const auto left = static_cast<std::uint64_t>(f[i]);
        const auto right = static_cast<std::uint64_t>(g[i]);
        const std::uint64_t difference = mask & (left ^ right);
        f[i] = static_cast<std::int64_t>(left ^ difference);
        g[i] = static_cast<std::int64_t>(right ^ difference);
    }
}

[[nodiscard]] FieldElement fe_from_bytes(
    std::span<const std::uint8_t, 32> bytes) noexcept {
    std::array<std::uint8_t, 32> buffer{};
    std::copy(bytes.begin(), bytes.end(), buffer.begin());
    buffer[31] &= 0x7fu; // RFC 7748: the high bit of u is always masked off.

    FieldElement h{};
    for (std::size_t i = 0; i < h.size(); ++i) {
        std::int64_t limb = 0;
        for (int b = 0; b < kLimbBits[i]; ++b) {
            const int bit = kLimbOffset[i] + b;
            const auto value =
                static_cast<std::int64_t>((buffer[bit / 8] >> (bit % 8)) & 1u);
            limb |= value << b;
        }
        h[i] = limb;
    }
    return h;
}

[[nodiscard]] std::array<std::uint8_t, 32> fe_to_bytes(FieldElement h) noexcept {
    fe_carry(h);

    // Add 2p (which is 0 mod p) so every limb is positive before the floor
    // carries below: p = 2^255 - 19 is all-ones limbs with 19 taken off limb 0.
    for (std::size_t i = 0; i < h.size(); ++i) {
        h[i] += 2 * ((std::int64_t{1} << kLimbBits[i]) - 1);
    }
    h[0] -= 2 * 18;

    for (int sweep = 0; sweep < 3; ++sweep) {
        for (std::size_t i = 0; i < h.size(); ++i) {
            const int bits = kLimbBits[i];
            const std::int64_t carry = h[i] >> bits; // floor division
            h[i] -= carry << bits;
            if (i == h.size() - 1) {
                h[0] += 19 * carry;
            } else {
                h[i + 1] += carry;
            }
        }
    }

    std::array<std::uint8_t, 32> packed{};
    for (std::size_t i = 0; i < h.size(); ++i) {
        for (int b = 0; b < kLimbBits[i]; ++b) {
            const int bit = kLimbOffset[i] + b;
            const auto value = static_cast<std::uint8_t>((h[i] >> b) & 1);
            packed[bit / 8] |= static_cast<std::uint8_t>(value << (bit % 8));
        }
    }

    // The value is now in [0, 2^255). Subtract p exactly once if it fits:
    // adding 19 overflows into bit 255 precisely when the value is >= p.
    std::array<std::uint8_t, 32> reduced{};
    std::uint32_t carry = 19u;
    for (std::size_t i = 0; i < reduced.size(); ++i) {
        const std::uint32_t sum = static_cast<std::uint32_t>(packed[i]) + carry;
        reduced[i] = static_cast<std::uint8_t>(sum & 0xffu);
        carry = sum >> 8u;
    }
    const auto mask =
        static_cast<std::uint8_t>(0u - static_cast<std::uint32_t>(reduced[31] >> 7u));
    reduced[31] &= 0x7fu;
    for (std::size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<std::uint8_t>((reduced[i] & mask)
                                              | (packed[i] & static_cast<std::uint8_t>(~mask)));
    }
    return packed;
}

// ---------------------------------------------------------------------------
// AES-256 (FIPS 197): Nk = 8, Nr = 14. Encryption only; GCM never decrypts
// blocks, it XORs a CTR keystream.
// ---------------------------------------------------------------------------

/**
 * Build the AES S-box from its definition (multiplicative inverse in
 * GF(2^8) followed by the affine transform) instead of transcribing 256
 * magic bytes — a table typo is invisible in review, this is not.
 */
[[nodiscard]] constexpr std::array<std::uint8_t, 256> build_aes_sbox() {
    std::array<std::uint8_t, 256> exponentials{};
    std::array<std::uint8_t, 256> logarithms{};

    std::uint8_t value = 1;
    for (int i = 0; i < 255; ++i) {
        exponentials[static_cast<std::size_t>(i)] = value;
        logarithms[value] = static_cast<std::uint8_t>(i);
        // value *= 3 in GF(2^8) with the AES reduction polynomial 0x11b.
        const auto doubled = static_cast<std::uint8_t>(
            (value << 1u) ^ ((value & 0x80u) != 0u ? 0x1bu : 0x00u));
        value = static_cast<std::uint8_t>(doubled ^ value);
    }

    std::array<std::uint8_t, 256> sbox{};
    for (std::size_t i = 0; i < sbox.size(); ++i) {
        std::uint8_t inverse = 0;
        if (i != 0) {
            // The log of 1 is 0, so the exponent wraps at 255 (== exp[0] == 1).
            inverse = exponentials[static_cast<std::size_t>(
                (255 - logarithms[i]) % 255)];
        }
        std::uint8_t transformed = inverse;
        for (int rotation = 1; rotation <= 4; ++rotation) {
            transformed = static_cast<std::uint8_t>(
                transformed
                ^ static_cast<std::uint8_t>((inverse << rotation)
                                            | (inverse >> (8 - rotation))));
        }
        sbox[i] = static_cast<std::uint8_t>(transformed ^ 0x63u);
    }
    return sbox;
}

constexpr std::array<std::uint8_t, 256> kAesSbox = build_aes_sbox();

constexpr std::array<std::uint8_t, 7> kAesRoundConstants{
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
};

constexpr std::size_t kAesRounds = 14;
constexpr std::size_t kAesScheduleWords = 4u * (kAesRounds + 1u); // 60

using AesKeySchedule = std::array<std::uint32_t, kAesScheduleWords>;

[[nodiscard]] std::uint32_t aes_sub_word(std::uint32_t word) noexcept {
    return (std::uint32_t{kAesSbox[(word >> 24u) & 0xffu]} << 24u)
         | (std::uint32_t{kAesSbox[(word >> 16u) & 0xffu]} << 16u)
         | (std::uint32_t{kAesSbox[(word >> 8u) & 0xffu]} << 8u)
         | std::uint32_t{kAesSbox[word & 0xffu]};
}

[[nodiscard]] AesKeySchedule aes_256_expand_key(
    std::span<const std::uint8_t, 32> key) noexcept {
    constexpr std::size_t key_words = 8; // Nk
    AesKeySchedule schedule{};
    for (std::size_t i = 0; i < key_words; ++i) {
        schedule[i] = (std::uint32_t{key[i * 4u]} << 24u)
                    | (std::uint32_t{key[i * 4u + 1u]} << 16u)
                    | (std::uint32_t{key[i * 4u + 2u]} << 8u)
                    | std::uint32_t{key[i * 4u + 3u]};
    }
    for (std::size_t i = key_words; i < schedule.size(); ++i) {
        std::uint32_t temp = schedule[i - 1u];
        if (i % key_words == 0u) {
            temp = (temp << 8u) | (temp >> 24u); // RotWord
            temp = aes_sub_word(temp);
            temp ^= std::uint32_t{kAesRoundConstants[i / key_words - 1u]} << 24u;
        } else if (i % key_words == 4u) {
            temp = aes_sub_word(temp);
        }
        schedule[i] = schedule[i - key_words] ^ temp;
    }
    return schedule;
}

[[nodiscard]] std::uint8_t aes_xtime(std::uint8_t value) noexcept {
    return static_cast<std::uint8_t>((value << 1u)
                                     ^ ((value & 0x80u) != 0u ? 0x1bu : 0x00u));
}

void aes_add_round_key(std::array<std::uint8_t, 16>& state,
                       const AesKeySchedule& schedule,
                       std::size_t round) noexcept {
    for (std::size_t column = 0; column < 4; ++column) {
        const std::uint32_t word = schedule[round * 4u + column];
        state[column * 4u] ^= static_cast<std::uint8_t>((word >> 24u) & 0xffu);
        state[column * 4u + 1u] ^= static_cast<std::uint8_t>((word >> 16u) & 0xffu);
        state[column * 4u + 2u] ^= static_cast<std::uint8_t>((word >> 8u) & 0xffu);
        state[column * 4u + 3u] ^= static_cast<std::uint8_t>(word & 0xffu);
    }
}

void aes_shift_rows(std::array<std::uint8_t, 16>& state) noexcept {
    const std::array<std::uint8_t, 16> source = state;
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t row = 0; row < 4; ++row) {
            state[column * 4u + row] = source[((column + row) % 4u) * 4u + row];
        }
    }
}

void aes_mix_columns(std::array<std::uint8_t, 16>& state) noexcept {
    for (std::size_t column = 0; column < 4; ++column) {
        const std::uint8_t a0 = state[column * 4u];
        const std::uint8_t a1 = state[column * 4u + 1u];
        const std::uint8_t a2 = state[column * 4u + 2u];
        const std::uint8_t a3 = state[column * 4u + 3u];
        state[column * 4u] = static_cast<std::uint8_t>(
            aes_xtime(a0) ^ (aes_xtime(a1) ^ a1) ^ a2 ^ a3);
        state[column * 4u + 1u] = static_cast<std::uint8_t>(
            a0 ^ aes_xtime(a1) ^ (aes_xtime(a2) ^ a2) ^ a3);
        state[column * 4u + 2u] = static_cast<std::uint8_t>(
            a0 ^ a1 ^ aes_xtime(a2) ^ (aes_xtime(a3) ^ a3));
        state[column * 4u + 3u] = static_cast<std::uint8_t>(
            (aes_xtime(a0) ^ a0) ^ a1 ^ a2 ^ aes_xtime(a3));
    }
}

[[nodiscard]] std::array<std::uint8_t, 16> aes_encrypt_block(
    const AesKeySchedule& schedule,
    const std::array<std::uint8_t, 16>& input) noexcept {
    std::array<std::uint8_t, 16> state = input;
    aes_add_round_key(state, schedule, 0);
    for (std::size_t round = 1; round < kAesRounds; ++round) {
        for (auto& byte : state) byte = kAesSbox[byte];
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, schedule, round);
    }
    for (auto& byte : state) byte = kAesSbox[byte];
    aes_shift_rows(state);
    aes_add_round_key(state, schedule, kAesRounds);
    return state;
}

// ---------------------------------------------------------------------------
// GCM (NIST SP 800-38D)
// ---------------------------------------------------------------------------

using Block = std::array<std::uint8_t, 16>;

/**
 * GF(2^128) multiplication in GCM's bit-reflected convention: bit 0 of the
 * block is the most significant coefficient, so the reduction shifts right
 * and folds in 0xe1 (the reversed 0x87 of x^128 + x^7 + x^2 + x + 1).
 */
[[nodiscard]] Block ghash_multiply(const Block& x, const Block& y) noexcept {
    Block result{};
    Block shifted = y;
    for (std::size_t bit = 0; bit < 128; ++bit) {
        const auto selected = static_cast<std::uint8_t>(
            0u - static_cast<std::uint32_t>((x[bit / 8u] >> (7u - bit % 8u)) & 1u));
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] ^= static_cast<std::uint8_t>(shifted[i] & selected);
        }

        const auto low_bit = static_cast<std::uint8_t>(
            0u - static_cast<std::uint32_t>(shifted[15] & 1u));
        for (std::size_t i = 15; i > 0; --i) {
            shifted[i] = static_cast<std::uint8_t>((shifted[i] >> 1u)
                                                   | (shifted[i - 1u] << 7u));
        }
        shifted[0] = static_cast<std::uint8_t>(shifted[0] >> 1u);
        shifted[0] ^= static_cast<std::uint8_t>(0xe1u & low_bit);
    }
    return result;
}

/** Absorb `data` (zero padded to whole blocks) into the running GHASH state. */
void ghash_update(Block& state, const Block& subkey,
                  std::span<const std::uint8_t> data) noexcept {
    for (std::size_t offset = 0; offset < data.size(); offset += 16u) {
        const std::size_t chunk = std::min<std::size_t>(16u, data.size() - offset);
        for (std::size_t i = 0; i < chunk; ++i) state[i] ^= data[offset + i];
        state = ghash_multiply(state, subkey);
    }
}

/** Absorb the trailing 64-bit big-endian bit-length pair. */
void ghash_update_lengths(Block& state, const Block& subkey,
                          std::uint64_t first_bits,
                          std::uint64_t second_bits) noexcept {
    Block trailer{};
    for (std::size_t i = 0; i < 8; ++i) {
        trailer[i] = static_cast<std::uint8_t>((first_bits >> ((7u - i) * 8u)) & 0xffu);
        trailer[8u + i] =
            static_cast<std::uint8_t>((second_bits >> ((7u - i) * 8u)) & 0xffu);
    }
    for (std::size_t i = 0; i < state.size(); ++i) state[i] ^= trailer[i];
    state = ghash_multiply(state, subkey);
}

void increment_counter(Block& counter) noexcept {
    for (std::size_t i = 16; i-- > 12;) {
        if (++counter[i] != 0u) break;
    }
}

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

/** SplitMix64 finalizer: cheap avalanche over the collected entropy words. */
[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

} // namespace

std::array<std::uint8_t, 32> sha256(std::span<const std::uint8_t> data) {
    return sha256_bytes(data);
}

std::array<std::uint8_t, 32> sha256(std::string_view data) {
    return sha256_bytes(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(data.data()), data.size()));
}

std::array<std::uint8_t, 32> x25519(std::span<const std::uint8_t, 32> scalar,
                                    std::span<const std::uint8_t, 32> u_coordinate) {
    std::array<std::uint8_t, 32> clamped{};
    std::copy(scalar.begin(), scalar.end(), clamped.begin());
    clamped[0] &= 248u;
    clamped[31] &= 127u;
    clamped[31] |= 64u;

    const FieldElement u = fe_from_bytes(u_coordinate);
    FieldElement x2 = fe_one();
    FieldElement z2 = fe_zero();
    FieldElement x3 = u;
    FieldElement z3 = fe_one();
    unsigned swap = 0;

    for (int position = 254; position >= 0; --position) {
        const unsigned bit =
            (clamped[static_cast<std::size_t>(position) >> 3u]
             >> (static_cast<unsigned>(position) & 7u)) & 1u;
        swap ^= bit;
        fe_conditional_swap(x2, x3, swap);
        fe_conditional_swap(z2, z3, swap);
        swap = bit;

        const FieldElement a = fe_add(x2, z2);
        const FieldElement aa = fe_square(a);
        const FieldElement b = fe_sub(x2, z2);
        const FieldElement bb = fe_square(b);
        const FieldElement e = fe_sub(aa, bb);
        const FieldElement c = fe_add(x3, z3);
        const FieldElement d = fe_sub(x3, z3);
        const FieldElement da = fe_mul(d, a);
        const FieldElement cb = fe_mul(c, b);

        x3 = fe_square(fe_add(da, cb));
        z3 = fe_mul(u, fe_square(fe_sub(da, cb)));
        x2 = fe_mul(aa, bb);
        z2 = fe_mul(e, fe_add(aa, fe_mul_small(e, 121665)));
    }

    fe_conditional_swap(x2, x3, swap);
    fe_conditional_swap(z2, z3, swap);
    return fe_to_bytes(fe_mul(x2, fe_invert(z2)));
}

std::array<std::uint8_t, 32> x25519_public_key(
    std::span<const std::uint8_t, 32> scalar) {
    std::array<std::uint8_t, 32> basepoint{};
    basepoint[0] = 9u;
    return x25519(scalar, basepoint);
}

std::array<std::uint8_t, 32> x25519_generate_private_key() {
    const std::vector<std::uint8_t> entropy = random_bytes(32);
    std::array<std::uint8_t, 32> scalar{};
    std::copy(entropy.begin(), entropy.end(), scalar.begin());
    scalar[0] &= 248u;
    scalar[31] &= 127u;
    scalar[31] |= 64u;
    return scalar;
}

std::optional<std::string> aes_256_gcm_decrypt(
    std::span<const std::uint8_t, 32> key,
    std::span<const std::uint8_t> nonce,
    std::span<const std::uint8_t> ciphertext,
    std::span<const std::uint8_t, 16> tag,
    std::span<const std::uint8_t> aad) {
    const AesKeySchedule schedule = aes_256_expand_key(key);
    const Block subkey = aes_encrypt_block(schedule, Block{});

    // J0: the 96-bit nonce shortcut, otherwise the GHASH-derived counter.
    Block counter_zero{};
    if (nonce.size() == 12u) {
        std::copy(nonce.begin(), nonce.end(), counter_zero.begin());
        counter_zero[15] = 1u;
    } else {
        ghash_update(counter_zero, subkey, nonce);
        ghash_update_lengths(counter_zero, subkey, 0u,
                             static_cast<std::uint64_t>(nonce.size()) * 8u);
    }

    Block counter = counter_zero;
    increment_counter(counter);

    std::string plaintext;
    plaintext.reserve(ciphertext.size());
    for (std::size_t offset = 0; offset < ciphertext.size(); offset += 16u) {
        const Block keystream = aes_encrypt_block(schedule, counter);
        increment_counter(counter);
        const std::size_t chunk = std::min<std::size_t>(16u, ciphertext.size() - offset);
        for (std::size_t i = 0; i < chunk; ++i) {
            plaintext.push_back(static_cast<char>(ciphertext[offset + i] ^ keystream[i]));
        }
    }

    Block hash{};
    ghash_update(hash, subkey, aad);
    ghash_update(hash, subkey, ciphertext);
    ghash_update_lengths(hash, subkey,
                         static_cast<std::uint64_t>(aad.size()) * 8u,
                         static_cast<std::uint64_t>(ciphertext.size()) * 8u);

    const Block mask = aes_encrypt_block(schedule, counter_zero);
    std::uint8_t difference = 0;
    for (std::size_t i = 0; i < hash.size(); ++i) {
        difference |= static_cast<std::uint8_t>((hash[i] ^ mask[i]) ^ tag[i]);
    }
    if (difference != 0u) return std::nullopt;
    return plaintext;
}

std::vector<std::uint8_t> random_bytes(std::size_t count) {
    // std::random_device is the only portable entropy source available to us;
    // on every platform Filo ships to it is backed by the OS CSPRNG. Some
    // freestanding implementations report entropy() == 0 and degrade to a
    // fixed PRNG sequence. We do not silently accept that and we cannot
    // substitute a better portable source, so we always additionally fold in
    // a high-resolution timestamp and a stack address (ASLR) through a
    // SplitMix64 finalizer. On a real CSPRNG this mixing is harmless; on a
    // degenerate one it is the difference between "reproducible" and merely
    // "weak". It is not a substitute for real entropy.
    std::random_device device;
    const std::uint8_t stack_marker = 0;
    const auto stack_salt =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&stack_marker));

    std::vector<std::uint8_t> out;
    out.reserve(count);
    while (out.size() < count) {
        const std::uint64_t device_word =
            (static_cast<std::uint64_t>(device()) << 32u)
            | static_cast<std::uint64_t>(device());
        const auto tick = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const std::uint64_t word =
            mix64(device_word ^ mix64(tick ^ stack_salt ^ out.size()));
        for (std::size_t i = 0; i < 8u && out.size() < count; ++i) {
            out.push_back(static_cast<std::uint8_t>((word >> (i * 8u)) & 0xffu));
        }
    }
    return out;
}

} // namespace core::utils::crypto
