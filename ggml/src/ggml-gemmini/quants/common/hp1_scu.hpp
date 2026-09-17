#pragma once

#include <cstdint>
#include <limits>

namespace ggml::gemmini::quants::hp1 {
// Exact reference for the existing hardware contract. No floating point and no
// materialized 2^e factor for unbounded architectural exponents.
constexpr std::uint32_t zero_carrier = 0x80000000u;
constexpr bool valid_carrier(std::uint32_t carrier) noexcept {
  return carrier <= 32767u || carrier == zero_carrier;
}
constexpr bool encode_carrier(std::int16_t exponent,
                              std::uint32_t &out) noexcept {
  if (exponent == std::numeric_limits<std::int16_t>::min())
    out = zero_carrier;
  else if (exponent >= 0)
    out = static_cast<std::uint32_t>(exponent);
  else
    return false;
  return true;
}
constexpr std::int32_t sat32(std::int64_t value) noexcept {
  return value > INT32_MAX   ? INT32_MAX
         : value < INT32_MIN ? INT32_MIN
                             : static_cast<std::int32_t>(value);
}
// Precondition: a validated carrier and a signed32 raw hardware fragment.
constexpr std::int32_t apply_validated(std::int32_t partial,
                                       std::uint32_t carrier) noexcept {
  if (carrier == zero_carrier || partial == 0)
    return 0;
  if (carrier >= 32)
    return partial < 0 ? INT32_MIN : INT32_MAX;
  return sat32(static_cast<std::int64_t>(partial) *
               (std::int64_t{1} << carrier));
}
constexpr std::int32_t accumulate(std::int32_t acc,
                                  std::int32_t value) noexcept {
  return sat32(static_cast<std::int64_t>(acc) + value);
}
} // namespace ggml::gemmini::quants::hp1
