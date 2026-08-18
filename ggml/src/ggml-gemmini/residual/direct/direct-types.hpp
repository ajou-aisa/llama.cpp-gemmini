#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ggml::gemmini::residual {

struct ResidualEvent {
    size_t local_row = 0;
    size_t original_k = 0;
    int32_t residual = 0;
};

inline bool operator==(const ResidualEvent &lhs, const ResidualEvent &rhs) {
    return lhs.local_row == rhs.local_row && lhs.original_k == rhs.original_k &&
        lhs.residual == rhs.residual;
}

// Immutable, self-contained CPU compensation input for one activation stripe.
struct DirectStripePayload {
    size_t stripe_id = 0;
    size_t row_begin = 0;
    size_t row_count = 0;
    size_t logical_k = 0;
    size_t logical_j = 0;
    std::vector<ResidualEvent> events;
};

using DirectStripePayloadHandle = std::shared_ptr<const DirectStripePayload>;

}
