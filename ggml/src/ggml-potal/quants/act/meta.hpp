#pragma once

// Copied from ggml-gemmini/quants/act/meta.hpp with the algorithms PoTal
// implements: none and tensor (block/exsia/stripe/token removed).

#include "tensor/types.hpp"

#include <variant>
#include <cstdint>
#include <type_traits>

namespace ggml::potal::quants::act {
struct NoneMeta {
    void reset() {}
};

enum class MetaKind : uint8_t {
    none,
    tensor,
};

using MetaStorage = std::variant<NoneMeta, tensor::Meta>;

struct Meta {
    MetaStorage storage_;

    Meta() = default;

    void reset() {
        storage_.emplace<NoneMeta>();
    }

    const MetaStorage & storage() const {
        return storage_;
    }

    MetaStorage & storage() {
        return storage_;
    }

    MetaKind kind() const {
        return std::visit(
            [](const auto & meta) -> MetaKind {
                using T = std::decay_t<decltype(meta)>;
                if constexpr (std::is_same_v<T, NoneMeta>)
                    return MetaKind::none;
                else
                    return MetaKind::tensor;
            },
            storage_);
    }
};
} // namespace ggml::potal::quants::act
