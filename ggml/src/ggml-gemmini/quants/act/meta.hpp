#pragma once

#include "block/types.hpp"
#include "exsia/types.hpp"
#include "stripe/types.hpp"
#include "tensor/types.hpp"
#include "token/types.hpp"

#include <variant>
#include <cstdint>
#include <type_traits>

namespace ggml::gemmini::quants::act
{
    struct NoneMeta
    {
        void reset() {}
    };

    enum class MetaKind : uint8_t
    {
        none,
        block,
        exsia,
        stripe,
        tensor,
        token,
    };

    using MetaStorage = std::variant<NoneMeta, block::Meta, exsia::Meta, stripe::Meta, tensor::Meta, token::Meta>;

    struct Meta
    {
        MetaStorage storage_;

        Meta() = default;

        void reset()
        {
            storage_.emplace<NoneMeta>();
        }

        const MetaStorage &storage() const
        {
            return storage_;
        }

        MetaStorage &storage()
        {
            return storage_;
        }

        MetaKind kind() const
        {
            return std::visit([](const auto & meta) -> MetaKind {
                using T = std::decay_t<decltype(meta)>;
                if constexpr (std::is_same_v<T, NoneMeta>) return MetaKind::none;
                else if constexpr (std::is_same_v<T, block::Meta>) return MetaKind::block;
                else if constexpr (std::is_same_v<T, exsia::Meta>) return MetaKind::exsia;
                else if constexpr (std::is_same_v<T, stripe::Meta>) return MetaKind::stripe;
                else if constexpr (std::is_same_v<T, tensor::Meta>) return MetaKind::tensor;
                else return MetaKind::token;
            }, storage_);
        }
    };
}
