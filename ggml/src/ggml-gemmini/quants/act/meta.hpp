#pragma once

#include "exsia/types.hpp"
#include "stripe/types.hpp"
#include "tensor/types.hpp"
#include "token/types.hpp"

#include <variant>

namespace ggml::gemmini::quants::act
{
    struct NoneMeta
    {
        void reset() {}
    };

    using MetaStorage = std::variant<NoneMeta, exsia::Meta, stripe::Meta, tensor::Meta, token::Meta>;

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
    };
}
