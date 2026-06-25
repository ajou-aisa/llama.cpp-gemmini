#pragma once

#include <string_view>
#include <cstring>
#include <cstdint>

namespace ggml::gemmini::types
{

    enum class LayerType : uint8_t
    {
        unknown = 0,
        attn_norm,
        ffn_norm,
        ffn_gelu,
        ffn_gate_par,
        kqv_out,
        result_norm,
    };

    inline const char *to_string(LayerType t)
    {
        switch (t)
        {
        case LayerType::attn_norm:
            return "attn_norm";
        case LayerType::ffn_norm:
            return "ffn_norm";
        case LayerType::ffn_gelu:
            return "ffn_gelu";
        case LayerType::ffn_gate_par:
            return "ffn_gate_par";
        case LayerType::kqv_out:
            return "kqv_out";
        case LayerType::result_norm:
            return "result_norm";
        default:
            return "unknown";
        }
    }

    inline std::string_view layer_name_view(const char *tensor_name)
    {
        if (tensor_name == nullptr)
        {
            return {};
        }

        const char *start = tensor_name;
        while (*start == '-')
        {
            ++start;
        }

        if (*start == '\0')
        {
            return {};
        }

        const char *end = std::strchr(start, '-');
        size_t len = end ? static_cast<size_t>(end - start) : std::strlen(start);
        return std::string_view(start, len);
    }

    inline LayerType parse_layer(std::string_view s)
    {
        if (s == "attn_norm")
            return LayerType::attn_norm;
        if (s == "ffn_norm")
            return LayerType::ffn_norm;
        if (s == "ffn_gelu")
            return LayerType::ffn_gelu;
        if (s == "ffn_gate_par")
            return LayerType::ffn_gate_par;
        if (s == "kqv_out")
            return LayerType::kqv_out;
        if (s == "result_norm")
            return LayerType::result_norm;
        return LayerType::unknown;
    }

    inline LayerType parse_layer(const char *tensor_name)
    {
        return parse_layer(layer_name_view(tensor_name));
    }

} // namespace ggml::gemmini::types
