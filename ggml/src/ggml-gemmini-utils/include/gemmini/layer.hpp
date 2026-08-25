#pragma once

#include <charconv>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <system_error>

namespace ggml::gemmini::types
{

    inline std::string resolve_matmul_layer(
        std::string_view model_arch,
        std::string_view weight_name,
        std::string_view input_name,
        std::string_view output_name)
    {
        const bool is_gpt2 = model_arch == "gpt2";
        const bool is_llama = model_arch == "llama";
        if (is_gpt2 || is_llama)
        {
            if (weight_name == "output.weight")
            {
                return "lm_head";
            }
            if (weight_name == "token_embd.weight")
            {
                bool output_is_head = output_name == "result_output";
                constexpr std::string_view prefix = "result_output-";
                if (!output_is_head && output_name.compare(0, prefix.size(), prefix) == 0)
                {
                    const std::string_view index = output_name.substr(prefix.size());
                    output_is_head = !index.empty();
                    for (const char c : index)
                    {
                        output_is_head = output_is_head && c >= '0' && c <= '9';
                    }
                }
                if (output_is_head)
                {
                    return "lm_head";
                }
            }

            constexpr std::string_view block_prefix = "blk.";
            if (weight_name.compare(0, block_prefix.size(), block_prefix) == 0)
            {
                const size_t separator = weight_name.find('.', block_prefix.size());
                if (separator != std::string_view::npos)
                {
                    const std::string_view block = weight_name.substr(
                        block_prefix.size(), separator - block_prefix.size());
                    uint64_t block_number = 0;
                    const auto parsed = std::from_chars(
                        block.data(), block.data() + block.size(), block_number);
                    if (!block.empty() && parsed.ec == std::errc{} &&
                        parsed.ptr == block.data() + block.size())
                    {
                        const std::string_view role = weight_name.substr(separator + 1);
                        std::string_view semantic;
                        if (is_gpt2)
                        {
                            if (role == "attn_qkv.weight") semantic = "attn.qkv_proj";
                            else if (role == "attn_output.weight") semantic = "attn.out_proj";
                            else if (role == "ffn_up.weight") semantic = "mlp.up_proj";
                            else if (role == "ffn_down.weight") semantic = "mlp.down_proj";
                        }
                        else
                        {
                            if (role == "attn_q.weight") semantic = "attn.q_proj";
                            else if (role == "attn_k.weight") semantic = "attn.k_proj";
                            else if (role == "attn_v.weight") semantic = "attn.v_proj";
                            else if (role == "attn_output.weight") semantic = "attn.out_proj";
                            else if (role == "ffn_up.weight") semantic = "mlp.up_proj";
                            else if (role == "ffn_gate.weight") semantic = "mlp.gate_proj";
                            else if (role == "ffn_down.weight") semantic = "mlp.down_proj";
                        }
                        if (!semantic.empty())
                        {
                            return "blk." + std::string(block) + "." + std::string(semantic);
                        }
                    }
                }
            }
        }

        const std::string_view source = !weight_name.empty() ? weight_name :
            (!output_name.empty() ? output_name : (!input_name.empty() ? input_name : "unknown"));
        std::string fallback = "unclassified.";
        fallback.reserve(fallback.size() + (source.size() < 96 ? source.size() : 96));
        for (size_t i = 0; i < source.size() && i < 96; ++i)
        {
            const unsigned char c = static_cast<unsigned char>(source[i]);
            fallback += (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-' ?
                static_cast<char>(c) : '_';
        }
        return fallback;
    }

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
