// labeling/label.h
#pragma once
#include <string.h>

static inline const char *labelFromWeight(const char *w_name)
{
    if (!w_name)
        return "others";
    if (strstr(w_name, "token_embd"))
        return "token_embd";z
    if (strstr(w_name, "position_embd"))
        return "position_embd";
    if (strstr(w_name, "attn_qkv"))
        return "attn_qkv";
    if (strstr(w_name, "attn_output"))
        return "attn_out";
    if (strstr(w_name, "ffn_up"))
        return "ffn_up";
    if (strstr(w_name, "ffn_down"))
        return "ffn_down";
    if (strstr(w_name, "output"))
        return "output";
    return "others";
}

static inline const char *labelFromCpuOp(const char *op, const char *dst_name)
{
    if (!op)
        return "others";
    if (!dst_name)
        return "";

    if (strcmp(op, "softmax") == 0)
        return "attn_softmax";
    
    if (strcmp(op, "norm") == 0 || strcmp(op, "rms_norm") == 0)
    {
        if (dst_name && strstr(dst_name, "attn_norm"))
            return "attn_norm";
        if (strstr(dst_name, "ffn_norm"))
            return "ffn_norm";
        if (strstr(dst_name, "output_norm"))
            return "output_norm";
        return "others";
    }
    if (strcmp(op, "gelu") == 0 || strcmp(op, "silu") == 0 || 
        strcmp(op, "relu") == 0 || strcmp(op, "leaky_relu") == 0)
        return "activation";
    return "others";
}
