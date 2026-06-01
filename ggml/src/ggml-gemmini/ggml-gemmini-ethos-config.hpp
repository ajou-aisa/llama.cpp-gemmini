#pragma once

#include <map>
#include <optional>
#include <string>

#include <gemmini/layer.hpp>

struct ggml_gemmini_ethos_patch
{
    std::optional<int> q;
    std::optional<int> delta;
    std::optional<bool> l2_enabled;
    std::optional<int> l2_c;
    std::optional<int> l2_d;
};

struct ggml_gemmini_ethos_arch_config
{
    ggml_gemmini_ethos_patch defaults;
    std::map<ggml::gemmini::types::LayerType, ggml_gemmini_ethos_patch> layers;
};

struct ggml_gemmini_ethos_config_registry
{
    bool available = false;
    ggml_gemmini_ethos_patch defaults;
    std::map<std::string, ggml_gemmini_ethos_arch_config> architectures;
};

struct ggml_gemmini_resolved_ethos_override
{
    int q = 0;
    int delta = 0;
    bool l2_enabled = false;
    int l2_c = 1;
    int l2_d = 1;
};

ggml_gemmini_ethos_config_registry ggml_gemmini_load_ethos_config_registry();

std::optional<ggml_gemmini_resolved_ethos_override> ggml_gemmini_resolve_ethos_override(
    const ggml_gemmini_ethos_config_registry &registry,
    const std::string &model_arch,
    ggml::gemmini::types::LayerType layer_type);
