#pragma once
#include <gemmini/semantic.hpp>
#include "ggml-impl.h"
#include <unordered_map>
#include <stdexcept>

namespace ggml::gemmini::semantic {
inline void capture_graph(const ggml_cgraph *graph) {
    const auto session = active_session();
    if (!session) return;
    const auto shape = [](const ggml_tensor *tensor) {
        std::string result = "[";
        for (int i = 0; i != GGML_MAX_DIMS; ++i) {
            if (i) result += ',';
            result += std::to_string(tensor->ne[i]);
        }
        return result + ']';
    };
    const auto tensor_payload = [&](const ggml_tensor *tensor) {
        return "\"name\":" + quote(tensor->name) + ",\"type\":" + quote(ggml_type_name(tensor->type)) +
            ",\"shape\":" + shape(tensor);
    };
    std::unordered_map<const ggml_tensor *, std::string> edges;
    for (int i = 0; i < graph->n_nodes; ++i)
        edges.emplace(graph->nodes[i], "{\"kind\":\"node\",\"ordinal\":" + std::to_string(i) + '}');
    std::string leaves = "[";
    for (int i = 0; i < graph->n_leafs; ++i) {
        edges.emplace(graph->leafs[i], "{\"kind\":\"leaf\",\"ordinal\":" + std::to_string(i) + '}');
        if (i) leaves += ',';
        leaves += '{' + tensor_payload(graph->leafs[i]) + '}';
    }
    leaves += ']';
    std::vector<Node> nodes;
    nodes.reserve(graph->n_nodes);
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto *node = graph->nodes[i];
        bool opaque = false;
        switch (node->op) {
            case GGML_OP_MAP_CUSTOM1: case GGML_OP_MAP_CUSTOM2: case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: opaque = true; break;
            default: break;
        }
        std::string parameters;
        if (!opaque) {
            constexpr char hex[] = "0123456789abcdef";
            const auto *bytes = reinterpret_cast<const unsigned char *>(node->op_params);
            for (size_t q = 0; q < sizeof(node->op_params); ++q) {
                parameters += hex[bytes[q] >> 4];
                parameters += hex[bytes[q] & 15];
            }
        }
        std::string inputs = "[";
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (j) inputs += ',';
            if (!node->src[j]) inputs += "null";
            else {
                const auto edge = edges.find(node->src[j]);
                if (edge == edges.end()) throw std::runtime_error("semantic metadata: original source edge not declared");
                inputs += edge->second;
            }
        }
        inputs += ']';
        const bool excluded = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_TRANSPOSE || ggml_is_empty(node);
        nodes.push_back({node, "{\"op\":" + quote(ggml_op_name(node->op)) + ',' + tensor_payload(node) +
            ",\"parameters_hex\":" + quote(parameters) + ",\"parameters_status\":" +
            quote(opaque ? "UNSUPPORTED_OPAQUE" : "STATIC_BYTES") + ",\"inputs\":" + inputs + '}', excluded});
    }
    session->graph(nodes, leaves);
}
}
