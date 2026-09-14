// Inspect stored tensors with the repository's own GGUF reader. This is not
// model inference, quantization, backend assignment, or FPGA qualification.
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

static std::string quoted(const std::string & input) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char ch : input) {
        if (ch == '"' || ch == '\\') {
            result += '\\';
            result += static_cast<char>(ch);
        } else if (ch < 32) {
            result += "\\u00";
            result += hex[ch >> 4];
            result += hex[ch & 15];
        } else {
            result += static_cast<char>(ch);
        }
    }
    return result + '"';
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: llama-im2p-model-inventory MODEL.gguf\n"
                  << "Read-only metadata JSON; no backend or model execution.\n";
        return 0;
    }
    if (argc != 2) {
        std::cerr << "Expected one GGUF filename; use --help.\n";
        return 2;
    }
    try {
        const std::filesystem::path path(argv[1]);
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("input must be a regular GGUF file");
        }
        const std::uintmax_t file_bytes = std::filesystem::file_size(path);
        ggml_context * raw_tensors = nullptr;
        const gguf_init_params params = {true, &raw_tensors};
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(argv[1], params), gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(raw_tensors, ggml_free);
        if (!metadata || !tensors) {
            throw std::runtime_error("GGUF metadata parser failed");
        }
        const auto * ctx = metadata.get();
        const auto count = gguf_get_n_tensors(ctx);
        const auto base = gguf_get_data_offset(ctx);
        if (base > file_bytes) {
            throw std::runtime_error("GGUF data offset is outside the file");
        }
        std::ostringstream out;
        out << "{\"schema\":1,\"scope\":\"metadata_only_no_invocation\","
            << "\"tensor_data_allocated\":false,\"file\":" << quoted(path.string())
            << ",\"file_size\":" << file_bytes
            << ",\"version\":" << gguf_get_version(ctx)
            << ",\"architecture\":";
        const auto arch = gguf_find_key(ctx, "general.architecture");
        if (arch >= 0 && gguf_get_kv_type(ctx, arch) == GGUF_TYPE_STRING) {
            out << quoted(gguf_get_val_str(ctx, arch));
        } else {
            out << "null";
        }
        out << ",\"tensor_count\":" << count << ",\"tensors\":[";
        for (std::int64_t i = 0; i < count; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            const ggml_tensor * tensor = ggml_get_tensor(tensors.get(), name);
            if (!tensor || tensor->data != nullptr) {
                throw std::runtime_error("metadata-only tensor contract violated");
            }
            const auto offset = gguf_get_tensor_offset(ctx, i);
            const auto bytes = gguf_get_tensor_size(ctx, i);
            if (offset > file_bytes - base || bytes > file_bytes - base - offset) {
                throw std::runtime_error("tensor data extent exceeds file size");
            }
            if (i) { out << ','; }
            out << "{\"name\":" << quoted(name)
                << ",\"type_id\":" << static_cast<int>(gguf_get_tensor_type(ctx, i))
                << ",\"type\":" << quoted(ggml_type_name(tensor->type))
                << ",\"ggml_shape\":[";
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (d) { out << ','; }
                out << tensor->ne[d];
            }
            out << "],\"bytes\":" << bytes << ",\"offset\":" << offset << '}';
        }
        out << "]}\n";
        // No partial JSON is published after a parse/extent error.
        std::cout << out.str();
        return std::cout ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "model inventory failed: " << error.what() << '\n';
        return 1;
    }
}
