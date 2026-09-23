#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifndef GGML_GEMMINI_ACT_QUANT_METRICS
#define GGML_GEMMINI_ACT_QUANT_METRICS 0
#endif
#ifndef GGML_GEMMINI_RESIDUAL_METRICS
#define GGML_GEMMINI_RESIDUAL_METRICS 0
#endif

namespace ggml::gemmini::evaluation {

struct Config {
    std::string activation_path, residual_path, run_id, workload_id;
    bool activation_reference_candidate = false;
};
struct Run {
    uint32_t original_block_id, original_k_mask;
    size_t compact_k_begin, compact_k_count;
};
struct Row {
    uint32_t original_lane_id, source_row;
};
class Invocation;
class Session : public std::enable_shared_from_this<Session> {
public:
    static std::shared_ptr<Session> start(const Config &config);
    ~Session();
    void chunk(uint64_t chunk_id);
    std::shared_ptr<Invocation> invocation(const std::string &layer, size_t m,
                                          size_t k, const float *original);
    void finish(bool success);
    void ensure_healthy() const;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
private:
    struct Impl;
    explicit Session(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class Invocation;
};
std::shared_ptr<Session> active_session();

class Invocation {
public:
    ~Invocation();
    void requantized(size_t row, size_t original_block);
    void position(size_t row, size_t column, bool selected, bool residual_nonzero);
    void finish_activation();
    void main_stripe(size_t stripe, size_t row_begin, size_t m, size_t n, size_t k);
    void compact_work(size_t stripe, size_t m, size_t n, size_t k, size_t original_k,
                      size_t tile_i, size_t tile_j, size_t tile_k,
                      const std::vector<Run> &runs, const std::vector<Row> &rows);
    Invocation(const Invocation &) = delete;
    Invocation &operator=(const Invocation &) = delete;
private:
    struct Impl;
    explicit Invocation(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class Session;
};

}
