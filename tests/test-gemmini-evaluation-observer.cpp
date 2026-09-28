#include "../ggml/src/ggml-gemmini/ggml-gemmini-evaluation-observer.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-run-aware.hpp"
#include "../common/json.hpp"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace ggml::gemmini;

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    constexpr size_t rows = DIM + 1, columns = 64, outputs = 17;
    std::vector<float> source(rows * columns, 0.5f);
    for (size_t row = 0; row + 1 < rows; ++row) {
        source[row * columns] = 256.0f;
        source[row * columns + 1] = 8.0f;
        source[row * columns + 40] = -17.0f;
    }
    std::vector<block_q4_hp1> q4(outputs * 2);
    std::vector<block_q8_hp1> q8(outputs * 2);
    for (size_t index = 0; index < q8.size(); ++index) {
        q4[index].channel_scale = q8[index].channel_scale = 0.5f;
        q4[index].m = q8[index].m = static_cast<int16_t>(index % 3);
        std::fill(std::begin(q4[index].qs), std::end(q4[index].qs), uint8_t{0x99});
        std::fill(std::begin(q8[index].qs), std::end(q8[index].qs), int8_t{1});
    }
    ggml_gemmini_args_t baseline{}, observed{};
    for (auto *args : {&baseline, &observed}) {
        args->I = rows; args->J = outputs; args->K = columns; args->sA = columns;
        args->tile_I = 1; args->tile_J = 2; args->tile_K = 2;
        args->activation_rows_per_stripe = DIM;
        args->matmul_layer = "synthetic.hp1.producer";
        args->residual_route = residual::ResidualRoute::ws_packet;
        args->block_size_k = 32;
        args->native_block_count = outputs * 2;
        args->native_blocks_per_row = 2;
        if (GGML_GEMMINI_ACTIVATION_BITS == 4) {
            args->weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
            args->q4_hp1_blocks = q4.data();
            args->native_weight_bytes = q4.size() * sizeof(block_q4_hp1);
        } else {
            args->weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
            args->q8_hp1_blocks = q8.data();
            args->q8_hp1_block_count = q8.size();
            args->q8_hp1_blocks_per_row = 2;
            args->native_weight_bytes = q8.size() * sizeof(block_q8_hp1);
        }
        assert(args->A.allocate(rows, columns, GGML_GEMMINI_ACTIVATION_BITS));
    }
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();
    quants::act::exsia::ExSIA first, second;
    quants::act::exsia::Meta first_meta, second_meta;
    assert(first.run(first_meta, &tensor, baseline));
    evaluation::Config config;
    config.run_id = "synthetic-hp1-producer";
    config.workload_id = "small-tensor-shape-and-carrier-observer";
    const char *hash = std::getenv("EVALUATION_MANIFEST_SHA256");
    config.manifest_sha256 = hash ? hash : std::string(64, 'a');
#if GGML_GEMMINI_ACT_QUANT_METRICS
    config.activation_path = (directory / "activation-quant-metrics.jsonl").string();
#endif
#if GGML_GEMMINI_RESIDUAL_METRICS
    config.residual_path = (directory / "residual-path-metrics.jsonl").string();
#endif
#if GGML_GEMMINI_SCALE_METRICS
    config.scale_path = (directory / "scale-alignment-metrics.jsonl").string();
#endif
    const auto session = evaluation::Session::start(config);
    if (!session) return 0;
    session->chunk(0);
    assert(second.run(second_meta, &tensor, observed));
    assert(*baseline.A.bytes == *observed.A.bytes);
    assert(first_meta.theta == second_meta.theta);
    assert(first.state().residual == second.state().residual);
    assert(first_meta.rmd_packets.size() == second_meta.rmd_packets.size());
    for (const auto &packet : second_meta.rmd_packets) {
        rmd::RunAwareRequest request;
        assert(rmd::build_run_aware_request(observed, *packet, request) == rmd::RmdStatus::success);
        std::vector<evaluation::Run> runs;
        std::vector<evaluation::Row> compact_rows;
        size_t fragments = 0;
        for (const auto &run : request.runs) {
            runs.push_back({run.original_block_id, run.union_k_mask, run.compact_k_begin, run.compact_k_count});
            fragments += (run.compact_k_count + DIM - 1) / DIM;
#if GGML_GEMMINI_SCALE_METRICS
            const auto plan = quants::wroute::resolve_weight_route_plan(
                observed, quants::wroute::WeightScaleInfoMode::ResidualHp1Scu);
            evaluation::observe_scu_block(observed, plan, packet->stripe_id, "RESIDUAL",
                run.original_block_id, request.m, (run.compact_k_count + DIM - 1) / DIM);
#endif
        }
        for (const auto &row : request.rows) compact_rows.push_back({row.original_lane_id, row.source_row});
        fragments *= ((request.m + DIM - 1) / DIM) * ((request.n + DIM - 1) / DIM);
#if GGML_GEMMINI_RESIDUAL_METRICS
        observed.evaluation_context->compact_work(packet->stripe_id, request.m, request.n, request.k,
            request.original_k, request.tile_i, request.tile_j, request.tile_k, runs, compact_rows,
            packet->required_planes, fragments);
#endif
    }
    session->finish(true);
#if GGML_GEMMINI_SCALE_METRICS
    std::ifstream scales(config.scale_path);
    std::string line;
    size_t dense_count = 0, residual_count = 0;
    while (std::getline(scales, line)) {
        const auto record = nlohmann::json::parse(line);
        if (record.at("kind") != "SCALE_ALIGNMENT") continue;
        const size_t column = record.at("column"), block = record.at("original_block");
        assert(record.at("scu_shift_offset") == (column * 2 + block) % 3);
        if (record.at("work_type") == "DENSE") ++dense_count;
        else ++residual_count;
    }
    assert(dense_count == outputs * 2 * 2 && residual_count > 0);
#endif
    std::cout << "EVALUATION_PRODUCER_OBSERVER numerical=PASS actual_hp1_carriers=PASS compact_runs=PASS\n";
}
