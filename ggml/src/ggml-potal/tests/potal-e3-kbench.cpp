// H3-6E3 controlled K28 experiment. Reuse the exact existing fixture, graph,
// cache, golden, route execution and output guards. This is not a new tiler.
#define main e3_original_smoke_main
#include "potal-paired-smoke.cpp"
#undef main

#include "matmul/stripe.hpp"
#include "matmul/stripe-api.hpp"
#include "support/potal-emulator.hpp"

#include <stdexcept>

namespace {
bool restrict_main_k = false;
bool capture_preparation = false;
size_t observed_k = 0;
uint64_t preparation_hash = 0;

struct fingerprint {
    uint64_t value = UINT64_C(14695981039346656037);
    void bytes(const void * data, size_t size) {
        const auto * p = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < size; ++i) {
            value ^= p[i];
            value *= UINT64_C(1099511628211);
        }
    }
    template<class T> void scalar(const T & x) { bytes(&x, sizeof(x)); }
    template<class T> void vector(const std::vector<T> & x) {
        scalar(x.size());
        bytes(x.data(), x.size() * sizeof(T));
    }
};

uint64_t prepared_fingerprint(const ggml_gemmini_args_t & args) {
    fingerprint f;
    f.scalar(args.I); f.scalar(args.J); f.scalar(args.K);
    f.scalar(args.activation_rows_per_stripe);
    f.bytes(args.A.raw_data(), args.A.raw_size());
    const auto * meta = std::get_if<ggml::gemmini::quants::act::exsia::Meta>(&args.act_quant.storage());
    if (!meta) throw std::runtime_error("E3 requires ExSIA metadata");
    f.vector(meta->theta);
    f.scalar(meta->rmd_packets.size());
    for (const auto & handle : meta->rmd_packets) {
        if (!handle) throw std::runtime_error("E3 null sealed packet");
        const auto & p = *handle;
#define FIELD(x) f.scalar(p.x)
        FIELD(version); FIELD(digit_bits); FIELD(lane_capacity); FIELD(digit_storage);
        FIELD(stripe_id); FIELD(row_begin); FIELD(row_count); FIELD(logical_k); FIELD(logical_j);
        FIELD(j_padded); FIELD(block_size); FIELD(array_dim); FIELD(activation_value_count);
        FIELD(residual_event_count); FIELD(residual_min); FIELD(residual_max); FIELD(required_planes);
        FIELD(digit_nnz); FIELD(active_original_rows); FIELD(active_original_rows_valid);
        FIELD(residual_observations_valid); FIELD(total_output_values);
#undef FIELD
        f.vector(p.k_indices);
        f.vector(p.stacked_activation.signed_int8);
        f.vector(p.stacked_activation.signed_int16);
        f.scalar(p.blocks.size());
        for (const auto & b : p.blocks) {
#define FIELD(x) f.scalar(b.x)
            FIELD(block_id); FIELD(global_k_begin); FIELD(compact_k_count); FIELD(padded_k_count);
            FIELD(active_lane_mask); FIELD(active_lane_count); FIELD(lane_ids); FIELD(lane_k_masks);
            FIELD(k_index_offset); FIELD(activation_offset); FIELD(activation_byte_offset);
            FIELD(activation_byte_count); FIELD(output_value_offset); FIELD(rows_padded);
            FIELD(lane_stride_values);
#undef FIELD
            f.scalar(b.groups.size());
            for (const auto & g : b.groups) {
                f.vector(g.lane_positions); f.scalar(g.row_offsets); f.vector(g.row_ids);
                f.scalar(g.k_mask); f.scalar(g.padded_k_count); f.scalar(g.activation_offset);
                f.scalar(g.activation_byte_offset); f.scalar(g.activation_byte_count);
            }
        }
    }
    return f.value;
}

const char * mode_name(int mode) {
    return mode == 0 ? "serial_original" : mode == 1 ? "serial_restricted" : "paired_original";
}

std::vector<uint64_t> signature(const route_run & r) {
    std::vector<uint64_t> result;
#define COUNT(x) result.push_back(r.after.x - r.before.x)
    COUNT(main_loop_ws); COUNT(residual_loop_ws); COUNT(main_fences); COUNT(residual_fences);
    COUNT(paired_main_loop_ws); COUNT(paired_residual_loop_ws);
    COUNT(paired_index_configs); COUNT(paired_index_writes);
    COUNT(paired_readbacks); COUNT(paired_readback_tiles);
    COUNT(paired_fences); COUNT(paired_drains); COUNT(scale_writes); COUNT(scale_releases);
    COUNT(paired_fallback_stripes); COUNT(stripes_semantic_completed);
#undef COUNT
    return result;
}

void print_run(const bench_case & c, int mode, int iteration, const route_run & r) {
    std::printf("E3_SAMPLE case=%s mode=%s iteration=%d frozen_requested_K=%zu compute_ms=%.6f "
                "pipeline_ms=%.6f scale_api_ms=%.6f issue_api_ms=%.6f fence_api_ms=%.6f "
                "act_quant_ms=%.6f ok=%d",
                c.name, mode_name(mode), iteration, observed_k, r.compute_ms, r.wall_ms,
                1e3*(r.after.scale_s-r.before.scale_s), 1e3*(r.after.issue_s-r.before.issue_s),
                1e3*(r.after.fence_s-r.before.fence_s), 1e3*(r.after.act_quant_s-r.before.act_quant_s), r.ok);
#define COUNT(x) std::printf(" " #x "=%llu", (unsigned long long)(r.after.x-r.before.x))
    COUNT(main_loop_ws); COUNT(residual_loop_ws); COUNT(main_fences); COUNT(residual_fences);
    COUNT(paired_main_loop_ws); COUNT(paired_residual_loop_ws);
    COUNT(paired_index_configs); COUNT(paired_index_writes);
    COUNT(paired_readbacks); COUNT(paired_readback_tiles); COUNT(paired_fences); COUNT(paired_drains);
    COUNT(scale_writes); COUNT(scale_releases); COUNT(paired_fallback_stripes);
    COUNT(stripes_semantic_completed); COUNT(weight_cache_hits); COUNT(weight_cache_misses);
    COUNT(weight_packs);
#undef COUNT
    std::printf("\n");
}

bool verify_frozen(const bench_case & c) {
    std::mt19937 rng(20261023);
    const ggml_init_params ip{size_t(4*(c.n*c.k+2*c.m*c.n+c.m*c.k)+64*1024*1024),nullptr,false};
    ggml_context_ptr context(ggml_init(ip));
    if (!context) return false;
    auto * w = hp1_weight_tensor(context.get(),c.k,c.n,rng);
    auto * x = activations(context.get(),c.k,c.m,rng,c.outlier,c.blocks,c.step);
    auto * y = ggml_new_tensor_2d(context.get(),GGML_TYPE_F32,c.n,c.m);
    const hp1_shape shape{c.m,c.n,c.k};
    potal_tile_plan original, restricted;
    if (potal_tile_select(c.m,c.n,c.k,2,&original) != POTAL_OK ||
        potal_frozen_plan(c.m,c.n,c.k,original.tile_I,original.tile_J,
                          std::min(size_t(28),original.tile_K),2,&restricted) != POTAL_OK) return false;
    uint64_t hashes[2]{};
    std::vector<int32_t> outputs[2];
    uint64_t loops[2]{}, scales[2]{};
    for (int mode = 0; mode < 2; ++mode) {
        const auto & plan = mode ? restricted : original;
        ggml_gemmini_args_t args;
        hp1_operands o;
        hp1::stats stats;
        if (!o.allocate(shape) ||
            !init_args(w,y,shape,"","e3_verify",ggml::gemmini::residual::ResidualRoute::ws_packet,plan,args) ||
            unpack_weights(args,o,&stats) != hp1::status::ok || !quantize_activations(x,args,o)) return false;
        hashes[mode] = prepared_fingerprint(args);
        std::vector<int32_t> independent(size_t(c.m*c.n));
        reference_gemm_i32(o.a,o.lda,o.b,o.ldb,o.carriers,o.cstride,32,c.m,c.n,c.k,independent.data(),c.n);
        potal_test::emulator emulator;
        auto ops = emulator.ops();
        potal_stripe_session session{{&ops,0,{}}};
        potal_stripe_gemm_desc d{};
        d.m=c.m; d.n=c.n; d.k=c.k; d.a=o.a; d.b=o.b; d.c=o.c;
        d.lda=o.lda; d.ldb=o.ldb; d.ldc=o.ldc; d.carriers=o.carriers;
        d.carrier_stride=o.cstride; d.scale_block_k=32;
        d.tile_i=plan.tile_I; d.tile_j=plan.tile_J; d.tile_k=plan.tile_K;
        const auto run=run_planned_stripe_gemm(session,d);
        if (!run.ok || emulator.scale_miss || emulator.in_flight || !emulator.scales.empty() ||
            emulator.writes != emulator.releases) return false;
        outputs[mode].resize(size_t(c.m*c.n));
        for (size_t i=0;i<size_t(c.m);++i)
            std::copy_n(o.c+i*o.ldc,size_t(c.n),outputs[mode].data()+i*c.n);
        if (outputs[mode] != independent) return false;
        loops[mode]=emulator.loops; scales[mode]=emulator.writes;
        if (run.report.loop_ws != plan.I0*plan.J0*plan.K0) return false;
    }
    const bool exact=outputs[0]==outputs[1] && hashes[0]==hashes[1] && scales[0]==scales[1] &&
                     original.tile_I==restricted.tile_I && original.tile_J==restricted.tile_J;
    std::printf("E3_FROZEN_VERIFY case=%s software_only=1 original_K=%zu restricted_K=%zu I=%zu J=%zu "
                "packet_and_quant_hash=%016llx quant_packet_equal=%d original_loops=%llu "
                "restricted_loops=%llu scale_writes=%llu exact=%d\n",
                c.name,original.tile_K,restricted.tile_K,original.tile_I,original.tile_J,
                (unsigned long long)hashes[0],hashes[0]==hashes[1],
                (unsigned long long)loops[0],(unsigned long long)loops[1],
                (unsigned long long)scales[0],exact);
    return exact;
}

void bench_control(ggml_backend_t backend,const bench_case & c,int repeats) {
    std::mt19937 rng(20261023);
    ggml_init_params ip{size_t(4*(c.n*c.k+2*c.m*c.n+c.m*c.k)+64*1024*1024),nullptr,false};
    ggml_context_ptr context(ggml_init(ip));
    if (!context) throw std::runtime_error("context");
    auto * w=hp1_weight_tensor(context.get(),c.k,c.n,rng);
    auto * x=activations(context.get(),c.k,c.m,rng,c.outlier,c.blocks,c.step);
    auto * expected=ggml_new_tensor_2d(context.get(),GGML_TYPE_F32,c.n,c.m);
    const auto gold=golden(w,x,expected);
    if (!gold.ok) throw std::runtime_error("golden");
    fingerprint weights,activations_before;
    weights.bytes(w->data,ggml_nbytes(w));
    activations_before.bytes(x->data,ggml_nbytes(x));
    ggml_init_params wip{2*ggml_tensor_overhead(),nullptr,true};
    ggml_context_ptr weight_context(ggml_init(wip));
    auto * cached=ggml_new_tensor_2d(weight_context.get(),GGML_TYPE_Q8_HP1,c.k,c.n);
    auto * buft=ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft,ggml_backend_buft_get_alloc_size(buft,cached)));
    if (!buffer || ggml_backend_tensor_alloc(buffer.get(),cached,ggml_backend_buffer_get_base(buffer.get()))!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("weight_buffer");
    ggml_backend_buffer_set_usage(buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(cached,w->data,0,ggml_nbytes(w));
    auto * y=ggml_mul_mat(context.get(),cached,x);
    auto * graph=ggml_new_graph(context.get());
    ggml_build_forward_expand(graph,y);
    uint64_t baseline_hash=0;
    std::array<std::vector<uint64_t>,3> counts;
    std::array<std::vector<double>,3> samples;
    // Verification/warmup is outside every performance sample, not full inference.
    capture_preparation=true;
    for (int mode=0;mode<3;++mode) {
        restrict_main_k=mode==1;
        const auto r=run_route(backend,c.name,mode==2?ROUTE_PAIRED:ROUTE_SERIAL,graph,y,expected,gold,false);
        if (!r.ok) throw std::runtime_error("warmup result");
        counts[mode]=signature(r);
        if (mode==0) baseline_hash=preparation_hash;
        if (preparation_hash!=baseline_hash) throw std::runtime_error("quantization or packet mismatch");
    }
    capture_preparation=false;
    std::printf("E3_FIXTURE case=%s seed=20261023 weights_hash=%016llx activations_hash=%016llx "
                "quant_packet_hash=%016llx diagnostic_preparation=0 timed_weight_packs=0\n",c.name,
                (unsigned long long)weights.value,(unsigned long long)activations_before.value,
                (unsigned long long)baseline_hash);
    for (int i=0;i<repeats;++i) {
        for (int j=0;j<3;++j) {
            const int mode=(i+(i%2?2-j:j))%3;
            restrict_main_k=mode==1;
            const auto r=run_route(backend,c.name,mode==2?ROUTE_PAIRED:ROUTE_SERIAL,graph,y,expected,gold,false);
            if (!r.ok || signature(r)!=counts[mode] || r.after.weight_cache_hits-r.before.weight_cache_hits!=1 ||
                r.after.weight_packs!=r.before.weight_packs) throw std::runtime_error("sample correctness/count/cache guard");
            samples[mode].push_back(r.compute_ms);
            print_run(c,mode,i+1,r);
        }
    }
    for (int mode=0;mode<3;++mode) {
        const double med=median(samples[mode]);
        std::vector<double> deviations;
        for (double value:samples[mode]) deviations.push_back(std::abs(value-med));
        std::printf("E3_SUMMARY case=%s mode=%s repeats=%d median_ms=%.6f mad_ms=%.6f\n",
                    c.name,mode_name(mode),repeats,med,median(deviations));
    }
    std::printf("E3_DIFFERENCE case=%s serial_K_split_ms=%.6f paired_same_K_extra_ms=%.6f "
                "B_is_not_pure_issue_overhead=1\n",c.name,
                median(samples[1])-median(samples[0]),median(samples[2])-median(samples[1]));
    fingerprint after_w,after_x;
    after_w.bytes(cached->data,ggml_nbytes(cached)); after_x.bytes(x->data,ggml_nbytes(x));
    if (after_w.value!=weights.value || after_x.value!=activations_before.value) throw std::runtime_error("input mutation");
    ggml_backend_potal_clear_weight_cache(backend);
}
}

namespace ggml::potal {
potal_status e3_tile_select(size_t m,size_t n,size_t k,uint32_t ppb,potal_tile_plan * plan) {
    const auto status=potal_tile_select(m,n,k,ppb,plan);
    if (status!=POTAL_OK) return status;
    if (restrict_main_k) {
        potal_tile_plan frozen;
        const auto checked=potal_frozen_plan(m,n,k,plan->tile_I,plan->tile_J,
                                             std::min(size_t(28),plan->tile_K),ppb,&frozen);
        if (checked!=POTAL_OK) return checked;
        *plan=frozen;
    }
    observed_k=plan->tile_K;
    return POTAL_OK;
}

pipeline_report e3_run_pipeline(const ggml_tensor * x,ggml_gemmini_args_t & args,const hp1_weight & w,
                                const potal_tile_plan & plan,const pipeline_config & cfg,stripe_owner & owner) {
    auto result=run_stripe_pipeline(x,args,w,plan,cfg,owner);
    if (capture_preparation && result.ok) preparation_hash=prepared_fingerprint(args);
    return result;
}
}

int main(int argc,char ** argv) {
    try {
        const bool verify=argc==2 && std::strcmp(argv[1],"--verify")==0;
        if (verify) {
            for (const auto & c:bench_cases)
                if ((std::strcmp(c.name,"decode_1x768x768")==0 ||
                     std::strcmp(c.name,"gpt2_ffn_up_16x3072x768")==0 ||
                     std::strcmp(c.name,"dense_fallback_160x320x512")==0) && !verify_frozen(c)) return 1;
            std::puts("E3_KBENCH SOFTWARE_ONLY PASS");
            return 0;
        }
        if (argc!=5 || std::strcmp(argv[1],"--case") || std::strcmp(argv[3],"--repeat")) {
            std::fprintf(stderr,"usage: %s --verify | --case NAME --repeat N\n",argv[0]); return 2;
        }
        char * end=nullptr;
        const long repeats=std::strtol(argv[4],&end,10);
        if (end==argv[4] || *end || repeats<7 || repeats>100) return 2;
        for (const char * key:{"POTAL_PLANNING_PROFILE","POTAL_RAW_TRACE","POTAL_PAIRED_WINDOW_DIAG",
                              "POTAL_PAIRED_DIAG_INDEX_ENTRIES","POTAL_PAIRED_STAGE_DIAG","POTAL_PAIRED_PROBE"})
            if (std::getenv(key)) throw std::runtime_error(std::string("diagnostic setting: ")+key);
        if (!select_pipeline_mode()) return 2;
        ggml_backend_ptr backend(ggml_backend_potal_init());
        if (!backend) return 1;
        for (const auto & c:bench_cases) if (!std::strcmp(c.name,argv[2])) {
            bench_control(backend.get(),c,int(repeats));
            std::printf("E3_KBENCH %s case=%s failures=%d\n",failures?"FAIL":"PASS",c.name,failures);
            return failures?1:0;
        }
        return 2;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"E3_KBENCH FAIL %s\n",error.what()); return 1;
    }
}
