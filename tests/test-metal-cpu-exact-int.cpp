#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"
#include "../ggml/src/ggml-metal/ggml-metal-cpu-exact-int.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

using namespace ggml::gemmini;

static bool raw_fixture(size_t k) {
    constexpr size_t rows = 3, columns = 5;
    std::vector<int32_t> a(rows*k), w(columns*k), dots(rows*columns*((k+31)/32));
    for (int repeat=0; repeat<3; ++repeat) {
        for (size_t i=0; i<a.size(); ++i) a[i] = int((i*53+repeat*17)%256)-128;
        for (size_t i=0; i<w.size(); ++i) w[i] = int((i*79+repeat*31)%256)-128;
        if (!ggml_metal_cpu_exact_int_dot(a.data(), w.data(), rows, columns, k, 32, dots.data())) return false;
        for (size_t b=0; b<(k+31)/32; ++b) for (size_t r=0; r<rows; ++r) for (size_t c=0; c<columns; ++c) {
            int64_t oracle = 0;
            for (size_t t=b*32; t<std::min(k,(b+1)*32); ++t) oracle += int64_t(a[r*k+t])*w[c*k+t];
            if (oracle != dots[(b*rows+r)*columns+c]) return false;
        }
    }
    const auto sentinel = dots;
    a[0] = 128;
    const bool passed = !ggml_metal_cpu_exact_int_dot(a.data(), w.data(), rows, columns, k, 32, dots.data()) && dots == sentinel;
    std::printf("CPU_EXACT_INT raw_tail_k=%zu invalid_code_preserves_output=%d %s\n", k, passed, passed ? "PASS" : "FAIL");
    return passed;
}

static bool production_fixture(bool hp1, size_t rows, size_t columns, size_t k, bool full, size_t row_offset, size_t tile_k=2) {
    constexpr int bits = GGML_GEMMINI_ACTIVATION_BITS;
    ggml_gemmini_args_t args{};
    args.I=rows; args.J=columns; args.K=k; args.sA=k;
    args.tile_I=2; args.tile_J=2; args.tile_K=tile_k;
    args.residual_route=residual::ResidualRoute::cpu_direct;
    args.activation_rows_per_stripe=DIM;
    args.activation_row_offset=row_offset;
    args.tiled_matmul_type=static_cast<tiled_matmul_type_t>(2);
    args.transpose_B=true;
    args.block_size_k=32; args.blocks_per_row=hp1 ? k/32 : 0; args.blocks_K=k/32;
    args.blocks_J=columns; args.blocks_I=rows;
    args.native_blocks_per_row=k/32; args.native_block_count=columns*(k/32);
    args.stride_f_out=columns*2+3; args.col_stride_f_out=2;
    std::vector<int32_t> bias(columns,7);
    args.D=bias.data(); args.repeating_bias=true;
    if (!args.A.allocate(rows,k,bits)) return false;
    for (size_t r=0;r<rows;++r) for (size_t t=0;t<k;++t) {
        if (!args.A.set(r,t,int((r*71+t*53)%(1<<bits))-(1<<(bits-1)))) return false;
    }
    if (hp1) {
        auto & meta=args.act_quant.storage().emplace<quants::act::exsia::Meta>();
        meta.theta.resize((rows+row_offset+DIM-1)/DIM);
        for (size_t i=0;i<meta.theta.size();++i) meta.theta[i]=int(i%5)-2;
        if (full) {
            residual::DirectStripeBuilder builder;
            builder.reset(0,0,rows,k,columns);
            if (!builder.add_residual(0,0,17)) return false;
            for (size_t local_k=0;local_k<32;++local_k)
                if (!builder.add_residual(1,local_k,int32_t(local_k%2 ? -47 : 39))) return false;
            if (!builder.add_residual(rows-1,k-1,-31)) return false;
            meta.direct_residuals={builder.finish()};
        }
    } else {
        auto & meta=args.act_quant.storage().emplace<quants::act::block::Meta>();
        meta.rows=rows+row_offset; meta.cols=k;
        meta.scales.resize(meta.rows*(k/32));
        for (size_t i=0;i<meta.scales.size();++i) meta.scales[i]=float((i*31)%117+1)/113.0f;
    }
    const size_t count=columns*(k/32);
    std::vector<block_q4_h0> q4(count);
    std::vector<block_q8_0> q8(count);
    std::vector<block_q4_hp1> h4(count);
    std::vector<block_q8_hp1> h8(count);
    for (size_t i=0;i<count;++i) {
        q4[i].d=q8[i].d=ggml_fp32_to_fp16(float(i%13+1)/37.0f);
        h4[i].channel_scale=h8[i].channel_scale=float((i/(k/32))%17+1)/97.0f;
        h4[i].m=h8[i].m=(i%11==0 ? INT16_MIN : int(i%7));
        for(size_t t=0;t<16;++t) q4[i].qs[t]=h4[i].qs[t]=uint8_t(i*97+t*29);
        for(size_t t=0;t<32;++t) q8[i].qs[t]=h8[i].qs[t]=int8_t((i*97+t*29)%256-128);
    }
    using F=ggml_gemmini_args_t::im2p_weight_format_t;
    if (hp1 && bits==4) { args.weight_format=F::q4_hp1; args.q4_hp1_blocks=h4.data(); args.native_weight_bytes=count*sizeof(h4[0]); }
    if (hp1 && bits==8) {
        args.weight_format=F::q8_hp1; args.q8_hp1_blocks=h8.data(); args.native_weight_bytes=count*sizeof(h8[0]);
        args.q8_hp1_block_count=count; args.q8_hp1_blocks_per_row=k/32;
    }
    if (!hp1 && bits==4) { args.weight_format=F::q4_h0; args.q4_h0_blocks=q4.data(); args.native_weight_bytes=count*sizeof(q4[0]); }
    if (!hp1 && bits==8) { args.weight_format=F::q8_h0; args.B_blocks=q8.data(); args.native_weight_bytes=count*sizeof(q8[0]); }
    std::vector<float> cpu(rows*args.stride_f_out,-777.0f), gpu=cpu;
    setenv("GGML_GEMMINI_METAL_CPU_EXACT","0",1);
    args.f_out=cpu.data();
    MatMul cpu_facade(&args);
    const auto cpu_begin=std::chrono::steady_clock::now();
    const auto reference=full ? cpu_facade.run_full() : cpu_facade.run_dense();
    const auto cpu_end=std::chrono::steady_clock::now();
    setenv("GGML_GEMMINI_METAL_CPU_EXACT","1",1);
    const uint64_t before=ggml_metal_cpu_exact_int_launches();
    args.f_out=gpu.data();
    MatMul gpu_facade(&args);
    const auto gpu_begin=std::chrono::steady_clock::now();
    const auto candidate=full ? gpu_facade.run_full() : gpu_facade.run_dense();
    const auto gpu_end=std::chrono::steady_clock::now();
    const uint64_t launch_count=ggml_metal_cpu_exact_int_launches()-before;
    const bool equal=std::memcmp(cpu.data(),gpu.data(),cpu.size()*sizeof(float))==0;
    bool residual_verified=true;
#if GGML_GEMMINI_ENABLE_RMD
    if (full && hp1) {
        std::get<quants::act::exsia::Meta>(args.act_quant.storage()).direct_residuals.clear();
        std::vector<float> dense(cpu.size(),-777.0f);
        args.f_out=dense.data();
        setenv("GGML_GEMMINI_METAL_CPU_EXACT","0",1);
        MatMul dense_facade(&args);
        residual_verified=dense_facade.run_dense().status==MatMulStatus::success &&
            std::memcmp(cpu.data(),dense.data(),cpu.size()*sizeof(float))!=0;
    }
#endif
    const bool passed=reference.status==MatMulStatus::success && candidate.status==MatMulStatus::success &&
        equal && launch_count>0 && residual_verified;
    std::printf("CPU_EXACT_INT hp1=%d bits=%d dim=%d shape=%zux%zux%zu full=%d row_offset=%zu cpu=%u gpu=%u bitwise=%d launches=%llu rmd=%d residual_verified=%d cpu_ms=%.3f gpu_ms=%.3f %s\n",
        hp1,bits,DIM,rows,columns,k,full,row_offset,unsigned(reference.status),unsigned(candidate.status),equal,
        (unsigned long long)launch_count,GGML_GEMMINI_ENABLE_RMD,residual_verified,
        std::chrono::duration<double,std::milli>(cpu_end-cpu_begin).count(),
        std::chrono::duration<double,std::milli>(gpu_end-gpu_begin).count(),passed?"PASS":"FAIL");
    return passed;
}

static bool residual_fixture(unsigned bits) {
    constexpr size_t rows=3, columns=19, k=96;
    std::vector<block_q4_hp1> q4(columns*3);
    std::vector<block_q8_hp1> q8(columns*3);
    const uint32_t offsets[]={0,3,5,5};
    const ggml_metal_residual_event events[]={{0,17},{31,-39},{64,71},{32,-57},{95,81}};
    std::vector<int64_t> expected(rows*columns), actual(rows*columns,-777);
    for (int repeat=0;repeat<3;++repeat) {
        for (size_t b=0;b<columns*3;++b) {
            q4[b].m=q8[b].m=b%7 == 0 ? INT16_MIN : int16_t(b%3);
            q4[b].channel_scale=q8[b].channel_scale=1.0f;
            for (size_t t=0;t<32;++t) {
                const int value=int((b+t+repeat)%15)-7;
                q8[b].qs[t]=int8_t(value);
                if (t<16) q4[b].qs[t]=uint8_t(value+8);
                else q4[b].qs[t-16] |= uint8_t(value+8)<<4;
            }
        }
        std::fill(expected.begin(),expected.end(),0);
        for (size_t row=0;row<rows;++row) for (size_t col=0;col<columns;++col) {
            for (size_t b=0;b<3;++b) {
                int64_t raw=0;
                for (size_t e=offsets[row];e<offsets[row+1];++e)
                    if (events[e].k/32 == b)
                        raw += int64_t(events[e].value)*(int((col*3+b+events[e].k%32+repeat)%15)-7);
                const int exponent=q8[col*3+b].m;
                if (exponent != INT16_MIN) expected[row*columns+col] += raw*(int64_t(1)<<exponent);
            }
        }
        const void * packed=bits==4 ? static_cast<const void *>(q4.data()) : static_cast<const void *>(q8.data());
        const size_t bytes=columns*3*(bits==4 ? sizeof(q4[0]) : sizeof(q8[0]));
        if (ggml_metal_cpu_exact_hp1_residual(packed,bytes,bits,rows,columns,k,offsets,events,5,actual.data()) !=
            ggml_metal_residual_status::success || actual != expected) return false;
    }
    const void * packed=bits==4 ? static_cast<const void *>(q4.data()) : static_cast<const void *>(q8.data());
    const size_t bytes=columns*3*(bits==4 ? sizeof(q4[0]) : sizeof(q8[0]));
    std::vector<int32_t> activation(rows*k);
    for (size_t i=0;i<activation.size();++i) activation[i]=int(i*71%256)-128;
    for (size_t fragment : {size_t(16),size_t(32)}) {
        std::vector<int32_t> partials(rows*columns*(k/fragment));
        if (!ggml_metal_cpu_exact_hp1_dot(activation.data(),packed,bits,rows,columns,k,fragment,partials.data())) return false;
        for (size_t block=0;block<k/fragment;++block) for (size_t row=0;row<rows;++row) for (size_t col=0;col<columns;++col) {
            int64_t sum=0;
            for (size_t offset=block*fragment;offset<(block+1)*fragment;++offset)
                sum+=int64_t(activation[row*k+offset])*(int((col*3+offset/32+offset%32+2)%15)-7);
            if (sum != partials[(block*rows+row)*columns+col]) return false;
        }
    }
    for (int exponent : {-1,63,62}) {
        q4[0].m=q8[0].m=int16_t(exponent);
        std::fill(actual.begin(),actual.end(),-777);
        const auto status=ggml_metal_cpu_exact_hp1_residual(packed,bytes,bits,rows,columns,k,offsets,events,5,actual.data());
        const auto expected_status=exponent < 0 ? ggml_metal_residual_status::invalid_input : ggml_metal_residual_status::overflow;
        if (status != expected_status || actual != std::vector<int64_t>(rows*columns,-777)) return false;
    }
    for (size_t b=0;b<2;++b) {
        std::memset(q4[b].qs,0x99,sizeof(q4[b].qs));
        std::memset(q8[b].qs,1,sizeof(q8[b].qs));
        q4[b].m=q8[b].m=62;
    }
    const uint32_t boundary_offsets[]={0,2};
    struct Boundary { int first, second; bool overflow; int64_t value; };
    const Boundary boundaries[]={{1,1,true,0},{-1,-1,false,INT64_MIN},
        {-2,1,false,-(int64_t(1)<<62)},{-2,-1,true,0},{1,-1,false,0}};
    for (const auto & boundary : boundaries) {
        const ggml_metal_residual_event input[]={{0,boundary.first},{32,boundary.second}};
        int64_t value=-777;
        const auto status=ggml_metal_cpu_exact_hp1_residual(packed,2*(bits==4 ? sizeof(q4[0]) : sizeof(q8[0])),
            bits,1,1,64,boundary_offsets,input,2,&value);
        if (status != (boundary.overflow ? ggml_metal_residual_status::overflow : ggml_metal_residual_status::success) ||
            value != (boundary.overflow ? -777 : boundary.value)) return false;
    }
    std::printf("PASS residual bits=%u original_indices cached_weight_mutation zero_carrier int64_overflow no_partial_publish\n",bits);
    return true;
}

int main(int argc, char ** argv) {
    if (argc==2 && std::strcmp(argv[1],"--benchmark")==0) {
        bool passed=raw_fixture(32);
        for (int repeat=0;repeat<2;++repeat) passed=production_fixture(false,512,768,768,false,0)&&passed;
        return passed ? 0 : 1;
    }
    bool ok=true;
    ok=residual_fixture(4)&&ok;
    ok=residual_fixture(8)&&ok;
    for (size_t k : {size_t(31),size_t(32),size_t(33),size_t(63),size_t(64),size_t(65),size_t(1025)}) ok=raw_fixture(k)&&ok;
    for (bool hp1 : {false,true}) {
        ok=production_fixture(hp1,1,5,64,false,0)&&ok;
        ok=production_fixture(hp1,3,5,96,false,1)&&ok;
        ok=production_fixture(hp1,3,5,96,false,DIM-1)&&ok;
        ok=production_fixture(hp1,35,37,128,false,0)&&ok;
        ok=production_fixture(hp1,129,1031,64,false,0)&&ok;
        ok=production_fixture(hp1,2,3,4096,false,0)&&ok;
        ok=production_fixture(hp1,3,5,64,true,0)&&ok;
        ok=production_fixture(hp1,3,37,96,true,0)&&ok;
        ok=production_fixture(hp1,3,5,96,false,0,1)&&ok;
        ok=production_fixture(hp1,3,5,96,false,0,3)&&ok;
    }
    return ok ? 0 : 1;
}
