#include "ggml-gemmini-args.h"
#include "quants/act/exsia/exsia.hpp"
#include "ggml-gemmini.h"
#include "ggml-metal-cpu-exact-int.h"
#include "ggml-quants.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void selection_fixture(size_t k, bool zero) {
    using namespace ggml::gemmini;
    constexpr size_t rows=DIM+1;
    std::vector<float> input(rows*k,0.0f);
    if (!zero) for (size_t r=0;r<rows;++r) for (size_t c=0;c<k;++c)
        input[r*k+c]=c==0 ? 256.0f : c==33 ? -17.0f : float(int(c%9)-4)*0.45f;
    ggml_tensor source{}; source.type=GGML_TYPE_F32; source.data=input.data();
    ggml_gemmini_args_t args{};
    args.I=rows; args.J=19; args.K=k; args.sA=k;
    args.tile_I=1; args.tile_J=1; args.tile_K=2; args.activation_rows_per_stripe=DIM;
    args.residual_route=residual::ResidualRoute::cpu_direct;
    require(args.A.allocate(rows,k,GGML_GEMMINI_ACTIVATION_BITS),"activation allocation");
    quants::act::exsia::ExSIA producer;
    quants::act::exsia::Meta meta;
    require(producer.run(meta,&source,args),"ExSIA producer failed");
#if !GGML_GEMMINI_EXSIA_OUTLIER_SELECTION
    const int rho=GGML_GEMMINI_ACTIVATION_BITS-2;
    for (size_t r=0;r<rows;++r) {
        const int theta=zero ? -rho : 8-rho;
        require(meta.theta[r/DIM]==theta,"selection OFF excluded a maximum exponent");
        for (size_t begin=0;begin<k;begin+=32) {
            int e=-32768;
            for (size_t c=begin;c<std::min(begin+32,k);++c)
                if (input[r*k+c]!=0) e=std::max(e,std::ilogb(std::fabs(input[r*k+c])));
            for (size_t c=begin;c<std::min(begin+32,k);++c) {
                const long local=e==-32768 ? 0 : std::lrint(std::ldexp(double(input[r*k+c]),rho-e));
                const double folded=e==-32768 ? 0.0 : std::ldexp(double(local),e-rho-theta);
                const int expected=std::clamp(int(std::round(folded)),-(1<<(GGML_GEMMINI_ACTIVATION_BITS-1)),(1<<(GGML_GEMMINI_ACTIVATION_BITS-1))-1);
                require(args.A.get(r,c)==expected,"selection OFF folding differs from independent oracle");
            }
        }
    }
    for (const auto & stripe:producer.state().stripe)
        for (uint64_t word:stripe.outlier_mask.words) require(word==0,"selection OFF produced a mask");
    for (int32_t value:producer.state().residual) require(value==0,"selection OFF produced a residual");
    require(meta.direct_residuals.empty() && meta.rmd_packets.empty(),"selection OFF produced a packet");
#else
    if (!zero) require(meta.theta.front()<8-(GGML_GEMMINI_ACTIVATION_BITS-2),"selection ON failed to exclude fixture outliers");
#if GGML_GEMMINI_ENABLE_RMD
    if (!zero) require(!meta.direct_residuals.empty(),"PoTal compensation missing");
#endif
#endif
    std::printf("PASS ExSIA bits=%d DIM=%d selection=%d RC=%d K=%zu zero=%d\n",GGML_GEMMINI_ACTIVATION_BITS,DIM,GGML_GEMMINI_EXSIA_OUTLIER_SELECTION,GGML_GEMMINI_ENABLE_RMD,k,zero);
}

static void q6_fixture(size_t rows) {
    constexpr size_t k=512,n=19;
    std::vector<block_q6_K> packed(n*k/256);
    for (size_t b=0;b<packed.size();++b) {
        auto & w=packed[b]; w.d=ggml_fp32_to_fp16(0.03125f);
        for (size_t i=0;i<128;++i) w.ql[i]=uint8_t(b*29+i*17);
        for (size_t i=0;i<64;++i) w.qh[i]=uint8_t(b*13+i*19);
        for (size_t i=0;i<16;++i) w.scales[i]=int8_t(int((i+b)%11)-5);
    }
    std::vector<int32_t> a(rows*k),raw(rows*n*k/16);
    for (size_t i=0;i<a.size();++i) a[i]=int(i%255)-128;
    require(ggml_metal_cpu_exact_q6_dot(a.data(),packed.data(),rows,n,k,raw.data()),"Q6 raw GPU dot failed");
    std::vector<float> decoded(n*k);
    dequantize_row_q6_K(packed.data(),decoded.data(),n*k);
    for (size_t b=0;b<k/16;++b) for (size_t r=0;r<rows;++r) for (size_t c=0;c<n;++c) {
        const auto & w=packed[c*k/256+b/16];
        const double scale=double(ggml_fp16_to_fp32(w.d))*w.scales[b%16];
        double expected=0;
        for (size_t i=0;i<16;++i) expected+=double(a[r*k+b*16+i])*decoded[c*k+b*16+i];
        require(double(raw[(b*rows+r)*n+c])*scale==expected,"Q6 unpack or scale boundary mismatch");
    }
    require(!ggml_metal_cpu_exact_q6_dot(a.data(),packed.data(),rows,n,k-1,raw.data()),"Q6 invalid K accepted");
    std::printf("PASS Q6 packed dot rows=%zu K=%zu\n",rows,k);
}

static void q6_graph(bool benchmark=false) {
    using namespace ggml::gemmini;
    if (config::CURRENT_COMPUTE_TYPE!=config::ComputeType::INT || config::CURRENT_ACTIVATION_QUANT!=config::ActivationQuantAlgo::BLOCK || GGML_GEMMINI_ENABLE_RMD) return;
    const size_t k=benchmark ? 2048 : 256,n=benchmark ? 512 : 19,m=benchmark ? 128 : 5;
    setenv("GGML_GEMMINI_METAL_CPU_EXACT_Q6_HEAD","1",1);
    ggml_backend_t backend=ggml_backend_gemmini_init();
    ggml_context * ctx=ggml_init({1<<20,nullptr,true});
    require(backend && ctx,"Q6 graph allocation");
    auto * w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q6_K,k,n);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);
    ggml_set_name(w,"token_embd.weight");
    auto * y=ggml_mul_mat(ctx,w,x); ggml_set_name(y,"result_output");
    auto * graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,y);
    require(ggml_backend_supports_op(backend,y),"Q6 RTN head not admitted");
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend); require(buffer,"Q6 graph buffer");
    std::vector<float> original(n*k),activation(m*k);
    for(size_t i=0;i<original.size();++i) original[i]=float(int(i%137)-68)/37;
    for(size_t i=0;i<activation.size();++i) activation[i]=float(int(i%131)-65)/29;
    std::vector<block_q6_K> packed(n*k/256);
    quantize_row_q6_K_ref(original.data(),packed.data(),n*k);
    ggml_backend_tensor_set(w,packed.data(),0,ggml_nbytes(w));
    ggml_backend_tensor_set(x,activation.data(),0,ggml_nbytes(x));
    std::vector<float> cpu(m*n),gpu(m*n);
    double elapsed[2]{};
    for(int enabled=0;enabled<2;++enabled) {
        setenv("GGML_GEMMINI_METAL_CPU_EXACT",enabled ? "1":"0",1);
        const auto start=std::chrono::steady_clock::now();
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Q6 RTN graph failed");
        elapsed[enabled]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        ggml_backend_tensor_get(y,enabled ? gpu.data():cpu.data(),0,ggml_nbytes(y));
    }
    require(std::memcmp(cpu.data(),gpu.data(),cpu.size()*sizeof(float))==0,"Q6 RTN CPU/Metal mismatch");
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
    std::printf("PASS Q6 RTN activation graph bits=%d shape=%zux%zux%zu bitwise=1 cpu_ms=%.3f gpu_ms=%.3f\n",GGML_GEMMINI_ACTIVATION_BITS,m,n,k,elapsed[0],elapsed[1]);
}

int main(int argc, char ** argv) {
    try {
        if (argc==2 && std::strcmp(argv[1],"--benchmark")==0) {
            for (int repeat=0;repeat<3;++repeat) q6_graph(true);
            return 0;
        }
        setenv("OMP_NUM_THREADS","4",1);
        setenv("GGML_GEMMINI_METAL_CPU_EXACT","1",1);
        for(size_t k:{size_t(31),size_t(32),size_t(33),size_t(65)}) { selection_fixture(k,false); selection_fixture(k,true); }
        q6_fixture(1); q6_fixture(7); q6_graph();
        return 0;
    } catch(const std::exception & e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
