#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "matmul/dense.hpp"
#include "matmul/options.hpp"
#include "matmul/stripe.hpp"
#include "quants/act/exsia/wide-main.hpp"
#include <support/stripe-owner.hpp>
#include <support/paired-h2-emulator.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <pthread.h>
#include <sched.h>

using namespace ggml::potal;
constexpr int bits = GGML_GEMMINI_ACTIVATION_BITS;
constexpr ggml_type weight_type = bits == 4 ? GGML_TYPE_MXINT4 : GGML_TYPE_Q8_HP1;
constexpr size_t K = 96, N = 17;

static void require(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(what);
}

template<int B> static void split_boundaries() {
    constexpr int lo = -(1 << (2*B-1)), hi = ((1 << (B-1))-1)*((1<<B)+1);
    for (int u=lo-1; u<=hi+1; ++u) {
        int8_t a=0,b=0;
        const bool ok = ggml::gemmini::quants::act::exsia::split_wide_main<B>(u,a,b);
        require(ok == (u>=lo && u<=hi), "two-digit boundary acceptance");
        if (ok) require(int(a)+(1<<B)*int(b)==u && a>=-(1<<(B-1)) && a<(1<<(B-1)) &&
                        b>=-(1<<(B-1)) && b<(1<<(B-1)), "exact signed digits");
    }
}

static void run_case(size_t m, size_t capacity, bool outliers, bool overflow, bool fail_high) {
    ggml_context_ptr ctx(ggml_init({8*1024*1024,nullptr,false}));
    require(bool(ctx), "context allocation");
    auto * w=ggml_new_tensor_2d(ctx.get(),weight_type,K,N);
    auto * x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,K,m);
    auto * y=ggml_mul_mat(ctx.get(),w,x);
    std::vector<float> wf(K*N,1.0f);
    ggml_quantize_chunk(weight_type,wf.data(),w->data,0,N,K,nullptr);
    auto * values=static_cast<float *>(x->data);
    std::vector<float> expected(m,0);
    for (size_t row=0;row<m;++row) for(size_t k=0;k<K;++k) {
        float v = k<32 ? 7.0f : k<64 ? 3.5f : 1.75f;
        if (overflow && k<64) v = std::ldexp(k<32 ? 1.75f : 0.875f, bits+1);
        if (outliers && k==64) v=14.0f;
        if (row%2) v=-v;
        values[row*K+k]=v;
        expected[row]+=v;
    }
    std::fill_n(static_cast<float *>(y->data),m*N,-12345.0f);
    potal_tile_plan plan;
    require(potal_frozen_plan(m,N,K,1,1,K/POTAL_HW_DIM,32/POTAL_HW_DIM,&plan)==POTAL_OK,"frozen plan");
    ggml_gemmini_args_t args;
    require(init_args(w,y,{int64_t(m),int64_t(N),int64_t(K)},"qwen2","main2-test",
                      ggml::gemmini::residual::ResidualRoute::ws_packet,plan,args),"args");
    hp1_weight weight;
    require(weight.allocate(N,K) && prepare_weight(args,weight)==quants::weight::hp1::status::ok,"weight");
    pipeline_config cfg;
    cfg.hart_roles=true; cfg.strict_issuer=true; cfg.paired=true; cfg.residual=true;
    cfg.job_capacity=capacity; cfg.backend="EMULATED"; cfg.layer="main2-test";
    potal_test::h2_emulator emu;
    uint32_t generation=0;
    size_t low_jobs=0, high_jobs=0, residual_rows=0;
    potal_test::thread_owner owner({}, [&](const potal_paired_request & req,potal_paired_report & report) {
        const bool high=(low_jobs+high_jobs)%2==1;
        if(high) ++high_jobs; else ++low_jobs;
        require(req.prepared && req.prepared->ready,"prepare both jobs before issuing");
        if(high) require(!req.residual && !req.c_residual,"high Main is a dense Main job");
        if(req.residual) residual_rows+=req.residual->rows.size();
        if(fail_high && high) {
            report.hw_status=POTAL_ERR_INVALID_ARGUMENT;
            report.stage="injected_high_failure";
            return potal_paired_status::invalid_argument;
        }
        auto copy=req;
        copy.generation=&generation;
        auto sink=emu.sink();
        return potal_paired_run_session(copy,&sink,report);
    });
    const auto result=run_stripe_pipeline(x,args,weight,plan,cfg,owner);
    owner.join();
    if(overflow || fail_high) {
        require(!result.ok,"refuse overflow or failed high GEMM");
        for(size_t i=0;i<m*N;++i) require(static_cast<float *>(y->data)[i]==-12345.0f,"no partial output commit");
        if(overflow) require(low_jobs==0 && high_jobs==0,"range guard before first issue");
    } else {
        if(!result.ok) std::fprintf(stderr,"stage=%s detail=%s\n",result.stage?result.stage:"-",result.detail.c_str());
        require(result.ok,"Main2 pipeline success");
        const size_t stripes=(m+POTAL_HW_DIM-1)/POTAL_HW_DIM;
        require(low_jobs==stripes && high_jobs==stripes,"two actual Main sessions per stripe");
        require((residual_rows!=0)==outliers,"residual only for initial outliers");
        require(result.counts.prepares_on_issuer_thread==0 && result.counts.prepares_on_producer==stripes,
                "no planning on the issuer");
        for(size_t row=0;row<m;++row) for(size_t col=0;col<N;++col)
            require(static_cast<float *>(y->data)[row*N+col]==expected[row],"exact scalar dot product");
    }
    require(emu.errors.empty(),"RoCC contract emulator errors");
    std::printf("MAIN2 PASS bits=%d rows=%zu capacity=%zu outliers=%d overflow=%d fail_high=%d low=%zu high=%zu\n",
                bits,m,capacity,int(outliers),int(overflow),int(fail_high),low_jobs,high_jobs);
}

int main() try {
#if !defined(__riscv)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    require(pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask) == 0, "read host affinity");
    std::string map;
    size_t count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE && count < 5; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) continue;
        if (count++) map += ",";
        map += std::to_string(cpu);
    }
    require(count == 5, "five CPUs for the host role test");
    setenv("GGML_POTAL_HART_MAP", map.c_str(), 1);
#endif
    setenv("GGML_POTAL_LINEAR_MAIN2","1",1);
    split_boundaries<4>(); split_boundaries<8>();
    for(size_t m:{size_t(1),size_t(2*POTAL_HW_DIM+3)})
        for(size_t capacity:{size_t(1),size_t(2)})
            for(bool outliers:{false,true}) run_case(m,capacity,outliers,false,false);
    run_case(1,1,false,true,false);
    run_case(1,1,true,false,true);
    return 0;
} catch(const std::exception & e) {
    std::fprintf(stderr,"MAIN2 FAIL: %s\n",e.what());
    return 1;
}
