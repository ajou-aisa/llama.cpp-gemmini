#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-gemmini.h"
#include "ggml-metal-cpu-exact.h"
#include "ggml-metal-cpu-exact-int.h"
#include "../tools/perplexity/perplexity-metal-cpu-exact.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct head_observer {
    ggml_tensor * tensor;
    uint64_t before=0;
    bool checked=false;
    static bool observe(ggml_tensor * node, bool ask, void * opaque) {
        auto & self=*static_cast<head_observer *>(opaque);
        if (node != self.tensor) { return !ask; }
        const uint64_t now=ggml_metal_cpu_exact_gpu_calls()+ggml_metal_cpu_exact_int_launches();
        if (ask) { self.before=now; return true; }
        self.checked=now==self.before;
        return self.checked;
    }
};

static std::vector<uint8_t> weights(ggml_type type, size_t k, size_t n) {
    std::vector<uint8_t> bytes(ggml_row_size(type,k)*n);
    if (type == GGML_TYPE_Q4_HP1) {
        std::vector<block_q4_hp1> blocks(k*n/32);
        for (size_t i=0; i<blocks.size(); ++i) {
            blocks[i].m=int16_t(i%3);
            blocks[i].channel_scale=float(i/(k/32)+1)/97.0f;
            for (size_t j=0; j<16; ++j) { blocks[i].qs[j]=uint8_t((i*53+j*19)&255); }
        }
        std::memcpy(bytes.data(),blocks.data(),bytes.size());
    } else if (type == GGML_TYPE_Q8_HP1) {
        std::vector<block_q8_hp1> blocks(k*n/32);
        for (size_t i=0; i<blocks.size(); ++i) {
            blocks[i].m=int16_t(i%3);
            blocks[i].channel_scale=float(i/(k/32)+1)/97.0f;
            for (size_t j=0; j<32; ++j) { blocks[i].qs[j]=int8_t(int((i*53+j*19)%256)-128); }
        }
        std::memcpy(bytes.data(),blocks.data(),bytes.size());
    } else {
        std::vector<float> original(k*n);
        for (size_t i=0; i<original.size(); ++i) { original[i]=float(int(i*53%257)-128)/113.0f; }
        require(ggml_quantize_chunk(type,original.data(),bytes.data(),0,n,k,nullptr)==bytes.size(),"Weight quantization failed");
    }
    return bytes;
}

enum class head_case { f16, q6_output, q6_tied, q6_optout, q6_wrong_weight, q6_wrong_result, missing_q6 };

static void fixture(ggml_type type, size_t rows, bool negative=false, head_case scenario=head_case::f16) {
    constexpr size_t k=64, vocab=13;
    const bool q6=scenario != head_case::f16 && scenario != head_case::missing_q6;
    const bool optin=scenario != head_case::f16 && scenario != head_case::q6_optout;
    const bool reject=scenario == head_case::q6_optout || scenario == head_case::q6_wrong_weight ||
                      scenario == head_case::q6_wrong_result || scenario == head_case::missing_q6;
    const size_t n=q6 ? 256 : 32;
    if (optin) {
        require(setenv("GGML_GEMMINI_METAL_CPU_EXACT_Q6_HEAD","1",1)==0,"Head environment update failed");
    } else {
        require(unsetenv("GGML_GEMMINI_METAL_CPU_EXACT_Q6_HEAD")==0,"Head environment reset failed");
    }
    ggml_backend_t gemmini=ggml_backend_gemmini_init(), cpu=ggml_backend_cpu_init();
    require(gemmini && cpu,"Backend initialization failed");
    ggml_backend_cpu_set_n_threads(cpu,2);
    ggml_backend_t backends[]={gemmini,cpu};
    ggml_backend_sched_t scheduler=ggml_backend_sched_new(backends,nullptr,2,64,false,true);
    require(scheduler != nullptr,"Scheduler initialization failed");
    ggml_context * ctx=ggml_init({1<<20,nullptr,true});
    require(ctx != nullptr,"Context initialization failed");
    ggml_tensor * w=ggml_new_tensor_2d(ctx,type,k,n);
    ggml_tensor * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,rows);
    ggml_tensor * head=ggml_new_tensor_2d(ctx,q6 ? GGML_TYPE_Q6_K : GGML_TYPE_F16,n,vocab);
    ggml_set_name(w,"blk.0.attn_q.weight");
    ggml_set_name(x,"fixture.input");
    ggml_set_name(head,scenario == head_case::q6_tied ? "token_embd.weight" :
                       scenario == head_case::q6_wrong_weight ? "blk.0.attn_q.weight" : "output.weight");
    ggml_set_input(x);
    ggml_set_input(w);
    ggml_set_input(head);
    ggml_tensor * dense=ggml_mul_mat(ctx,w,x);
    ggml_tensor * norm=ggml_rms_norm(ctx,dense,1.0e-5f);
    ggml_tensor * silu=ggml_silu(ctx,norm);
    ggml_tensor * logits=ggml_mul_mat(ctx,head,silu);
    ggml_tensor * nodes[]={dense,norm,silu,logits};
    const char * names[]={"fixture.dense","fixture.rms_norm","fixture.silu",
                         scenario == head_case::q6_wrong_result ? "result_output_extra" : "result_output"};
    for (size_t i=0; i<4; ++i) { ggml_set_name(nodes[i],names[i]); ggml_set_output(nodes[i]); }
    ggml_cgraph * graph=ggml_new_graph_custom(ctx,64,false);
    ggml_build_forward_expand(graph,logits);
    ggml_backend_sched_set_tensor_backend(scheduler,dense,negative ? cpu : gemmini);
    ggml_backend_sched_set_tensor_backend(scheduler,norm,cpu);
    ggml_backend_sched_set_tensor_backend(scheduler,silu,cpu);
    if (q6) { ggml_backend_sched_set_tensor_backend(scheduler,logits,cpu); }
    require(ggml_backend_sched_alloc_graph(scheduler,graph),"Graph allocation failed");
    ggml_backend_t head_backend=ggml_backend_sched_get_tensor_backend(scheduler,logits);
    const auto packed=weights(type,k,n);
    std::vector<float> activation(k*rows);
    for (size_t i=0; i<activation.size(); ++i) { activation[i]=float(int(i*71%129)-64)/127.0f; }
    activation[0]=64.0f;
    std::vector<uint8_t> head_data;
    if (q6) { head_data=weights(GGML_TYPE_Q6_K,n,vocab); }
    else {
        std::vector<ggml_fp16_t> half(n*vocab);
        for (size_t i=0; i<half.size(); ++i) { half[i]=ggml_fp32_to_fp16(float(int(i*37%97)-48)/97.0f); }
        head_data.resize(half.size()*sizeof(ggml_fp16_t));
        std::memcpy(head_data.data(),half.data(),head_data.size());
    }
    const uint64_t initial_float=ggml_metal_cpu_exact_gpu_calls(), initial_int=ggml_metal_cpu_exact_int_launches();
    std::vector<std::vector<uint8_t>> reference;
    size_t compared=0;
    for (int enabled=0; enabled<2; ++enabled) {
        std::fprintf(stderr,"GRAPH_RUN type=%s rows=%zu metal_exact=%d negative=%d head_case=%d\n",ggml_type_name(type),rows,enabled,negative,int(scenario));
        require(setenv("GGML_GEMMINI_METAL_CPU_EXACT",enabled ? "1" : "0",1)==0,"Environment update failed");
        ggml_backend_tensor_set(w,packed.data(),0,packed.size());
        ggml_backend_tensor_set(x,activation.data(),0,activation.size()*sizeof(float));
        ggml_backend_tensor_set(head,head_data.data(),0,head_data.size());
        common_params params;
        params.n_gpu_layers=0;
        params.warmup=false;
        head_observer observer{logits};
        params.cb_eval=head_observer::observe;
        params.cb_eval_user_data=&observer;
        perplexity_metal_cpu_exact_guard guard;
        const bool guarded=enabled && (!TEST_CPU_EXACT_FLOAT || type ==
            (GGML_GEMMINI_ACTIVATION_BITS == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0));
        if (guarded) {
            guard.install(params);
        }
        ggml_backend_sched_set_eval_callback(scheduler,params.cb_eval,params.cb_eval_user_data);
        const ggml_status status=ggml_backend_sched_graph_compute(scheduler,graph);
        ggml_backend_sched_synchronize(scheduler);
        if (enabled && negative) {
            require(guarded && !guard.finish(status==GGML_STATUS_SUCCESS,int(rows)) && guard.failed && guard.verified==0,
                    "PPL guard accepted CPU-only quantized matmul");
            require(ggml_metal_cpu_exact_gpu_calls()==initial_float && ggml_metal_cpu_exact_int_launches()==initial_int,
                    "CPU-only negative graph unexpectedly used GPU");
            std::printf("PASS PPL guard rejects CPU-only quantized matmul\n");
            break;
        }
        if (enabled && reject) {
            require(guarded && !guard.finish(status==GGML_STATUS_SUCCESS,int(rows)),"PPL guard accepted invalid Q6 head scenario");
            require(guard.cpu_q6_head_matmuls==0 && guard.verified==1,"Invalid head affected body GPU verification");
            require(scenario == head_case::missing_q6 ? !guard.failed : guard.failed,"Unexpected Q6 head failure reason");
            std::printf("PASS Q6 head rejection head_case=%d body_gpu_verified=1\n",int(scenario));
            break;
        }
        require(status==GGML_STATUS_SUCCESS,"Graph compute failed");
        require(observer.checked,"Head did not retain CPU arithmetic");
        if (guarded) {
            require(guard.finish(true,int(rows)),"PPL guard rejected valid GPU graph");
            require(guard.cpu_q6_head_matmuls==(q6 ? 1 : 0) && guard.observed==1 && guard.verified==1,
                    "Incorrect body or CPU head proof count");
        }
        for (size_t i=0; i<4; ++i) {
            require(ggml_backend_sched_get_tensor_backend(scheduler,nodes[i])==
                    (i == 3 ? head_backend : i == 0 && !negative ? gemmini : cpu),"Unexpected tensor backend");
            std::vector<uint8_t> output(ggml_nbytes(nodes[i]));
            ggml_backend_tensor_get(nodes[i],output.data(),0,output.size());
            if (!enabled) { reference.push_back(output); }
            else {
                if (reference[i] != output) {
                    std::fprintf(stderr,"Mismatch type=%s rows=%zu node=%s\n",ggml_type_name(type),rows,names[i]);
                    for (size_t j=0; j<output.size(); j+=4) {
                        uint32_t expected,actual;
                        std::memcpy(&expected,reference[i].data()+j,4);
                        std::memcpy(&actual,output.data()+j,4);
                        if (expected != actual) { std::fprintf(stderr,"index=%zu CPU=%08x GPU=%08x\n",j/4,expected,actual); break; }
                    }
                    throw std::runtime_error("CPU/Metal graph output mismatch");
                }
                compared += output.size()/sizeof(float);
            }
        }
        if (!enabled) {
            require(ggml_metal_cpu_exact_gpu_calls()==initial_float && ggml_metal_cpu_exact_int_launches()==initial_int,"CPU oracle used GPU");
        }
    }
    const uint64_t float_calls=ggml_metal_cpu_exact_gpu_calls()-initial_float;
    const uint64_t int_calls=ggml_metal_cpu_exact_int_launches()-initial_int;
    if (!negative && !reject) {
      require(TEST_CPU_EXACT_FLOAT ? float_calls>0 && int_calls==0 : int_calls>0 && float_calls==0,"Requested Metal arithmetic did not execute");
      std::printf("PASS scheduler type=%s rows=%zu bitwise_outputs=%zu float_gpu_calls=%llu int_gpu_calls=%llu CPU_rms_norm_silu head_type=%s head_weight=%s head_backend=%s\n",
                ggml_type_name(type),rows,compared,(unsigned long long)float_calls,(unsigned long long)int_calls,
                ggml_type_name(head->type),head->name,ggml_backend_name(head_backend));
    }
    ggml_backend_sched_free(scheduler);
    ggml_free(ctx);
    ggml_backend_free(gemmini);
    ggml_backend_free(cpu);
}

static void hp1_missing_gpu_proof(ggml_type type) {
    ggml_context * ctx=ggml_init({1<<16,nullptr,true});
    require(ctx != nullptr,"Negative proof context initialization failed");
    ggml_tensor * w=ggml_new_tensor_2d(ctx,type,32,4);
    ggml_tensor * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,1);
    ggml_tensor * dense=ggml_mul_mat(ctx,w,x);
    common_params params;
    params.n_gpu_layers=0; params.warmup=false;
    perplexity_metal_cpu_exact_guard guard;
    require(setenv("GGML_GEMMINI_METAL_CPU_EXACT","1",1)==0,"Environment update failed");
    guard.install(params);
    require(params.cb_eval(dense,true,params.cb_eval_user_data) && !guard.failed,"Valid HP1 type rejected before execution");
    require(!params.cb_eval(dense,false,params.cb_eval_user_data) && guard.failed && !guard.finish(true,1),
            "HP1 proof accepted missing GPU execution");
    ggml_free(ctx);
    std::printf("PASS HP1 proof callback rejects missing GPU execution\n");
}

int main() {
    try {
        std::vector<ggml_type> types;
        if (TEST_CPU_EXACT_FLOAT) { types={GGML_TYPE_Q4_0,GGML_TYPE_Q8_0}; }
        else if (TEST_CPU_EXACT_HP1) { types={GGML_GEMMINI_ACTIVATION_BITS == 4 ? GGML_TYPE_Q4_HP1 : GGML_TYPE_Q8_HP1}; }
        else { types={GGML_GEMMINI_ACTIVATION_BITS == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0}; }
        for (ggml_type type : types) { fixture(type,1); fixture(type,5); }
        if (!TEST_CPU_EXACT_HP1) {
            fixture(TEST_CPU_EXACT_FLOAT ? (GGML_GEMMINI_ACTIVATION_BITS == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0) : types.front(),1,true);
        } else { hp1_missing_gpu_proof(types.front()); }
        const ggml_type body_type=TEST_CPU_EXACT_FLOAT ?
            (GGML_GEMMINI_ACTIVATION_BITS == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0) : types.front();
        fixture(body_type,1,false,head_case::q6_output);
        fixture(body_type,5,false,head_case::q6_tied);
        if (!TEST_CPU_EXACT_HP1) { fixture(body_type,1,true,head_case::q6_output); }
        fixture(body_type,1,false,head_case::q6_optout);
        fixture(body_type,1,false,head_case::q6_wrong_weight);
        fixture(body_type,1,false,head_case::q6_wrong_result);
        fixture(body_type,1,false,head_case::missing_q6);
        require(setenv("GGML_GEMMINI_METAL_CPU_EXACT","0",1)==0,"Environment update failed");
        common_params params;
        perplexity_metal_cpu_exact_guard inactive;
        inactive.install(params);
        require(!inactive.active && !inactive.allow_q6_head && inactive.finish(false,0),"Q6 head opt-in affected disabled exact mode");
        std::printf("PASS Q6 head opt-in ignored outside exact mode\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL %s\n",error.what());
        return 1;
    }
}
