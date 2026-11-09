#include "../ggml/src/ggml-cuda/cpu-exact/ggml-cuda-rtn.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * reason) { if (!ok) throw std::runtime_error(reason); }
static float half(float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); }
static int code(const ggml_tensor * w, int row, int k) {
    size_t b=size_t(row)*(w->ne[0]/32)+k/32; int p=k%32;
    if (w->type==GGML_TYPE_Q4_0) {
        auto & x=static_cast<const block_q4_0 *>(w->data)[b];
        return int((x.qs[p%16]>>(p<16?0:4))&15)-8;
    }
    return static_cast<const block_q8_0 *>(w->data)[b].qs[p];
}
static float scale(const ggml_tensor * w, int row, int block) {
    size_t b=size_t(row)*(w->ne[0]/32)+block;
    if (w->type==GGML_TYPE_MXINT4) {
        const auto & value=static_cast<const block_q4_hp1 *>(w->data)[b];
        return value.m==INT16_MIN?0.0f:std::ldexp(value.channel_scale,value.m);
    }
    if (w->type==GGML_TYPE_MXINT8) {
        const auto & value=static_cast<const block_q8_hp1 *>(w->data)[b];
        return value.m==INT16_MIN?0.0f:std::ldexp(value.channel_scale,value.m);
    }
    return ggml_fp16_to_fp32(w->type==GGML_TYPE_Q4_0 ? static_cast<const block_q4_0 *>(w->data)[b].d : static_cast<const block_q8_0 *>(w->data)[b].d);
}
static bool benchmark_mode=false;
static void dense(ggml_type type, int bits, int m, int n, int k) {
    setenv("GGML_CUDA_RTN_A_BITS",std::to_string(bits).c_str(),1);
    ggml_cuda_rtn_clear();
    auto * ctx=ggml_init({size_t(16)<<20,nullptr,false}); require(ctx,"ggml_init");
    auto * w=ggml_new_tensor_2d(ctx,type,k,n); auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);
    std::vector<float> values(size_t(n)*k);
    for (size_t i=0;i<values.size();++i) values[i]=std::sin(float(i)*0.013f)*0.7f;
    const bool mxint=type==GGML_TYPE_MXINT4 || type==GGML_TYPE_MXINT8;
    if (mxint) {
        for (int c=0;c<n;++c) {
            for (int j=0;j<k;++j)
                values[size_t(c)*k+j]=j<32?0.0f:std::ldexp(values[size_t(c)*k+j],(j/32+c)%7-3);
            if (type==GGML_TYPE_MXINT4) {
                auto * blocks=static_cast<block_q4_hp1 *>(w->data)+size_t(c)*(k/32);
                quantize_row_q4_hp1_ref(values.data()+size_t(c)*k,blocks,k);
                dequantize_row_q4_hp1(blocks,values.data()+size_t(c)*k,k);
            } else {
                auto * blocks=static_cast<block_q8_hp1 *>(w->data)+size_t(c)*(k/32);
                quantize_row_q8_hp1_ref(values.data()+size_t(c)*k,blocks,k);
                dequantize_row_q8_hp1(blocks,values.data()+size_t(c)*k,k);
            }
        }
    } else
    if (type==GGML_TYPE_F16) ggml_fp32_to_fp16_row(values.data(),static_cast<ggml_fp16_t *>(w->data),values.size());
    else if (type==GGML_TYPE_Q4_0) quantize_row_q4_0_ref(values.data(),static_cast<block_q4_0 *>(w->data),values.size());
    else if (type==GGML_TYPE_Q8_0) quantize_row_q8_0_ref(values.data(),static_cast<block_q8_0 *>(w->data),values.size());
    else {
        std::fill(values.begin(),values.begin()+256,0.0f);
        quantize_row_q6_K_ref(values.data(),static_cast<block_q6_K *>(w->data),values.size());
        // Independent oracle uses the existing llama.cpp CPU decoder, not CUDA bit extraction.
        dequantize_row_q6_K(static_cast<block_q6_K *>(w->data),values.data(),values.size());
    }
    auto * a=static_cast<float *>(x->data);
    for (int i=0;i<m*k;++i) a[i]=i<32?0.0f:std::cos(float(i)*0.021f)*1.3f;
    // Exactly representable .5 ties in a block with a unit scale.
    if (m*k>=64) { std::fill(a+32,a+64,0.0f); a[32]=float((1<<(bits==16?7:bits-1))-1); a[33]=0.5f; a[34]=1.5f; a[35]=-0.5f; a[36]=-1.5f; }
    if (mxint && m*k>=96) a[64]=1e6f; // MXINT retains this range without PoTal outlier selection.
    auto * out=ggml_mul_mat(ctx,w,x);
    require(ggml_cuda_rtn_supported(out),"dense unsupported");
    const auto cold_start=std::chrono::steady_clock::now();
    require(ggml_cuda_rtn_compute(out),ggml_cuda_rtn_error());
    const double cold_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
    if (benchmark_mode) {
        const auto start=std::chrono::steady_clock::now();
        for (int repeat=0;repeat<10;++repeat) require(ggml_cuda_rtn_compute(out),ggml_cuda_rtn_error());
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/10;
        uint64_t hash=14695981039346656037ULL;
        for (size_t i=0;i<ggml_nbytes(out);++i) {
            hash^=static_cast<const unsigned char *>(out->data)[i]; hash*=1099511628211ULL;
        }
        std::printf("CUDA_RTN_BENCH type=%s A%d m=%d n=%d k=%d mean_ms=%.6f cold_ms=%.6f output_hash=%016llx\n",
                    ggml_type_name(type),bits,m,n,k,ms,cold_ms,static_cast<unsigned long long>(hash));
    }
    // Exercise cache reuse with changed activations.
    a[m*k-1]*=-1;
    require(ggml_cuda_rtn_compute(out),ggml_cuda_rtn_error());
    if (type==GGML_TYPE_Q6_K && bits==16) {
        auto * decoded=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,k,n);
        ggml_fp32_to_fp16_row(values.data(),static_cast<ggml_fp16_t *>(decoded->data),values.size());
        auto * reference=ggml_mul_mat(ctx,decoded,x);
        require(ggml_cuda_rtn_compute(reference),ggml_cuda_rtn_error());
        for (int i=0;i<m*n;++i) require(static_cast<float *>(out->data)[i]==static_cast<float *>(reference->data)[i],
                                      "Q6_K CUDA decode differs from existing CPU decoder before FP16 GEMM");
        std::printf("PASS Q6_K CPU/CUDA decode parity m=%d n=%d k=%d\n",m,n,k);
    }
    float max_error=0;
    for (int r=0;r<m;++r) for (int c=0;c<n;++c) {
        double sum=0, absolute_products=0;
        if (bits==16 || type==GGML_TYPE_F16) {
            for (int j=0;j<k;++j) {
                float v=type==GGML_TYPE_F16?ggml_fp16_to_fp32(static_cast<ggml_fp16_t *>(w->data)[size_t(c)*k+j]):
                    type==GGML_TYPE_Q6_K?half(values[size_t(c)*k+j]):half(float(code(w,c,j))*scale(w,c,j/32));
                double product=double(v)*half(a[size_t(r)*k+j]);
                sum+=product; absolute_products+=std::fabs(product);
            }
        } else for (int b=0;b<k/32;++b) {
            float largest=0; int limit=(1<<(bits-1))-1;
            for (int j=0;j<32;++j) largest=std::max(largest,std::fabs(a[size_t(r)*k+b*32+j]));
            float s=largest==0?1.0f:mxint?std::ldexp(1.0f,std::max(-127,std::ilogb(largest))-(bits-2)):
                                      float(double(largest)/limit); int dot=0;
            float ws=type==GGML_TYPE_Q6_K?ggml_fp16_to_fp32(static_cast<block_q6_K *>(w->data)[size_t(c)*(k/256)+b/8].d):scale(w,c,b);
            for (int j=0;j<32;++j) {
                const double value=double(a[size_t(r)*k+b*32+j])/double(s);
                int q=std::clamp(int(mxint?std::round(value):std::nearbyint(value)),mxint?-limit:-limit-1,limit);
                int weight_code=(type==GGML_TYPE_Q6_K || mxint)?(ws==0?0:int(std::nearbyint(double(values[size_t(c)*k+b*32+j])/double(ws)))):code(w,c,b*32+j);
                dot+=q*weight_code;
            }
            sum+=double(dot)*(double(ws)*double(s));
        }
        float expected=float(sum), got=static_cast<float *>(out->data)[size_t(r)*n+c];
        max_error=std::max(max_error,std::fabs(expected-got));
        require(std::isfinite(got),"non-finite CUDA result");
        if (bits!=16 && type!=GGML_TYPE_F16) require(expected==got,"RTN-WA differs from scalar BLOCK oracle");
        else {
            double tolerance=2e-4+std::fabs(expected)*2e-5;
            if (type==GGML_TYPE_Q6_K || (benchmark_mode && type==GGML_TYPE_F16)) {
                // Q6 decode already has exact CPU-decoder/GEMM parity above.
                // Long cancellation dots, including the F16 benchmark, need the
                // F32 bound gamma_k * sum|xy| against F64. Benchmark output hashes
                // are also compared byte-for-byte with the frozen implementation.
                const double u=std::numeric_limits<float>::epsilon()/2.0;
                tolerance=std::max(tolerance,(k*u)/(1-k*u)*absolute_products+u*std::fabs(sum));
            }
            if (std::fabs(expected-got)>tolerance)
                std::fprintf(stderr,"FP16 mismatch type=%s m=%d n=%d k=%d row=%d col=%d got=%.9g expected=%.9g error=%.9g tolerance=%.9g\n",
                             ggml_type_name(type),m,n,k,r,c,got,expected,std::fabs(expected-got),tolerance);
            require(std::fabs(expected-got)<=tolerance,"FP16 CUDA dot exceeds FP32 accumulation tolerance");
        }
    }
    std::printf("PASS dense type=%s A%d m=%d n=%d k=%d max_error=%.9g\n",ggml_type_name(type),bits,m,n,k,max_error);
    ggml_cuda_rtn_clear(); ggml_free(ctx);
}
static void mxint_ties_away(int bits, int requested_bits) {
    setenv("GGML_CUDA_RTN_A_BITS",std::to_string(requested_bits).c_str(),1);
    ggml_cuda_rtn_clear();
    auto * ctx=ggml_init({size_t(1)<<20,nullptr,false}); require(ctx,"ties fixture context");
    auto * w=ggml_new_tensor_2d(ctx,bits==4?GGML_TYPE_MXINT4:GGML_TYPE_MXINT8,32,1);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,10);
    std::memset(w->data,0,ggml_nbytes(w)); std::memset(x->data,0,ggml_nbytes(x));
    if (bits==4) {
        auto & block=*static_cast<block_q4_hp1 *>(w->data);
        std::memset(block.qs,0x88,sizeof(block.qs)); block.qs[0]=0x89; block.channel_scale=1;
    } else { auto & block=*static_cast<block_q8_hp1 *>(w->data); block.qs[0]=1; block.channel_scale=1; }
    const float below=std::nextafter(.5f,0.0f), above=std::nextafter(.5f,1.0f);
    const float ties[]{.5f,1.5f,2.5f,-.5f,-1.5f,-2.5f,below,above,-below,-above};
    const int expected[]{1,2,3,-1,-2,-3,0,1,0,-1};
    const float step=std::ldexp(1.0f,-(bits-2));
    auto * data=static_cast<float *>(x->data);
    for (int row=0;row<10;++row) { data[row*32]=ties[row]*step; data[row*32+31]=1; }
    auto * output=ggml_mul_mat(ctx,w,x);
    require(ggml_cuda_rtn_supported(output),"MXINT weight must select quantized activation");
    require(ggml_cuda_rtn_compute(output),ggml_cuda_rtn_error());
    for (int row=0;row<10;++row)
        require(static_cast<float *>(output->data)[row]==expected[row]*step,"MXINT activation PoT/ties-away mismatch");
    ggml_cuda_rtn_clear(); ggml_free(ctx);
}
static void mxint_scale_edges(int bits) {
    setenv("GGML_CUDA_RTN_A_BITS",std::to_string(bits).c_str(),1);
    auto * ctx=ggml_init({size_t(1)<<20,nullptr,false});
    require(ctx,"scale fixture context");
    auto * w=ggml_new_tensor_2d(ctx,bits==4?GGML_TYPE_MXINT4:GGML_TYPE_MXINT8,32,1);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,2);
    auto * out=ggml_mul_mat(ctx,w,x);
    const int limit=(1<<(bits-1))-1;
    for (int we:{-149,-133,-24,-1,0,17,100,127}) {
        const float ws=std::ldexp(1.0f,we);
        std::memset(w->data,0,ggml_nbytes(w));
        if (bits==4) {
            auto & block=*static_cast<block_q4_hp1 *>(w->data);
            std::memset(block.qs,0xff,sizeof(block.qs));
            block.channel_scale=ws;
        } else {
            auto & block=*static_cast<block_q8_hp1 *>(w->data);
            std::memset(block.qs,127,sizeof(block.qs));
            block.channel_scale=ws;
        }
        for (int maximum_exponent:{-126,-63,-1,0,31,80,126}) {
            const float as=std::ldexp(1.0f,maximum_exponent-(bits-2));
            auto * input=static_cast<float *>(x->data);
            std::fill(input,input+32,float(limit)*as);
            std::fill(input+32,input+64,-float(limit)*as);
            ggml_cuda_rtn_clear();
            require(ggml_cuda_rtn_compute(out),ggml_cuda_rtn_error());
            const double scale_product=double(ws)*double(as);
            const float expected[]{float(double(32*limit*limit)*scale_product),
                                   float(-double(32*limit*limit)*scale_product)};
            require(std::memcmp(out->data,expected,sizeof(expected))==0,
                    "MXINT scale extremes changed ordered-F64 output bits");
        }
    }
    ggml_cuda_rtn_clear(); ggml_free(ctx);
    std::printf("PASS MXINT%d signed scale extremes and F32 output boundaries\n",bits);
}
static float value(const ggml_tensor * t,int k,int n,int h) {
    auto * p=static_cast<const char *>(t->data)+k*t->nb[0]+n*t->nb[1]+h*t->nb[2];
    return t->type==GGML_TYPE_F16?ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p)):half(*reinterpret_cast<const float *>(p));
}
static void attention(bool transposed) {
    setenv("GGML_CUDA_RTN_A_BITS","4",1);
    auto * ctx=ggml_init({size_t(8)<<20,nullptr,false});
    const int k=64,n=37,m=9;
    auto * storage=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,k,transposed?2:n,transposed?n:2);
    auto * p=static_cast<ggml_fp16_t *>(storage->data);
    for (int i=0;i<k*n*2;++i) p[i]=ggml_fp32_to_fp16(std::sin(float(i)*0.03f));
    auto * w=transposed?ggml_permute(ctx,storage,0,2,1,3):storage;
    auto * x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,m,14);
    for (int i=0;i<k*m*14;++i) static_cast<float *>(x->data)[i]=std::cos(float(i)*0.023f);
    auto * out=ggml_mul_mat(ctx,w,x);
    require(ggml_cuda_rtn_compute(out),ggml_cuda_rtn_error());
    for (int h=0;h<14;++h) for (int r=0;r<m;++r) for (int c=0;c<n;++c) {
        double expected=0; for (int j=0;j<k;++j) expected+=double(value(w,j,c,h/7))*value(x,j,r,h);
        float got=static_cast<float *>(out->data)[(h*m+r)*n+c];
        require(std::isfinite(got) && std::fabs(got-expected)<2e-4+std::fabs(expected)*2e-5,"GQA/strided attention mismatch");
    }
    std::printf("PASS attention GQA=14/2 transposed=%d\n",int(transposed)); ggml_free(ctx);
}
int main(int argc, char ** argv) {
    benchmark_mode=argc==2 && (!std::strcmp(argv[1],"--benchmark") || !std::strcmp(argv[1],"--benchmark-large"));
    try {
        if (benchmark_mode && !std::strcmp(argv[1],"--benchmark-large")) { dense(GGML_TYPE_MXINT4,4,512,2048,2048); return 0; }
        mxint_ties_away(4,4); mxint_ties_away(8,8);
        mxint_ties_away(4,16); mxint_ties_away(8,16);
        mxint_scale_edges(4); mxint_scale_edges(8);
        if (benchmark_mode) {
            dense(GGML_TYPE_F16,16,1,512,2048);
            dense(GGML_TYPE_F16,16,128,512,2048);
            for (auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_MXINT4,GGML_TYPE_MXINT8})
                for (int bits:{4,8}) dense(type,bits,128,512,2048);
            return 0;
        }
        for (auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q8_0}) for (int bits:{4,8,16}) {
            dense(type,bits,9,35,64); dense(type,bits,17,65,896);
        }
        for (int bits:{4,8,16}) { dense(GGML_TYPE_Q6_K,bits,9,35,256); dense(GGML_TYPE_Q6_K,bits,17,67,2048); }
        for (auto type:{GGML_TYPE_MXINT4,GGML_TYPE_MXINT8}) for (int bits:{4,8}) {
            dense(type,bits,9,35,64); dense(type,bits,17,65,896); dense(type,bits,9,67,2048);
        }
        for (auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_MXINT4,GGML_TYPE_MXINT8})
            for (int bits:{4,8}) dense(type,bits,9,521,64);
        // Cover the prefill tile with partial row/column blocks in regular CTest.
        for (auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q8_0}) for (int bits:{4,8})
            dense(type,bits,129,515,64);
        dense(GGML_TYPE_F16,16,9,35,64); attention(false); attention(true);
        std::puts("PASS CUDA RTN scalar oracle, ties, zeros, tile tails, weight cache, Qwen K and GQA"); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
