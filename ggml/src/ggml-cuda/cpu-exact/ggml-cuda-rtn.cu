#include "ggml-cuda-rtn.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
thread_local std::string error;
std::atomic<uint64_t> dense_calls{0}, attention_calls{0};
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(cublasStatus_t e) { if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(e)); }
struct buffer {
    void * p = nullptr;
    size_t capacity = 0;
    ~buffer() { if (p) cudaFree(p); }
    void reserve(size_t n) {
        if (n <= capacity) return;
        if (p) { check(cudaFree(p)); p = nullptr; capacity = 0; }
        check(cudaMalloc(&p, n)); capacity = n;
    }
};
struct weight { buffer data; ggml_type type; int64_t k, n; bool half; };
struct runtime {
    std::mutex mutex;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    buffer input, activation, scales, output, weights, staging;
    std::unordered_map<const void *, std::unique_ptr<weight>> cache;
    void init() {
        if (stream) return;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cublasCreate(&blas)); check(cublasSetStream(blas, stream));
        check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        int device; cudaDeviceProp info{};
        check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&info, device));
        std::fprintf(stderr, "CUDA_RTN_DEVICE name=%s sm=%d%d activation_bits=%d block=32 compensation=off\n",
                     info.name, info.major, info.minor, ggml_cuda_rtn_activation_bits());
    }
    ~runtime() { if (blas) cublasDestroy(blas); if (stream) cudaStreamDestroy(stream); }
};
runtime & state() { static runtime r; return r; }
struct layout { size_t ne[4], nb[4]; int type; };
layout describe(const ggml_tensor * t) {
    layout l{}; l.type = t->type;
    for (int d=0; d<4; ++d) { l.ne[d]=t->ne[d]; l.nb[d]=t->nb[d]; }
    return l;
}
__global__ void pack_half(const char * source, __half * dest, size_t count, layout s) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) return;
    size_t j=i, offset=0;
    for (int d=0; d<4; ++d) { offset += (j%s.ne[d])*s.nb[d]; j/=s.ne[d]; }
    dest[i] = s.type == GGML_TYPE_F16 ? *reinterpret_cast<const __half *>(source+offset)
                                   : __float2half_rn(*reinterpret_cast<const float *>(source+offset));
}
// Read the existing GGML Q6_K layout; this does not quantize or alter model weights.
__device__ int q6_code(const uint8_t * b, int pos) {
    int half=pos/128, quadrant=(pos%128)/32, lane=pos%32;
    int low=(b[half*64+lane+(quadrant%2)*32]>>(quadrant>=2?4:0))&15;
    int high=(b[128+half*32+lane]>>(quadrant*2))&3;
    return (low|(high<<4))-32;
}
bool hp1_weight(ggml_type type) { return type==GGML_TYPE_Q4_HP1 || type==GGML_TYPE_Q8_HP1; }
int weight_bits(ggml_type type) { return type==GGML_TYPE_Q4_0 || type==GGML_TYPE_Q4_HP1?4:type==GGML_TYPE_Q6_K?6:8; }
__global__ void decode_half(const uint8_t * source, __half * dest, size_t count, int bits) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    if (bits==6) {
        const uint8_t * b=source+(i/256)*210; int pos=i%256;
        float d=__half2float(*reinterpret_cast<const __half *>(b+208));
        dest[i]=__float2half_rn(d*float(static_cast<int8_t>(b[192+pos/16]))*float(q6_code(b,pos)));
        return;
    }
    const uint8_t * b=source+(i/32)*(bits==4?18:34);
    int k=i%32;
    int q=bits==4?int((b[2+k%16]>>(k<16?0:4))&15)-8:int(static_cast<int8_t>(b[2+k]));
    dest[i]=__float2half_rn(__half2float(*reinterpret_cast<const __half *>(b))*float(q));
}
// Match Gemmini BLOCK: F32 scale from double max/qmax, nearest-even double quotient.
__global__ void quantize_activation(const float * x, int8_t * q, float * scales, size_t groups, int bits) {
    size_t b=size_t(blockIdx.x)*blockDim.x/32+threadIdx.x/32;
    unsigned lane=threadIdx.x%32;
    if (b>=groups) return;
    float value=x[b*32+lane], a=isfinite(value)?fabsf(value):0.0f;
    for (int d=16;d;d/=2) a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
    int limit=bits==4?7:127;
    float scale=a==0?1.0f:float(double(a)/double(limit));
    int code=isfinite(value)?__double2int_rn(double(value)/double(scale)):0;
    q[b*32+lane]=int8_t(max(-limit-1,min(limit,code)));
    if (!lane) scales[b]=scale;
}
// Integer products and ordered F64 scale restoration, with no fragment transfers to CPU.
__global__ void block_matmul(const int8_t * a, const float * scales, const uint8_t * w,
                             float * output, int m, int n, int k, int bits, bool hp1) {
    __shared__ int av[8][8], wv[32][8];
    __shared__ int subscale[32][2];
    __shared__ float as[8], ws[32];
    int t=threadIdx.y*32+threadIdx.x, row=blockIdx.y*8+threadIdx.y, col=blockIdx.x*32+threadIdx.x;
    int blocks=k/32, stride=hp1?(bits==4?24:40):(bits==4?18:34);
    double sum=0.0;
    for (int b=0;b<blocks;++b) {
        for (int i=t;i<40*8;i+=256) {
            int r=i/8, group=i%8;
            int packed=0;
            if (r<8) {
                int rr=blockIdx.y*8+r;
                if (rr<m) packed=reinterpret_cast<const int *>(a)[(size_t(rr)*blocks+b)*8+group];
                av[r][group]=packed;
                if (!group) as[r]=rr<m?scales[size_t(rr)*blocks+b]:0;
            } else {
                int cc=blockIdx.x*32+r-8;
                if (cc<n) {
                    const uint8_t * wb=bits==6?w+(size_t(cc)*(k/256)+b/8)*210:w+(size_t(cc)*blocks+b)*stride;
                    unsigned p=0;
                    for (int j=0;j<4;++j) {
                        int pos=group*4+j;
                        const int offset=hp1?0:2;
                        int v=bits==6?q6_code(wb,(b%8)*32+pos):bits==4?int((wb[offset+pos%16]>>(pos<16?0:4))&15)-8:int(static_cast<int8_t>(wb[offset+pos]));
                        p|=unsigned(uint8_t(v))<<(j*8);
                    }
                    packed=int(p);
                    if (!group) {
                        if (hp1) {
                            const int offset=bits==4?16:32;
                            const int exponent=*reinterpret_cast<const int16_t *>(wb+offset);
                            const float channel=*reinterpret_cast<const float *>(wb+offset+4);
                            ws[r-8]=exponent==INT16_MIN?0.0f:ldexpf(channel,exponent);
                        } else {
                            ws[r-8]=__half2float(*reinterpret_cast<const __half *>(wb+(bits==6?208:0)));
                        }
                        if (bits==6) for (int j=0;j<2;++j) subscale[r-8][j]=static_cast<int8_t>(wb[192+(b%8)*2+j]);
                    }
                } else if (!group) { ws[r-8]=0; subscale[r-8][0]=subscale[r-8][1]=0; }
                wv[r-8][group]=packed;
            }
        }
        __syncthreads();
        int dot0=0,dot1=0;
#pragma unroll
        for (int group=0;group<4;++group) dot0=__dp4a(av[threadIdx.y][group],wv[threadIdx.x][group],dot0);
#pragma unroll
        for (int group=4;group<8;++group) dot1=__dp4a(av[threadIdx.y][group],wv[threadIdx.x][group],dot1);
        int dot=bits==6?dot0*subscale[threadIdx.x][0]+dot1*subscale[threadIdx.x][1]:dot0+dot1;
        sum=__dadd_rn(sum,__dmul_rn(double(dot),__dmul_rn(double(ws[threadIdx.x]),double(as[threadIdx.y]))));
        __syncthreads();
    }
    if (row<m && col<n) output[size_t(row)*n+col]=float(sum);
}

const void * model_weight(runtime & r, const ggml_tensor * w, bool half) {
    auto it=r.cache.find(w->data);
    if (it!=r.cache.end()) {
        if (it->second->type!=w->type || it->second->k!=w->ne[0] || it->second->n!=w->ne[1] || it->second->half!=half)
            throw std::runtime_error("Cached weight identity changed; clear CUDA RTN cache");
        return it->second->data.p;
    }
    auto item=std::make_unique<weight>(); item->type=w->type; item->k=w->ne[0]; item->n=w->ne[1]; item->half=half;
    size_t count=ggml_nelements(w), bytes=ggml_nbytes(w);
    item->data.reserve(half?count*sizeof(__half):bytes);
    if (half && w->type!=GGML_TYPE_F16) {
        r.staging.reserve(bytes); check(cudaMemcpyAsync(r.staging.p,w->data,bytes,cudaMemcpyHostToDevice,r.stream));
        decode_half<<<(count+255)/256,256,0,r.stream>>>(static_cast<const uint8_t *>(r.staging.p),static_cast<__half *>(item->data.p),count,weight_bits(w->type));
    } else check(cudaMemcpyAsync(item->data.p,w->data,bytes,cudaMemcpyHostToDevice,r.stream));
    const void * result=item->data.p; r.cache.emplace(w->data,std::move(item)); return result;
}
void pack(runtime & r, const ggml_tensor * t, buffer & dest) {
    size_t bytes=ggml_nbytes(t), count=ggml_nelements(t);
    r.staging.reserve(bytes); dest.reserve(count*sizeof(__half));
    check(cudaMemcpyAsync(r.staging.p,t->data,bytes,cudaMemcpyHostToDevice,r.stream));
    pack_half<<<(count+255)/256,256,0,r.stream>>>(static_cast<const char *>(r.staging.p),static_cast<__half *>(dest.p),count,describe(t));
}
void gemm(runtime & r, const void * w, const void * a, float * c, int m, int n, int k,
          long long sw=0, long long sa=0, int batch=1) {
    const float alpha=1,beta=0;
    check(cublasGemmStridedBatchedEx(r.blas,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
        w,CUDA_R_16F,k,sw,a,CUDA_R_16F,k,sa,&beta,c,CUDA_R_32F,n,static_cast<long long>(m)*n,
        batch,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
}
}
bool ggml_cuda_rtn_enabled() {
    const char * p=std::getenv("GGML_CUDA_RTN_A_BITS"); return p && (!std::strcmp(p,"16") || !std::strcmp(p,"4") || !std::strcmp(p,"8"));
}
int ggml_cuda_rtn_activation_bits() { const char * p=std::getenv("GGML_CUDA_RTN_A_BITS"); return p?std::atoi(p):0; }
bool ggml_cuda_rtn_supported(const ggml_tensor * op) {
    if (!ggml_cuda_rtn_enabled() || !op || op->op!=GGML_OP_MUL_MAT || op->type!=GGML_TYPE_F32 || !ggml_is_contiguous(op)) return false;
    const auto * w=op->src[0]; const auto * x=op->src[1];
    if (!w || !x || x->type!=GGML_TYPE_F32 || x->nb[0]!=sizeof(float) || w->ne[0]!=x->ne[0]) return false;
    for (int d=0;d<4;++d) if (w->ne[d]<=0 || x->ne[d]<=0 || w->ne[d]>INT_MAX || x->ne[d]>INT_MAX) return false;
    if (x->ne[2]%w->ne[2] || x->ne[3]%w->ne[3]) return false;
    bool attention=x->ne[2]>1 || x->ne[3]>1;
    if (attention) return (w->type==GGML_TYPE_F16 || w->type==GGML_TYPE_F32) && w->nb[0]==ggml_type_size(w->type);
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x)) return false;
    return w->type==GGML_TYPE_F16 || ((w->type==GGML_TYPE_Q4_0 || w->type==GGML_TYPE_Q8_0) && w->ne[0]%32==0)
        || (hp1_weight(w->type) && ggml_cuda_rtn_activation_bits()!=16 && w->ne[0]%32==0)
        || (w->type==GGML_TYPE_Q6_K && w->ne[0]%256==0);
}
bool ggml_cuda_rtn_compute(ggml_tensor * op) {
    error.clear(); auto & r=state(); std::lock_guard<std::mutex> lock(r.mutex);
    try {
        if (!ggml_cuda_rtn_supported(op) || !op->data || !op->src[0]->data || !op->src[1]->data) throw std::runtime_error("Unsupported CUDA RTN tensor");
        r.init(); const auto * w=op->src[0]; const auto * x=op->src[1];
        int m=x->ne[1], n=w->ne[1], k=w->ne[0], bits=ggml_cuda_rtn_activation_bits();
        r.output.reserve(ggml_nbytes(op));
        bool attention=x->ne[2]>1 || x->ne[3]>1;
        if (attention) {
            pack(r,w,r.weights); pack(r,x,r.activation);
            int repeat=x->ne[2]/w->ne[2];
            for (int b=0;b<x->ne[3];++b) for (int h=0;h<w->ne[2];++h) {
                size_t wh=(size_t(b/(x->ne[3]/w->ne[3]))*w->ne[2]+h)*n*k;
                size_t xh=(size_t(b)*x->ne[2]+h*repeat)*m;
                gemm(r,static_cast<const __half *>(r.weights.p)+wh,static_cast<const __half *>(r.activation.p)+xh*k,
                     static_cast<float *>(r.output.p)+xh*n,m,n,k,0,static_cast<long long>(m)*k,repeat);
            }
        } else if (bits==16 || w->type==GGML_TYPE_F16) {
            const void * weights=model_weight(r,w,true); pack(r,x,r.activation);
            gemm(r,weights,r.activation.p,static_cast<float *>(r.output.p),m,n,k);
        } else {
            const void * weights=model_weight(r,w,false);
            size_t count=size_t(m)*k, groups=count/32;
            r.input.reserve(count*sizeof(float)); r.activation.reserve(count); r.scales.reserve(groups*sizeof(float));
            check(cudaMemcpyAsync(r.input.p,x->data,count*sizeof(float),cudaMemcpyHostToDevice,r.stream));
            quantize_activation<<<(groups+7)/8,256,0,r.stream>>>(static_cast<const float *>(r.input.p),static_cast<int8_t *>(r.activation.p),static_cast<float *>(r.scales.p),groups,bits);
            block_matmul<<<dim3((n+31)/32,(m+7)/8),dim3(32,8),0,r.stream>>>(static_cast<const int8_t *>(r.activation.p),static_cast<const float *>(r.scales.p),static_cast<const uint8_t *>(weights),static_cast<float *>(r.output.p),m,n,k,weight_bits(w->type),hp1_weight(w->type));
        }
        check(cudaGetLastError());
        check(cudaMemcpyAsync(op->data,r.output.p,ggml_nbytes(op),cudaMemcpyDeviceToHost,r.stream));
        check(cudaStreamSynchronize(r.stream));
        if (attention) ++attention_calls; else ++dense_calls;
        return true;
    } catch (const std::exception & e) { if (r.stream) cudaStreamSynchronize(r.stream); error=e.what(); return false; }
}
const char * ggml_cuda_rtn_error() { return error.c_str(); }
uint64_t ggml_cuda_rtn_dense_calls() { return dense_calls.load(); }
uint64_t ggml_cuda_rtn_attention_calls() { return attention_calls.load(); }
void ggml_cuda_rtn_clear() { auto & r=state(); std::lock_guard<std::mutex> lock(r.mutex); if (r.stream) check(cudaStreamSynchronize(r.stream)); r.cache.clear(); }
