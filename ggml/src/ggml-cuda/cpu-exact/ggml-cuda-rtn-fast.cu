// Reuse the frozen activation, attention, support, identity and counter contracts.
#define ggml_cuda_rtn_compute ggml_cuda_rtn_original_compute
#define ggml_cuda_rtn_clear ggml_cuda_rtn_original_clear
#include "ggml-cuda-rtn.cu"
#undef ggml_cuda_rtn_compute
#undef ggml_cuda_rtn_clear
#include <unordered_set>

namespace {
std::unordered_set<const void *> expanded;
size_t expansion_bytes=0;
constexpr size_t expansion_limit=size_t(512)<<20;

__global__ void expand_weights(const uint8_t * source, int * packed, size_t blocks, int n, int bits, bool mxint) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=blocks*size_t(n)*8) return;
    int col=i%n, group=(i/n)%8;
    size_t b=i/(size_t(n)*8), count=blocks*n;
    int stride=mxint?(bits==4?24:40):(bits==4?18:34);
    const uint8_t * wb=bits==6?source+(size_t(col)*(blocks/8)+b/8)*210:source+(size_t(col)*blocks+b)*stride;
    unsigned value=0;
    for (int j=0;j<4;++j) {
        int pos=group*4+j, offset=mxint?0:2;
        int q=bits==6?q6_code(wb,(b%8)*32+pos):bits==4?int((wb[offset+pos%16]>>(pos<16?0:4))&15)-8:int(static_cast<int8_t>(wb[offset+pos]));
        value|=unsigned(uint8_t(q))<<(j*8);
    }
    packed[i]=int(value);
    if (group) return;
    float scale;
    if (mxint) {
        int offset=bits==4?16:32;
        int exponent=*reinterpret_cast<const int16_t *>(wb+offset);
        float channel=*reinterpret_cast<const float *>(wb+offset+4);
        scale=exponent==INT16_MIN?0.0f:ldexpf(channel,exponent);
    } else scale=__half2float(*reinterpret_cast<const __half *>(wb+(bits==6?208:0)));
    reinterpret_cast<float *>(packed+count*8)[b*n+col]=scale;
    packed[count*9+b*n+col]=bits==6?int(unsigned(wb[192+(b%8)*2])|(unsigned(wb[193+(b%8)*2])<<8)):0;
}

const void * cached_weights(runtime & r, const ggml_tensor * w) {
    auto it=r.cache.find(w->data);
    if (it!=r.cache.end()) {
        if (it->second->type!=w->type || it->second->k!=w->ne[0] || it->second->n!=w->ne[1] || it->second->half)
            throw std::runtime_error("Cached weight identity changed; clear CUDA RTN cache");
        return it->second->data.p;
    }
    size_t count=size_t(w->ne[0]/32)*size_t(w->ne[1]);
    if (count>SIZE_MAX/40) throw std::runtime_error("CUDA RTN expanded weight size overflow");
    size_t bytes=count*40, original=ggml_nbytes(w), extra=bytes>original?bytes-original:0;
    size_t available,total; check(cudaMemGetInfo(&available,&total));
    // Bound expansion on 8 GiB cards; untouched tensors use the original packed kernel.
    if (extra>expansion_limit-expansion_bytes || bytes>available || original>available-bytes || available-bytes-original<(size_t(256)<<20))
        return model_weight(r,w,false);
    auto item=std::make_unique<weight>();
    item->type=w->type; item->k=w->ne[0]; item->n=w->ne[1]; item->half=false;
    buffer compressed; compressed.reserve(original); item->data.reserve(bytes);
    check(cudaMemcpyAsync(compressed.p,w->data,original,cudaMemcpyHostToDevice,r.stream));
    expand_weights<<<(count*8+255)/256,256,0,r.stream>>>(static_cast<const uint8_t *>(compressed.p),static_cast<int *>(item->data.p),w->ne[0]/32,w->ne[1],weight_bits(w->type),mxint_weight(w->type));
    check(cudaGetLastError()); check(cudaStreamSynchronize(r.stream));
    const void * result=item->data.p;
    r.cache.emplace(w->data,std::move(item)); expanded.insert(w->data); expansion_bytes+=extra;
    return result;
}

__global__ void register_cached_matmul(const int8_t * a,const float * scales,const int * w,float * output,int m,int n,int k,bool q6) {
    __shared__ int av[16][8],wv[8][32],subscale[32];
    __shared__ float as[16],ws[32];
    const int t=threadIdx.y*32+threadIdx.x,row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*32+threadIdx.x,blocks=k/32;
    const size_t count=size_t(blocks)*n;
    double sum0=0,sum1=0;
    for (int b=0;b<blocks;++b) {
        if (t<128) {
            const int rr=blockIdx.y*16+t/8,g=t%8;
            av[t/8][g]=rr<m?reinterpret_cast<const int *>(a)[(size_t(rr)*blocks+b)*8+g]:0;
            if (!g) as[t/8]=rr<m?scales[size_t(rr)*blocks+b]:0;
        }
        wv[threadIdx.y][threadIdx.x]=col<n?w[(size_t(b)*8+threadIdx.y)*n+col]:0;
        if (!threadIdx.y) {
            ws[threadIdx.x]=col<n?reinterpret_cast<const float *>(w+count*8)[size_t(b)*n+col]:0;
            subscale[threadIdx.x]=q6 && col<n?w[count*9+size_t(b)*n+col]:0;
        }
        __syncthreads();
        int lo0=0,hi0=0,lo1=0,hi1=0;
#pragma unroll
        for (int g=0;g<4;++g) {
            const int weight=wv[g][threadIdx.x];
            lo0=__dp4a(av[threadIdx.y][g],weight,lo0); lo1=__dp4a(av[threadIdx.y+8][g],weight,lo1);
        }
#pragma unroll
        for (int g=4;g<8;++g) {
            const int weight=wv[g][threadIdx.x];
            hi0=__dp4a(av[threadIdx.y][g],weight,hi0); hi1=__dp4a(av[threadIdx.y+8][g],weight,hi1);
        }
        const int sub=subscale[threadIdx.x],s0=int(static_cast<int8_t>(sub&255)),s1=int(static_cast<int8_t>((sub>>8)&255));
        const int dot0=q6?lo0*s0+hi0*s1:lo0+hi0,dot1=q6?lo1*s0+hi1*s1:lo1+hi1;
        sum0=__dadd_rn(sum0,__dmul_rn(double(dot0),__dmul_rn(double(ws[threadIdx.x]),double(as[threadIdx.y]))));
        sum1=__dadd_rn(sum1,__dmul_rn(double(dot1),__dmul_rn(double(ws[threadIdx.x]),double(as[threadIdx.y+8]))));
        __syncthreads();
    }
    if (row<m && col<n) output[size_t(row)*n+col]=float(sum0);
    if (row+8<m && col<n) output[size_t(row+8)*n+col]=float(sum1);
}

__global__ void cached_matmul(const int8_t * a, const float * scales, const int * w, float * output, int m, int n, int k, bool q6) {
    __shared__ int av[8][8], wv[8][32], subscale[32];
    __shared__ float as[8], ws[32];
    int t=threadIdx.y*32+threadIdx.x, row=blockIdx.y*8+threadIdx.y, col=blockIdx.x*32+threadIdx.x;
    int blocks=k/32;
    size_t count=size_t(blocks)*n;
    double sum=0.0;
    for (int b=0;b<blocks;++b) {
        if (t<64) {
            int rr=blockIdx.y*8+t/8, group=t%8;
            av[t/8][group]=rr<m?reinterpret_cast<const int *>(a)[(size_t(rr)*blocks+b)*8+group]:0;
            if (!group) as[t/8]=rr<m?scales[size_t(rr)*blocks+b]:0;
        }
        wv[threadIdx.y][threadIdx.x]=col<n?w[(size_t(b)*8+threadIdx.y)*n+col]:0;
        if (!threadIdx.y) {
            ws[threadIdx.x]=col<n?reinterpret_cast<const float *>(w+count*8)[size_t(b)*n+col]:0;
            subscale[threadIdx.x]=q6 && col<n?w[count*9+size_t(b)*n+col]:0;
        }
        __syncthreads();
        int dot0=0,dot1=0;
#pragma unroll
        for (int g=0;g<4;++g) dot0=__dp4a(av[threadIdx.y][g],wv[g][threadIdx.x],dot0);
#pragma unroll
        for (int g=4;g<8;++g) dot1=__dp4a(av[threadIdx.y][g],wv[g][threadIdx.x],dot1);
        int sub=subscale[threadIdx.x];
        int dot=q6?dot0*int(static_cast<int8_t>(sub&255))+dot1*int(static_cast<int8_t>((sub>>8)&255)):dot0+dot1;
        sum=__dadd_rn(sum,__dmul_rn(double(dot),__dmul_rn(double(ws[threadIdx.x]),double(as[threadIdx.y]))));
        __syncthreads();
    }
    if (row<m && col<n) output[size_t(row)*n+col]=float(sum);
}
}

GGML_API bool ggml_cuda_rtn_compute(ggml_tensor * op) {
    if (!ggml_cuda_rtn_supported(op) || op->src[1]->ne[2]>1 || op->src[1]->ne[3]>1 || op->src[0]->type==GGML_TYPE_F16 || (ggml_cuda_rtn_activation_bits()==16 && !mxint_weight(op->src[0]->type)))
        return ggml_cuda_rtn_original_compute(op);
    error.clear(); auto & r=state(); std::lock_guard<std::mutex> lock(r.mutex);
    try {
        if (!op->data || !op->src[0]->data || !op->src[1]->data) throw std::runtime_error("Unsupported CUDA RTN tensor");
        r.init(); const auto * w=op->src[0]; const auto * x=op->src[1];
        int m=x->ne[1], n=w->ne[1], k=w->ne[0], bits=ggml_cuda_rtn_activation_bits();
        if (mxint_weight(w->type) && bits==16) bits=weight_bits(w->type);
        r.output.reserve(ggml_nbytes(op));
        const void * weights=cached_weights(r,w);
        size_t count=size_t(m)*k, groups=count/32;
        r.input.reserve(count*sizeof(float)); r.activation.reserve(count); r.scales.reserve(groups*sizeof(float));
        check(cudaMemcpyAsync(r.input.p,x->data,count*sizeof(float),cudaMemcpyHostToDevice,r.stream));
        quantize_activation<<<(groups+7)/8,256,0,r.stream>>>(static_cast<const float *>(r.input.p),static_cast<int8_t *>(r.activation.p),static_cast<float *>(r.scales.p),groups,bits,mxint_weight(w->type));
        if (expanded.count(w->data)) {
            if (m>=128 && n>=512 && (w->type==GGML_TYPE_Q4_0 || w->type==GGML_TYPE_Q8_0))
                register_cached_matmul<<<dim3((n+31)/32,(m+15)/16),dim3(32,8),0,r.stream>>>(static_cast<const int8_t *>(r.activation.p),static_cast<const float *>(r.scales.p),static_cast<const int *>(weights),static_cast<float *>(r.output.p),m,n,k,w->type==GGML_TYPE_Q6_K);
            else
                cached_matmul<<<dim3((n+31)/32,(m+7)/8),dim3(32,8),0,r.stream>>>(static_cast<const int8_t *>(r.activation.p),static_cast<const float *>(r.scales.p),static_cast<const int *>(weights),static_cast<float *>(r.output.p),m,n,k,w->type==GGML_TYPE_Q6_K);
        }
        else
            block_matmul<<<dim3((n+31)/32,(m+7)/8),dim3(32,8),0,r.stream>>>(static_cast<const int8_t *>(r.activation.p),static_cast<const float *>(r.scales.p),static_cast<const uint8_t *>(weights),static_cast<float *>(r.output.p),m,n,k,weight_bits(w->type),mxint_weight(w->type));
        check(cudaGetLastError());
        check(cudaMemcpyAsync(op->data,r.output.p,ggml_nbytes(op),cudaMemcpyDeviceToHost,r.stream));
        check(cudaStreamSynchronize(r.stream)); ++dense_calls; return true;
    } catch (const std::exception & e) { if (r.stream) cudaStreamSynchronize(r.stream); error=e.what(); return false; }
}
GGML_API void ggml_cuda_rtn_clear() {
    auto & r=state(); std::lock_guard<std::mutex> lock(r.mutex);
    if (r.stream) check(cudaStreamSynchronize(r.stream));
    r.cache.clear(); expanded.clear(); expansion_bytes=0;
}
