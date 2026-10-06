#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "test-metal-soft-f64-oracle.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>

#ifndef GGML_METAL_SOFT_F64_HEADER_PATH
#error GGML_METAL_SOFT_F64_HEADER_PATH must identify the software binary64 shader header
#endif

static int run_test() {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) { puts("SKIP: no Metal device available"); return 77; }
    NSError *error=nil;
    NSString *headerPath=[NSString stringWithUTF8String:GGML_METAL_SOFT_F64_HEADER_PATH];
    NSString *header=[NSString stringWithContentsOfFile:headerPath encoding:NSUTF8StringEncoding error:&error];
    if(!header) { fprintf(stderr,"HEADER: %s\n",error.description.UTF8String); return 2; }
    NSString *kernel=@R"MSL(
struct Input { long i; ulong a,b; uint s,t,ma,mb; };
struct Output { ulong integer,scale,product,sum; uint result,merge; };
kernel void check_f64(device const Input* input [[buffer(0)]], device Output* output [[buffer(1)]], uint n [[thread_position_in_grid]]) {
    Input x=input[n]; Output y;
    y.integer=mq_f64_from_i64(x.i); y.scale=mq_f64_from_f32_bits(x.s);
    y.product=mq_f64_mul(x.a,x.b); y.sum=mq_f64_add(x.a,x.b);
    y.result=mq_f64_to_f32_bits(mq_f64_mul(mq_f64_mul(y.integer,y.scale),mq_f64_from_f32_bits(x.t)));
    y.merge=mq_f32_add_bits(x.ma,x.mb);
    output[n]=y;
}
)MSL";
    MTLCompileOptions *options=[MTLCompileOptions new];
    options.languageVersion=MTLLanguageVersion3_1;
    options.mathMode=MTLMathModeSafe; options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
    id<MTLLibrary> library=[device newLibraryWithSource:[header stringByAppendingString:kernel] options:options error:&error];
    if(!library) { fprintf(stderr,"COMPILE: %s\n",error.description.UTF8String); return 3; }
    id<MTLComputePipelineState> pipe=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"check_f64"] error:&error];
    if(!pipe) { fprintf(stderr,"PIPELINE: %s\n",error.description.UTF8String); return 4; }
    constexpr size_t count=262144;
    std::vector<F64Input> input(count);
    std::vector<F64Output> expected(count);
    uint64_t rng=0x6a17f82376542110ULL;
    auto next=[&]() { rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng; };
    auto finite64=[&]() { uint64_t v=next(); if((v&0x7ff0000000000000ULL)==0x7ff0000000000000ULL) v^=1ULL<<52; return v; };
    auto finite32=[&]() { uint32_t v=uint32_t(next()); if((v&0x7f800000U)==0x7f800000U) v^=1U<<23; return v; };
    // Correlated exponents and opposite signs exercise alignment and cancellation.
    for(size_t n=0;n<count;n++) {
        auto &x=input[n]; x.i=int64_t(next()); x.a=finite64(); x.b=finite64();
        x.s=finite32(); x.t=finite32(); x.ma=finite32(); x.mb=finite32();
        if(n%4==0) x.i=int32_t(x.i);
        if(n%3==0) {
            int eb=int((x.a>>52)&2047)-int(next()%65);
            if(eb<0) eb=0;
            x.b=(x.b&0x800fffffffffffffULL)|(uint64_t(eb)<<52);
        }
        if(n%11==0) x.b=x.a^0x8000000000000000ULL;
        if(n%13==0) x.b=(x.a^0x8000000000000000ULL)+(n%5)-2;
        if(n%17==0) x.a&=0x800fffffffffffffULL;
        if(n%19==0) x.s&=0x807fffffU;
        if(n%23==0) x.ma&=0x807fffffU;
        if(n%29==0) x.mb=x.ma^0x80000000U;
    }
    // Explicit zeros, halfway integer conversions, underflow, and overflow.
    const uint64_t d[]={0,0x8000000000000000ULL,1,0x8000000000000001ULL,0x000fffffffffffffULL,
        0x0010000000000000ULL,0x3fefffffffffffffULL,0x3ff0000000000000ULL,0x3ff0000000000001ULL,
        0x7fefffffffffffffULL,0xffefffffffffffffULL};
    const uint32_t f[]={0,0x80000000U,1,0x80000001U,0x007fffffU,0x00800000U,0x3f000000U,
        0x3f800000U,0x3f800001U,0x7f7fffffU,0xff7fffffU};
    size_t n=0;
    for(auto a:d) for(auto b:d) for(auto s:f) for(auto t:f) {
        input[n]={int64_t((1ULL<<53)+(n%9)-4),a,b,s,t,s,t}; ++n;
    }
    input[n++]={std::numeric_limits<int64_t>::min(),0,0,0x3f800000U,0x3f800000U,0,0};
    input[n++]={std::numeric_limits<int64_t>::max(),0,0,0x3f800000U,0x3f800000U,0,0};
    oracle_f64(input.data(),expected.data(),count);
    id<MTLBuffer> ib=[device newBufferWithBytes:input.data() length:sizeof(F64Input)*count options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob=[device newBufferWithLength:sizeof(F64Output)*count options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q=[device newCommandQueue];
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
    if (!ib || !ob || !q || !cb || !ce) {
        fputs("GPU: failed to allocate resources\n", stderr);
        return 5;
    }
    [ce setComputePipelineState:pipe];
    [ce setBuffer:ib offset:0 atIndex:0];
    [ce setBuffer:ob offset:0 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [ce endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted) { fprintf(stderr,"GPU: %s\n",cb.error.description.UTF8String); return 5; }
    const F64Output *actual=(const F64Output*)ob.contents;
    size_t mismatches[6]={}; const char *names[]={"from_i64","from_f32","mul","add","scale_to_f32","merge_f32"};
    for(size_t j=0;j<count;j++) {
        const uint64_t e[]={expected[j].integer,expected[j].scale,expected[j].product,expected[j].sum,expected[j].result,expected[j].merge};
        const uint64_t g[]={actual[j].integer,actual[j].scale,actual[j].product,actual[j].sum,actual[j].result,actual[j].merge};
        for(size_t k=0;k<6;k++) if(e[k]!=g[k]) {
            if(mismatches[k]<3) printf("MISMATCH %s index=%zu i=%016llx a=%016llx b=%016llx s=%08x t=%08x ma=%08x mb=%08x expected=%016llx actual=%016llx\n",names[k],j,(unsigned long long)input[j].i,(unsigned long long)input[j].a,(unsigned long long)input[j].b,input[j].s,input[j].t,input[j].ma,input[j].mb,(unsigned long long)e[k],(unsigned long long)g[k]);
            mismatches[k]++;
        }
    }
    size_t failures=0;
    printf("device=%s cases=%zu explicit_cases=%zu gpu_seconds=%.9f\n",device.name.UTF8String,count,n,cb.GPUEndTime-cb.GPUStartTime);
    for(size_t k=0;k<6;k++) { printf("%s mismatches=%zu/%zu\n",names[k],mismatches[k],count); failures+=mismatches[k]; }
    return failures?1:0;

}

int main() {
    @autoreleasepool {
        return run_test();
    }
}
