// gemmini_tensor/tensor_cache_key.h
#pragma once

#include "ggml.h"

//  TensorCacheKey 구조체
// 캐시 키에 transpose와 acc 정보 포함
// 같은 텐서 포인터라도 설정이 다르면 별도로 캐싱

struct TensorCacheKey {
    const ggml_tensor* ptr;
    bool transpose;
    bool acc;
    
    // map의 키로 사용하기 위한 비교 연산자
    bool operator<(const TensorCacheKey& other) const {
        if (ptr != other.ptr) return ptr < other.ptr;
        if (transpose != other.transpose) return transpose < other.transpose;
        return acc < other.acc;
    }
    
    // 디버깅용 equality 연산자
    bool operator==(const TensorCacheKey& other) const {
        return ptr == other.ptr && transpose == other.transpose && acc == other.acc;
    }
};