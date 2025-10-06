// gemmini_tensor/transient_key.h
#pragma once
#include <cstdint>

// Activation/Output 버퍼 풀용 키
// dimension만 비교 (값은 매번 업데이트되므로)
struct TransientKey
{
    int64_t rows;
    int64_t cols;

    bool operator<(const TransientKey &other) const
    {
        if (rows != other.rows)
            return rows < other.rows;
        return cols < other.cols;
    }

    bool operator==(const TransientKey &other) const
    {
        return rows == other.rows && cols == other.cols;
    }
};