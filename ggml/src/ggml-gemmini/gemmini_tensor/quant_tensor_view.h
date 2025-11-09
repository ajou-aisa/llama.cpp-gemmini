// #pragma once

// #include <cstddef>
// #include <cstdint>

// namespace aisa {

// template <typename T>
// struct QuantTensorViewT {
//     T *data = nullptr;
//     size_t rows = 0;
//     size_t cols = 0;
//     size_t stride = 0; // elements between consecutive rows

//     bool empty() const noexcept {
//         return data == nullptr || rows == 0 || cols == 0;
//     }

//     T *row_ptr(size_t r) const noexcept {
//         return data + r * stride;
//     }
// };

// using QuantTensorView = QuantTensorViewT<int8_t>;
// using ConstQuantTensorView = QuantTensorViewT<const int8_t>;

// inline ConstQuantTensorView make_const_view(const QuantTensorView &view) noexcept {
//     return {view.data, view.rows, view.cols, view.stride};
// }

// } // namespace aisa

