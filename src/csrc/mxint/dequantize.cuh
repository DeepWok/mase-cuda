#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "torch/types.h"
#include <cassert>
#include <cstdint>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>
#include <cutlass/bfloat16.h>
#include <set>
#include <thrust/host_vector.h>
#include <torch/extension.h>
#pragma once
namespace mase_cuda {
namespace mxint8 {
namespace dequantize {
template <class TypeX,                              // input type
          class ShapeX,                             // input/output shape
          class StrideX,                            // input/output stride
          class TypeScale,                          // scale type
          class ScaleShape,                         // scale shape
          class StrideScale,                        // scale stride
          class GroupTiler,                         // group size
          class CtaTiler,                           // CTA tiler, (BLK_M, BLK_K)
          class SmemLayoutX, class SmemLayoutScale, // shared mem layout
          class ThreadLayoutX>
__global__ static void dequantize1d_device(TypeX const *x, ShapeX shape_x, StrideX stride_x, // input
                                           TypeScale const *scales, ScaleShape shape_scale, StrideScale stride_scale,
                                           GroupTiler group_tiler, // scale
                                           CtaTiler cta_tiler, SmemLayoutX layout_sX, SmemLayoutScale layout_sScale,
                                           ThreadLayoutX layout_tX, // cuda
                                           cutlass::bfloat16_t *y   // output
) {
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(shape_x) == Int<1>{});     // 1D tensor
    CUTE_STATIC_ASSERT_V(rank(shape_scale) == Int<1>{}); // 1D tensor
    CUTE_STATIC_ASSERT_V(rank(group_tiler) == Int<1>{}); // 1D tensor
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<2>{});   // (BLK_M, BLK_K)

    static_assert(is_static<ThreadLayoutX>::value);
    static_assert(is_static<SmemLayoutX>::value);
    static_assert(is_static<SmemLayoutScale>::value);

    // assert(size<0>(shape_x) % size<0>(group_tiler) == 0);
    // assert(size<0>(shape_x) / size<0>(group_tiler) == size<0>(shape_scale));

    // represent full tensor and scale in global memory
    // input tensor, shape: (M)
    uint8_t const *x_raw_uint8 = reinterpret_cast<uint8_t const *>(x);
    uint8_t const *scales_raw_uint8 = reinterpret_cast<uint8_t const *>(scales);
    Tensor mX_raw = make_tensor(make_gmem_ptr(x_raw_uint8), shape_x, stride_x);
    // divide by group size, (_group_size, num_groups): (_1, _group_size)
    Tensor mX = flatten(flat_divide(mX_raw, group_tiler));
    // scale tensor in global memory, shape: (num_groups): (1)
    Tensor vScale = make_tensor(make_gmem_ptr(scales_raw_uint8), shape_scale, stride_scale);
    Tensor mY_raw = make_tensor(make_gmem_ptr(y), shape_x, stride_x); // (M)
    // (_group_size, num_groups): (_1, _group_size)
    Tensor mY = flatten(flat_divide(mY_raw, group_tiler));

    // get the appropriate block for this thread block
    auto cta_coord = make_coord(blockIdx.x, _);
    // gX: (BLK_M, BLK_K, num_blks_k), this num_blks_k is for temporal iteration
    Tensor gX = local_tile(mX, cta_tiler, cta_coord);
    // gScale: (BLK_N, num_blks_k)
    Tensor gScale = local_tile(vScale, select<1>(cta_tiler), select<1>(cta_coord));
    // gY: (BLK_M, BLK_K, num_blks_k)
    Tensor gY = local_tile(mY, cta_tiler, cta_coord);

    // create shared memory for input and scale
    __shared__ uint8_t smemX[size(cta_tiler)];                                   // (BLK_M, BLK_K)
    __shared__ uint8_t smemScale[size(select<1>(cta_tiler))];                    // (BLK_K)
    Tensor sX = make_tensor(make_smem_ptr(smemX), cta_tiler);                    // (BLK_M, BLK_K)
    Tensor sScale = make_tensor(make_smem_ptr(smemScale), select<1>(cta_tiler)); // (BLK_K)

    // partition copy
    Tensor tXgX = local_partition(gX, layout_tX, threadIdx.x); // (thd_m, thd_k, num_blks_k), thd_k = 1
    Tensor tXsX = local_partition(sX, layout_tX, threadIdx.x); // (thd_m, thd_k)

    Tensor tXgY = local_partition(gY, layout_tX, threadIdx.x); // (thd_m, thd_k, num_blks_k), bf16
    Tensor tXrY = make_tensor_like(tXgY(_, _, 0));             // (thd_m, thd_k), bf16

    auto K_TILE_MAX = size<2>(tXgX);
    clear(tXrY);
    // iterate over num_blks_k
    for (int k = 0; k < K_TILE_MAX; ++k) {
        // copy from global to shared
        Tensor tXgXk = tXgX(_, _, k);
        Tensor gScalek = gScale(_, k);

        // copy Scale
        if (threadIdx.x % size<0>(layout_tX) == 0) {
            sScale[threadIdx.x / size<0>(layout_tX)] = gScalek[threadIdx.x / size<0>(layout_tX)];
        }

        CUTE_UNROLL
        for (int i = 0; i < size(tXsX); ++i) {
            tXsX[i] = tXgXk[i];
        }
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        CUTE_UNROLL
        for (int i = 0; i < size(tXrY); ++i) {
            auto scaleIdx = threadIdx.x / size<0>(layout_tX);
            auto sign = static_cast<uint16_t>(tXsX[i] & 0x80) << 8;
            auto exp = static_cast<uint16_t>(sScale[scaleIdx]) << 7;
            auto mantissa = static_cast<uint16_t>((tXsX[i] << 1) & 0x7E);
            tXrY[i] = cutlass::bfloat16_t::bitcast(sign | exp | mantissa);
        }
        // copy the dequantized value to global memory
        Tensor tXgYk = tXgY(_, _, k);
        CUTE_UNROLL
        for (int i = 0; i < size(tXrY); ++i) {
            tXgYk[i] = tXrY[i];
        }
    }
}

template <class TypeX, class TypeScale>
__host__ void dequantize1d_host(TypeX const *x, const int M, TypeScale const *scales, const int group_size,
                                cutlass::bfloat16_t *y) {
    assert(M % group_size == 0);

    const int num_groups = M / group_size;
    uint8_t const *x_raw_uint8 = reinterpret_cast<uint8_t const *>(x);
    uint8_t const *scales_raw_uint8 = reinterpret_cast<uint8_t const *>(scales);
    thrust::host_vector<uint8_t> hX(x_raw_uint8, x_raw_uint8 + M);
    thrust::host_vector<uint8_t> hScales(scales_raw_uint8, scales_raw_uint8 + num_groups);

    for (int i = 0; i < M; ++i) {
        auto sign = static_cast<uint16_t>(hX[i] & 0x80) << 8;
        auto exp = static_cast<uint16_t>(hScales[i / group_size]) << 7;
        auto mantissa = static_cast<uint16_t>((hX[i] << 1) & 0x7E);
        y[i] = cutlass::bfloat16_t::bitcast(sign | exp | mantissa);
    }
}

torch::Tensor dequantize1d(torch::Tensor x, torch::Tensor scales, const int group_size) {
    using namespace cute;
    const int available_group_sizes[] = {8, 16, 32, 64, 128, 256, 512};
    std::set<int> group_sizes(available_group_sizes, available_group_sizes + 7);
    if (group_sizes.find(group_size) == group_sizes.end()) {
        throw std::invalid_argument("group_size not supported, must be one of {8, 16, 32, 64, 128, 256, 512}");
    }
    // infer number of elements in x
    const int m = x.numel() * x.itemsize() / sizeof(torch::kUInt8);
    const int num_groups = m / group_size;
    // check arguments
    if (m % group_size != 0) {
        throw std::invalid_argument("m %% group_size != 0");
    }
    if (num_groups != scales.numel() * scales.itemsize() / sizeof(torch::kUInt8)) {
        throw std::invalid_argument("m / group_size != num scale elements");
    }
    if (x.device() != scales.device()) {
        throw std::invalid_argument("x.device() != scales.device()");
    }

    auto y_options = torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16);
    auto y = torch::empty({m}, y_options);

    auto _x = x.contiguous();
    auto _scales = scales.contiguous();
    // auto _y = y.contiguous();

    auto x_ptr = _x.const_data_ptr();
    auto scales_ptr = _scales.const_data_ptr();
    cutlass::bfloat16_t *y_ptr = reinterpret_cast<cutlass::bfloat16_t *>(y.data_ptr());

    if (_x.device().is_cpu()) {
        dequantize1d_host(x_ptr, m, scales_ptr, group_size, y_ptr);
    } else if (_x.device().is_cuda()) {
        // determining BLK_M, BLK_K
        // 32 threads per warp, warp load size: 32, 64, 128 bytes
        auto shape_x = make_shape(m);
        auto stride_x = make_stride(Int<1>{});
        auto shape_scale = make_shape(num_groups);
        auto stride_scale = make_stride(Int<1>{});
        auto group_tiler = make_shape(group_size);

        if (group_size == 8) {
            auto BLK_M = Int<8>{};
            auto BLK_K = Int<128>{};
            auto THD_M = Int<8>{};
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
            dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                             stride_scale, group_tiler, cta_tiler, layout_sX,
                                                             layout_sScale, layout_tX, y_ptr);
        } else if (group_size == 16) {
            auto BLK_M = Int<16>{};
            auto BLK_K = Int<64>{};
            auto THD_M = Int<16>{};
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
            dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                             stride_scale, group_tiler, cta_tiler, layout_sX,
                                                             layout_sScale, layout_tX, y_ptr);
        } else if (group_size == 32) {
            auto BLK_M = Int<32>{};
            auto BLK_K = Int<32>{};
            auto THD_M = Int<32>{};
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
            dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                             stride_scale, group_tiler, cta_tiler, layout_sX,
                                                             layout_sScale, layout_tX, y_ptr);
        } else if (group_size == 64) {
            auto BLK_M = Int<64>{};
            auto BLK_K = Int<16>{};
            auto THD_M = Int<64>{};
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
            dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                             stride_scale, group_tiler, cta_tiler, layout_sX,
                                                             layout_sScale, layout_tX, y_ptr);
        } else if (group_size == 128) {
            auto BLK_M = Int<128>{};
            auto BLK_K = Int<8>{};
            auto THD_M = Int<64>{};
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
            dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                             stride_scale, group_tiler, cta_tiler, layout_sX,
                                                             layout_sScale, layout_tX, y_ptr);

        } else if (group_size == 256) {
            // group_size == 256
            auto BLK_M = Int<256>{};
            auto BLK_K = Int<4>{};
            auto THD_M = Int<128>{};
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
            dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                             stride_scale, group_tiler, cta_tiler, layout_sX,
                                                             layout_sScale, layout_tX, y_ptr);
        } else {
            // group_size == 512
            auto BLK_M = Int<256>{};
            auto BLK_K = Int<4>{};
            auto THD_M = Int<128>{};
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
            dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                             stride_scale, group_tiler, cta_tiler, layout_sX,
                                                             layout_sScale, layout_tX, y_ptr);
        }
    } else {
        throw std::invalid_argument("x.device() not supported");
    }

    return y;
}
} // namespace dequantize
} // namespace mxint8
} // namespace mase_cuda