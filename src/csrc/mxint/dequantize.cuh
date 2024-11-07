#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>
#include <cutlass/bfloat16.h>
#include <thrust/host_vector.h>

#pragma once
namespace mase_cuda {
namespace mxint8 {
namespace dequantize {
template <class ShapeX,                             // input/output shape
          class StrideX,                            // input/output stride
          class ScaleShape,                         // scale shape
          class StrideScale,                        // scale stride
          class GroupTiler,                         // group size
          class CtaTiler,                           // CTA tiler, (BLK_M, BLK_K)
          class SmemLayoutX, class SmemLayoutScale, // shared mem layout
          class ThreadLayoutX>
__global__ static void dequantize1d_device(uint8_t const *x, ShapeX shape_x, StrideX stride_x, // input
                                           uint8_t const *scales, ScaleShape shape_scale, StrideScale stride_scale,
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
    Tensor mX_raw = make_tensor(make_gmem_ptr(x), shape_x, stride_x);
    // divide by group size, (_group_size, num_groups): (_1, _group_size)
    Tensor mX = flatten(flat_divide(mX_raw, group_tiler));
    // scale tensor in global memory, shape: (num_groups): (1)
    Tensor vScale = make_tensor(make_gmem_ptr(scales), shape_scale, stride_scale);
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
} // namespace dequantize
} // namespace mxint8
} // namespace mase_cuda