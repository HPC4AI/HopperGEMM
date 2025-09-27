#pragma once

#include "../utils.cuh"

namespace mat1 {

__device__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int n>
__device__ void wgmma_wait() {
    static_assert(n >= 0 && n <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(n) : "memory");
}

template <int width>
__device__ void wgmma(float c[2][16], bf16 *sa, bf16 *sb) {
    uint64_t a_desc = matrix_descriptor_format<width>(sa);
    uint64_t b_desc = matrix_descriptor_format<width>(sb);
    asm volatile(
        " { "
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        " {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "  %8,  %9,  %10, %11, %12, %13, %14, %15, "
        "  %16, %17, %18, %19, %20, %21, %22, %23, "
        "  %24, %25, %26, %27, %28, %29, %30, %31}, "
        "  %32, %33, "
        "  %34, %35, %36, "
        "  %37, %38; \n "
        " } "
        : "+f"(c[0][0]),  "+f"(c[0][1]),  "+f"(c[1][0]),  "+f"(c[1][1]),  "+f"(c[0][2]),  "+f"(c[0][3]),  "+f"(c[1][2]),  "+f"(c[1][3]),
          "+f"(c[0][4]),  "+f"(c[0][5]),  "+f"(c[1][4]),  "+f"(c[1][5]),  "+f"(c[0][6]),  "+f"(c[0][7]),  "+f"(c[1][6]),  "+f"(c[1][7]),
          "+f"(c[0][8]),  "+f"(c[0][9]),  "+f"(c[1][8]),  "+f"(c[1][9]),  "+f"(c[0][10]), "+f"(c[0][11]), "+f"(c[1][10]), "+f"(c[1][11]),
          "+f"(c[0][12]), "+f"(c[0][13]), "+f"(c[1][12]), "+f"(c[1][13]), "+f"(c[0][14]), "+f"(c[0][15]), "+f"(c[1][14]), "+f"(c[1][15])
        : "l"(a_desc), "l"(b_desc), 
          "n"(int32_t(1)), "n"(int32_t(1)), "n"(int32_t(1)),
          "n"(int32_t(0)), "n"(int32_t(0))
    );
}

template <size_t tile_m, size_t tile_n, size_t tile_k>
__global__ void kernel(int M, int N, int K, 
    float *c, const __grid_constant__ CUtensorMap a, const __grid_constant__ CUtensorMap b) {
    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;

    /////// create barrier ///////
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar_a;
    __shared__ barrier bar_b;
    if (threadIdx.x == 0) {
        init(&bar_a, blockDim.x);
        init(&bar_b, blockDim.x);
        cde::fence_proxy_async_shared_cta(); // ????
    }
    __syncthreads();

    /////// lgsts ///////
    __shared__ __align__(128) bf16 smem_a[tile_m * tile_k];
    __shared__ __align__(128) bf16 smem_b[tile_n * tile_k];
    barrier::arrival_token token_a;
    barrier::arrival_token token_b;

    float frag_c[2][16] = {0};
    static_assert(sizeof(frag_c) * 128 == tile_m * tile_n * sizeof(float));
    for(int iter = 0; iter < K / tile_k; iter ++) {
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(smem_a, &a, iter*tile_k, tile_m_id*tile_m, bar_a); // be careful, k first then m
            cde::cp_async_bulk_tensor_2d_global_to_shared(smem_b, &b, iter*tile_k, tile_n_id*tile_n, bar_b); // be careful, k first then n
            token_a = cuda::device::barrier_arrive_tx(bar_a, 1, sizeof(smem_a));
            token_b = cuda::device::barrier_arrive_tx(bar_b, 1, sizeof(smem_b));
        } else {
            token_a = bar_a.arrive();
            token_b = bar_b.arrive();
        }
        bar_a.wait(std::move(token_a));
        bar_b.wait(std::move(token_b));
        __syncthreads(); // ????

        /////// wgmma ///////
        wgmma_fence();
        wgmma<tile_k>(frag_c, smem_a, smem_b);
        wgmma<tile_k>(frag_c, smem_a+16, smem_b+16);
        wgmma<tile_k>(frag_c, smem_a+32, smem_b+32);
        wgmma<tile_k>(frag_c, smem_a+48, smem_b+48);
        wgmma_commit();
        wgmma_wait<0>();
    }

    /////// stg ///////
    size_t warp = threadIdx.x / 32,
        lane = threadIdx.x % 32,
        row_offset = lane / 4,
        col_offset = lane % 4 * 2;
    float* c_ptr = c + 
            tile_m_id * tile_m * N + 
            tile_n_id * tile_n;
    for(size_t rt = 0; rt < 2; rt ++) {
        for(size_t ct = 0; ct < 8; ct ++) {
            size_t i = warp * 16 + rt * 8 + row_offset;
            size_t j =             ct * 8 + col_offset;
            c_ptr[i * N + j + 0] = frag_c[rt][ct * 2 + 0];
            c_ptr[i * N + j + 1] = frag_c[rt][ct * 2 + 1];
        }
    }
}

void runKernel1(size_t M, size_t N, size_t K, bf16 *A, bf16 *B, float *C) {
    constexpr size_t tile_m = 64;
    constexpr size_t tile_n = 64;
    constexpr size_t tile_k = 64;

    CUtensorMap tensor_a, tensor_b;
    createTensorMap(A, M, K, tile_m, tile_k, &tensor_a);
    createTensorMap(B, N, K, tile_n, tile_k, &tensor_b);

    dim3 block(128);
    dim3 grid(N / tile_n, M / tile_m);

    kernel<tile_m, tile_n, tile_k><<<grid, block>>>(M, N, K, C, tensor_a, tensor_b);
}

}

using mat1::runKernel1;