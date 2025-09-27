#pragma once

#include "../utils.cuh"

namespace mat2 {

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
__device__ void wgmma(float c[2][32], bf16 *sa, bf16 *sb) {
    uint64_t a_desc = matrix_descriptor_format<width>(sa);
    uint64_t b_desc = matrix_descriptor_format<width>(sb);
    asm volatile(
        " { "
        " wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        " {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "  %8,  %9,  %10, %11, %12, %13, %14, %15, "
        "  %16, %17, %18, %19, %20, %21, %22, %23, "
        "  %24, %25, %26, %27, %28, %29, %30, %31, "
        "  %32, %33, %34, %35, %36, %37, %38, %39, "
        "  %40, %41, %42, %43, %44, %45, %46, %47, "
        "  %48, %49, %50, %51, %52, %53, %54, %55, "
        "  %56, %57, %58, %59, %60, %61, %62, %63}, "
        "  %64, %65, "
        "  %66, %67, %68, "
        "  %69, %70; \n "
        " } "
        : "+f"(c[0][0]),  "+f"(c[0][1]),  "+f"(c[1][0]),  "+f"(c[1][1]),  "+f"(c[0][2]),  "+f"(c[0][3]),  "+f"(c[1][2]),  "+f"(c[1][3]),
          "+f"(c[0][4]),  "+f"(c[0][5]),  "+f"(c[1][4]),  "+f"(c[1][5]),  "+f"(c[0][6]),  "+f"(c[0][7]),  "+f"(c[1][6]),  "+f"(c[1][7]),
          "+f"(c[0][8]),  "+f"(c[0][9]),  "+f"(c[1][8]),  "+f"(c[1][9]),  "+f"(c[0][10]), "+f"(c[0][11]), "+f"(c[1][10]), "+f"(c[1][11]),
          "+f"(c[0][12]), "+f"(c[0][13]), "+f"(c[1][12]), "+f"(c[1][13]), "+f"(c[0][14]), "+f"(c[0][15]), "+f"(c[1][14]), "+f"(c[1][15]),
          "+f"(c[0][16]), "+f"(c[0][17]), "+f"(c[1][16]), "+f"(c[1][17]), "+f"(c[0][18]), "+f"(c[0][19]), "+f"(c[1][18]), "+f"(c[1][19]),
          "+f"(c[0][20]), "+f"(c[0][21]), "+f"(c[1][20]), "+f"(c[1][21]), "+f"(c[0][22]), "+f"(c[0][23]), "+f"(c[1][22]), "+f"(c[1][23]),
          "+f"(c[0][24]), "+f"(c[0][25]), "+f"(c[1][24]), "+f"(c[1][25]), "+f"(c[0][26]), "+f"(c[0][27]), "+f"(c[1][26]), "+f"(c[1][27]),
          "+f"(c[0][28]), "+f"(c[0][29]), "+f"(c[1][28]), "+f"(c[1][29]), "+f"(c[0][30]), "+f"(c[0][31]), "+f"(c[1][30]), "+f"(c[1][31])
        : "l"(a_desc), "l"(b_desc), 
          "n"(int32_t(1)), "n"(int32_t(1)), "n"(int32_t(1)), // scaleD, scaleA, scaleB
          "n"(int32_t(0)), "n"(int32_t(0)) // transA, transB
    );
}

template <size_t tile_m, size_t tile_n, size_t tile_k>
__global__ void kernel(int M, int N, int K, 
    float *c, const __grid_constant__ CUtensorMap a, const __grid_constant__ CUtensorMap b) {
    constexpr size_t wgmma_m = 64, wgmma_n = 128, wgmma_k = 16;

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

    float frag_c[4][32] = {0};
    static_assert(sizeof(frag_c) * 128 == tile_m * tile_n * sizeof(float));
    for(int ok = 0; ok < K / tile_k; ok ++) {
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(smem_a, &a, ok*tile_k, tile_m_id*tile_m, bar_a); // be careful, k first then m
            cde::cp_async_bulk_tensor_2d_global_to_shared(smem_b, &b, ok*tile_k, tile_n_id*tile_n, bar_b); // be careful, k first then n
            token_a = cuda::device::barrier_arrive_tx(bar_a, 1, sizeof(smem_a));
            token_b = cuda::device::barrier_arrive_tx(bar_b, 1, sizeof(smem_b));
        } else {
            token_a = bar_a.arrive();
            token_b = bar_b.arrive();
        }
        bar_a.wait(std::move(token_a));
        bar_b.wait(std::move(token_b));
        __syncthreads(); // ????

        static_assert(tile_m / wgmma_m == 2 && tile_k / wgmma_k == 4
                        && tile_n == wgmma_n);
    
        /////// wgmma ///////
        wgmma_fence();
        for(int im = 0; im < tile_m / wgmma_m; im ++) {
            for(int ik = 0; ik < tile_k / wgmma_k; ik ++) {
                wgmma<tile_k>(frag_c+im*2,
                        smem_a+im*wgmma_m*tile_k+ik*wgmma_k,
                        smem_b                  +ik*wgmma_k);
            }
        }
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
    for(size_t im = 0; im < tile_m / wgmma_m; im++) {
        for(size_t rt = 0; rt < 2; rt ++) {
            for(size_t ct = 0; ct < 16; ct ++) {
                size_t i = im * wgmma_m + warp * 16 + rt * 8 + row_offset;
                size_t j =                            ct * 8 + col_offset;
                c_ptr[i * N + j + 0] = frag_c[im * 2 + rt][ct * 2 + 0];
                c_ptr[i * N + j + 1] = frag_c[im * 2 + rt][ct * 2 + 1];
            }
        }
    }
}

void runKernel2(size_t M, size_t N, size_t K, bf16 *A, bf16 *B, float *C) {
    constexpr size_t tile_m = 128;
    constexpr size_t tile_n = 128;
    constexpr size_t tile_k = 64;

    CUtensorMap tensor_a, tensor_b;
    createTensorMap(A, M, K, tile_m, tile_k, &tensor_a);
    createTensorMap(B, N, K, tile_n, tile_k, &tensor_b);

    dim3 block(128);
    dim3 grid(N / tile_n, M / tile_m);

    kernel<tile_m, tile_n, tile_k><<<grid, block>>>(M, N, K, C, tensor_a, tensor_b);
}

}

using mat2::runKernel2;