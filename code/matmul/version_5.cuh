#pragma once

#include "../utils.cuh"

namespace mat5 {

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
__device__ void wgmma(float c[2][64], bf16 *sa, bf16 *sb) {
    uint64_t a_desc = matrix_descriptor_format<width>(sa);
    uint64_t b_desc = matrix_descriptor_format<width>(sb);
    asm volatile(
        " { "
        " wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        " {%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,    "
        "  %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,   "
        "  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,   "
        "  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,   "
        "  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,   "
        "  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,   "
        "  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,   "
        "  %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,   "
        "  %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,   "
        "  %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,   "
        "  %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,   "
        "  %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,   "
        "  %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        "  %104, %105, %106, %107, %108, %109, %110, %111,  "
        "  %112, %113, %114, %115, %116, %117, %118, %119,  "
        "  %120, %121, %122, %123, %124, %125, %126, %127}, "
        "  %128, %129, "
        "  %130, %131, %132, "
        "  %133, %134; \n "
        " } "
        : "+f"(c[0][0]),  "+f"(c[0][1]),  "+f"(c[1][0]),  "+f"(c[1][1]),  "+f"(c[0][2]),  "+f"(c[0][3]),  "+f"(c[1][2]),  "+f"(c[1][3]),
          "+f"(c[0][4]),  "+f"(c[0][5]),  "+f"(c[1][4]),  "+f"(c[1][5]),  "+f"(c[0][6]),  "+f"(c[0][7]),  "+f"(c[1][6]),  "+f"(c[1][7]),
          "+f"(c[0][8]),  "+f"(c[0][9]),  "+f"(c[1][8]),  "+f"(c[1][9]),  "+f"(c[0][10]), "+f"(c[0][11]), "+f"(c[1][10]), "+f"(c[1][11]),
          "+f"(c[0][12]), "+f"(c[0][13]), "+f"(c[1][12]), "+f"(c[1][13]), "+f"(c[0][14]), "+f"(c[0][15]), "+f"(c[1][14]), "+f"(c[1][15]),
          "+f"(c[0][16]), "+f"(c[0][17]), "+f"(c[1][16]), "+f"(c[1][17]), "+f"(c[0][18]), "+f"(c[0][19]), "+f"(c[1][18]), "+f"(c[1][19]),
          "+f"(c[0][20]), "+f"(c[0][21]), "+f"(c[1][20]), "+f"(c[1][21]), "+f"(c[0][22]), "+f"(c[0][23]), "+f"(c[1][22]), "+f"(c[1][23]),
          "+f"(c[0][24]), "+f"(c[0][25]), "+f"(c[1][24]), "+f"(c[1][25]), "+f"(c[0][26]), "+f"(c[0][27]), "+f"(c[1][26]), "+f"(c[1][27]),
          "+f"(c[0][28]), "+f"(c[0][29]), "+f"(c[1][28]), "+f"(c[1][29]), "+f"(c[0][30]), "+f"(c[0][31]), "+f"(c[1][30]), "+f"(c[1][31]),
          "+f"(c[0][32]), "+f"(c[0][33]), "+f"(c[1][32]), "+f"(c[1][33]), "+f"(c[0][34]), "+f"(c[0][35]), "+f"(c[1][34]), "+f"(c[1][35]),
          "+f"(c[0][36]), "+f"(c[0][37]), "+f"(c[1][36]), "+f"(c[1][37]), "+f"(c[0][38]), "+f"(c[0][39]), "+f"(c[1][38]), "+f"(c[1][39]),
          "+f"(c[0][40]), "+f"(c[0][41]), "+f"(c[1][40]), "+f"(c[1][41]), "+f"(c[0][42]), "+f"(c[0][43]), "+f"(c[1][42]), "+f"(c[1][43]),
          "+f"(c[0][44]), "+f"(c[0][45]), "+f"(c[1][44]), "+f"(c[1][45]), "+f"(c[0][46]), "+f"(c[0][47]), "+f"(c[1][46]), "+f"(c[1][47]),
          "+f"(c[0][48]), "+f"(c[0][49]), "+f"(c[1][48]), "+f"(c[1][49]), "+f"(c[0][50]), "+f"(c[0][51]), "+f"(c[1][50]), "+f"(c[1][51]),
          "+f"(c[0][52]), "+f"(c[0][53]), "+f"(c[1][52]), "+f"(c[1][53]), "+f"(c[0][54]), "+f"(c[0][55]), "+f"(c[1][54]), "+f"(c[1][55]),
          "+f"(c[0][56]), "+f"(c[0][57]), "+f"(c[1][56]), "+f"(c[1][57]), "+f"(c[0][58]), "+f"(c[0][59]), "+f"(c[1][58]), "+f"(c[1][59]),
          "+f"(c[0][60]), "+f"(c[0][61]), "+f"(c[1][60]), "+f"(c[1][61]), "+f"(c[0][62]), "+f"(c[0][63]), "+f"(c[1][62]), "+f"(c[1][63])
        : "l"(a_desc), "l"(b_desc), 
          "n"(int32_t(1)), "n"(int32_t(1)), "n"(int32_t(1)), // scaleD, scaleA, scaleB
          "n"(int32_t(0)), "n"(int32_t(0)) // transA, transB
    );
}

template<int SMs, int BM, int BN, int WM, int WN>
struct Scheduler {
    int it_;
    int blockIdx_;
    int total_block_at_m_; // block at m dim
    int total_block_at_n_; // block at n dim

    __device__ __forceinline__ Scheduler(int M, int N, int blockIdx) {
        total_block_at_m_ = M / BM;
        total_block_at_n_ = N / BN;
        blockIdx_ = blockIdx;
        it_ = 0; // be careful
        assert(total_block_at_m_ % WM == 0);
        assert(total_block_at_n_ % WN == 0);
        assert(SMs == WM * WN);
        assert(blockIdx_ < SMs);
    }

    __device__ __forceinline__ int next() {
        int block_id = it_ * SMs + blockIdx_;
        if(block_id >= total_block_at_m_*total_block_at_n_)
            return -1;
        
        // which tile
        int tile_id = block_id / (WM * WN);
        int m = tile_id / (total_block_at_n_ / WN) * WM;
        int n = tile_id % (total_block_at_n_ / WN) * WN;
        // which block
        m += (blockIdx_ / WN);
        n += (blockIdx_ % WN);

        it_++;

        return m * total_block_at_n_ + n;
    }
};

template <int block_m, int block_n, int block_k, int num_stage>
struct SMem {
    alignas(16) bf16 A[block_m * block_k * num_stage];
    alignas(16) bf16 B[block_n * block_k * num_stage];
};

template <int num_stage, int block_m, int block_n, int block_k, int num_sm>
__global__ void kernel(int M, int N, int K, 
    float *c, const __grid_constant__ CUtensorMap a, const __grid_constant__ CUtensorMap b) {
    constexpr int wgmma_m = 64, wgmma_n = 256, wgmma_k = 16;

    static_assert(block_m / wgmma_m == 2 && block_k / wgmma_k == 4
                        && block_n == wgmma_n);

    // Scheduler<num_sm, block_m, block_n, 8, 16> scheduler(M, N, blockIdx.x);
    Scheduler<num_sm, block_m, block_n, 16, 8> scheduler(M, N, blockIdx.x); // better performance

    extern __shared__ __align__(128) uint8_t smem[];
    SMem<block_m, block_n, block_k, num_stage> &s = *reinterpret_cast<SMem<block_m, block_n, block_k, num_stage>*>(smem);
    bf16 *smem_a = s.A;
    bf16 *smem_b = s.B;

    // int block_m_id = blockIdx.y;
    // int block_n_id = blockIdx.x;
    /////// create barrier ///////
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full[num_stage];
    __shared__ barrier empty[num_stage];
    if (threadIdx.x == 0) {
        for(int i = 0; i < num_stage; i ++) {
            init(&full[i], 128 * 2 + 1);
            init(&empty[i], 128 * 2 + 1);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    /////// lgsts ///////
    // __shared__ __align__(128) bf16 smem_a[num_stage * block_m * block_k];
    // __shared__ __align__(128) bf16 smem_b[num_stage * block_n * block_k];

    int wg_id = threadIdx.x / 128;
    if(wg_id == 0) { // producer
        if (threadIdx.x == 0) {
            for(int block_id = scheduler.next(); block_id >= 0; block_id = scheduler.next()) {
                int block_m_id = block_id / (N / block_n);
                int block_n_id = block_id % (N / block_n);
                if(blockIdx.x == 0) {
                    printf("producer: %d, %d\n", block_m_id, block_n_id);
                }
                for(int ok = 0; ok < K / block_k; ok ++) {
                    empty[ok%num_stage].wait(empty[ok%num_stage].arrive());
                    cde::cp_async_bulk_tensor_2d_global_to_shared(smem_a+(ok%num_stage)*block_m*block_k, &a, ok*block_k, block_m_id*block_m, full[ok%num_stage]); // be careful, k first then m
                    cde::cp_async_bulk_tensor_2d_global_to_shared(smem_b+(ok%num_stage)*block_n*block_k, &b, ok*block_k, block_n_id*block_n, full[ok%num_stage]); // be careful, k first then n
                    barrier::arrival_token _ = cuda::device::barrier_arrive_tx(full[ok%num_stage], 1, (block_m*block_k + block_n*block_k)*sizeof(bf16));
                }
            }
        }
    } else { // consumer
        float frag_c[2][64] = {0};
        static_assert(sizeof(frag_c) * 128 == block_m / 2 * block_n * sizeof(float));

        for(int ns = 0; ns < num_stage; ns ++) {
            barrier::arrival_token _ = empty[ns].arrive();
        }

        for(int block_id = scheduler.next(); block_id >= 0; block_id = scheduler.next()) {
        int block_m_id = block_id / (N / block_n);
        int block_n_id = block_id % (N / block_n);

        if((blockIdx.x == 0) && (threadIdx.x == 128 || threadIdx.x == 256)) {
            printf("consumer: %d, %d\n", block_m_id, block_n_id);
        }

        for(int ok = 0; ok < K / block_k; ok ++) {
            full[ok%num_stage].wait(full[ok%num_stage].arrive());

            /////// wgmma ///////
            wgmma_fence();
            // #pragma unroll
            // for(int im = 0; im < block_m / wgmma_m; im++) {
                #pragma unroll
                for(int ik = 0; ik < block_k / wgmma_k; ik ++) {
                    wgmma<block_k>(frag_c,
                            smem_a + (ok%num_stage)*block_m*block_k +(wg_id-1)*wgmma_m*block_k+ik*wgmma_k,
                            smem_b + (ok%num_stage)*block_n*block_k                          +ik*wgmma_k);
                }
            // }
            wgmma_commit();
            wgmma_wait<0>();

            barrier::arrival_token _ = empty[ok%num_stage].arrive();
        }

        /////// stg ///////
        int warp = (threadIdx.x % 128) / 32, // be careful
            lane = threadIdx.x % 32,
            row_offset = lane / 4,
            col_offset = lane % 4 * 2;
        float* c_ptr = c + 
                block_m_id * block_m * N + 
                block_n_id * block_n;
        
        // #pragma unroll
        // for(int im = 0; im < block_m / wgmma_m; im++) {
            #pragma unroll
            for(int rt = 0; rt < 2; rt ++) {
                #pragma unroll
                for(int ct = 0; ct < 32; ct ++) {
                    int i = (wg_id-1) * wgmma_m + warp * 16 + rt * 8 + row_offset;
                    int j =                           ct * 8 + col_offset;
                    c_ptr[i * N + j + 0] = frag_c[0 * 2 + rt][ct * 2 + 0];
                    c_ptr[i * N + j + 1] = frag_c[0 * 2 + rt][ct * 2 + 1];
                }
            }
        // }
        }
    }
}

void runKernel5(int M, int N, int K, bf16 *A, bf16 *B, float *C) {
    constexpr int block_m = 128;
    constexpr int block_n = 256;
    constexpr int block_k = 64;
    constexpr int num_stage = 3;
    constexpr int num_sm = 128;

    // constexpr int smem_size = num_stage * (block_m * block_k + block_n * block_k) * sizeof(bf16) / 1024; // shm
    CUtensorMap tensor_a, tensor_b;
    createTensorMap(A, M, K, block_m, block_k, &tensor_a);
    createTensorMap(B, N, K, block_n, block_k, &tensor_b);

    dim3 block(128 * 3);
    dim3 grid(num_sm);
    // dim3 grid(N / block_n, M / block_m);

    auto *kernel_ptr = kernel<num_stage, block_m, block_n, block_k, num_sm>;
    size_t smem_size = sizeof(SMem<block_m, block_n, block_k, num_stage>);
    cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel_ptr<<<grid, block, smem_size>>>(M, N, K, C, tensor_a, tensor_b);

    // 立即检查启动错误
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(launchError));
        return;
    }
}

}

using mat5::runKernel5;
