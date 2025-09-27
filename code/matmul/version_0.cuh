#pragma once

#include <omp.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include "../utils.cuh"

void runCublas(size_t M, size_t N, size_t K, bf16 *A, bf16 *B, float *C) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f; // use float type
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha, B, CUDA_R_16BF, K, // k major
                A, CUDA_R_16BF, K,
        &beta, C, CUDA_R_32F, N, // col major, be careful
        CUDA_R_32F, 
        CUBLAS_GEMM_DEFAULT
    );
}

void runCpu(size_t M, size_t N, size_t K, bf16 *A, bf16 *B, float *C) {
    #pragma omp prallel
    for(int i = 0; i < M; i ++) {
        #pragma omp prallel
        for(int j = 0; j < N; j ++) {
            float sum  = 0.0;
            for(int k = 0; k < K; k ++) {
                sum += static_cast<float>(A[i * K + k]) * 
                    static_cast<float>(B[k + j * K]);
            }
            C[i * N + j] = sum;
        }
    }
}
