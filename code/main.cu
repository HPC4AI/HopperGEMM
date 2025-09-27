#include <iostream>
#include <cstdlib>

#include "utils.cuh"
#include "./matmul/version_0.cuh"
#include "./matmul/version_1.cuh"
#include "./matmul/version_2.cuh"
#include "./matmul/version_3.cuh"
#include "./matmul/version_4.cuh"
#include "./matmul/version_5.cuh"

// nvcc -o main main.cu -O2 -arch=sm_90a -Xcompiler -fopenmp -lcuda -std=c++17 -lcublas  -I. && ./main

void runKernel(int kernel_num, 
    int M, int N, int K, 
    bf16 *A, bf16 *B, float *C, 
    int *DB = nullptr) 
{
    switch (kernel_num)
    {
    case -1:
        std::cout << "[INFO] runCpu\n";
        runCpu(M, N, K, A, B, C);
        break;
    case 0:
        std::cout << "[INFO] runCublas\n";
        runCublas(M, N, K, A, B, C);
        break;
    case 1:
        std::cout << "[INFO] runKernel1\n";
        runKernel1(M, N, K, A, B, C);
        break;
    case 2:
        std::cout << "[INFO] runKernel2\n";
        runKernel2(M, N, K, A, B, C);
        break;
    case 3:
        std::cout << "[INFO] runKernel3\n";
        runKernel3(M, N, K, A, B, C);
        break;
    case 4:
        std::cout << "[INFO] runKernel4\n";
        runKernel4(M, N, K, A, B, C);
        break;
    case 5:
        std::cout << "[INFO] runKernel5\n";
        runKernel5(M, N, K, A, B, C);
        break;
    }
}

int main(int argc, const char *argv[])
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "[INFO] Num SMs: " << prop.multiProcessorCount << std::endl;

    int kernel_num = 5;

    /// TODO: a loop here & edge case
    size_t M, N, K;
    // M = 128*64, N = 128*64, K = 128*64;
    M = 2048*2, N = 2048, K = 8192;

    ///////////// prepare host data /////////////
    size_t size_a_in_bytes = M * K * sizeof(bf16),
        size_b_in_bytes = N * K * sizeof(bf16);
    bf16 *h_a_ptr, *h_b_ptr;
    h_a_ptr = (bf16*) malloc (size_a_in_bytes);
    h_b_ptr = (bf16*) malloc (size_b_in_bytes);

    randMatrix(h_a_ptr, M * K);
    randMatrix(h_b_ptr, N * K);

    ///////////// prepare device data /////////////
    bf16 *d_a_ptr, *d_b_ptr;
    cudaCheck(cudaMalloc(&d_a_ptr, size_a_in_bytes));
    cudaCheck(cudaMalloc(&d_b_ptr, size_b_in_bytes));
    cudaCheck(cudaMemcpy(d_a_ptr, h_a_ptr, size_a_in_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b_ptr, h_b_ptr, size_b_in_bytes, cudaMemcpyHostToDevice));

    ///////////// run my kernel /////////////
    size_t size_c_in_bytes = M * N * sizeof(float);
    float *h_c_ptr, *d_c_ptr;
    h_c_ptr = (float*) malloc (size_c_in_bytes);
    cudaMalloc(&d_c_ptr, size_c_in_bytes);
    runKernel(kernel_num, M, N, K, d_a_ptr, d_b_ptr, d_c_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c_ptr, d_c_ptr, size_c_in_bytes, cudaMemcpyDeviceToHost);

    ///////////// run cublas kernel /////////////
    float *h_c_cublas_ptr, *d_c_cublas_ptr;
    h_c_cublas_ptr = (float*) malloc (size_c_in_bytes);
    cudaMalloc(&d_c_cublas_ptr, size_c_in_bytes);
    runKernel(0, M, N, K, d_a_ptr, d_b_ptr, d_c_cublas_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c_cublas_ptr, d_c_cublas_ptr, size_c_in_bytes, cudaMemcpyDeviceToHost);

    ///////////// run cpu kernel /////////////
    // float *h_c_cpu_ptr;a_a_ptr, h_b_ptr, h_c_cpu_ptr);

    ///////////// verify results /////////////
    verifyMatrix(h_c_cublas_ptr, h_c_ptr, M, N, 1e-4);

    ///////////// bench /////////////
    int repeat = 8;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int r = 0; r < repeat; r ++) {
        runKernel(kernel_num, M, N, K, d_a_ptr, d_b_ptr, d_c_ptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    long flops = (2LL * M * N * K);
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n",
        elapsed_time / 1000.0 / repeat,
        (repeat * flops * 1e-9) / elapsed_time, M
    );
}
