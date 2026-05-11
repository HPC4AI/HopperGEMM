#include <iostream>
#include <cstdlib>
#include <unistd.h>

#include "utils.cuh"
#include "./matmul/version_0.cuh"
#include "./matmul/version_1.cuh"
#include "./matmul/version_2.cuh"
#include "./matmul/version_3.cuh"
#include "./matmul/version_4.cuh"
#include "./matmul/version_5.cuh"

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
    default:
        std::cerr << "[ERROR] unknown kernel_num: " << kernel_num << std::endl;
        exit(1);
    }
}

const char* kernelName(int kernel_num) {
    switch (kernel_num) {
        case -1: return "CPU";
        case 0:  return "cuBLAS";
        case 1:  return "Kernel1";
        case 2:  return "Kernel2";
        case 3:  return "Kernel3";
        case 4:  return "Kernel4";
        case 5:  return "Kernel5";
        default: return "Unknown";
    }
}

int main(int argc, const char *argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "[INFO] Num SMs: " << prop.multiProcessorCount << std::endl;

    // All kernels to benchmark
    // -1 = CPU, 0 = cuBLAS, 1~5 = custom kernels
    int all_kernels[] = {-1, 0, 1, 2, 3, 4, 5};
    int num_kernels = sizeof(all_kernels) / sizeof(all_kernels[0]);

    size_t M, N, K;
    M = 8192, N = 8192, K = 8192;

    ///////////// prepare host data /////////////
    size_t size_a_in_bytes = M * K * sizeof(bf16),
           size_b_in_bytes = N * K * sizeof(bf16),
           size_c_in_bytes = M * N * sizeof(float);
    bf16 *h_a_ptr, *h_b_ptr;
    h_a_ptr = (bf16*) malloc(size_a_in_bytes);
    h_b_ptr = (bf16*) malloc(size_b_in_bytes);

    randMatrix(h_a_ptr, M * K);
    randMatrix(h_b_ptr, N * K);

    ///////////// prepare device data /////////////
    bf16 *d_a_ptr, *d_b_ptr;
    cudaCheck(cudaMalloc(&d_a_ptr, size_a_in_bytes));
    cudaCheck(cudaMalloc(&d_b_ptr, size_b_in_bytes));
    cudaMemcpy(d_a_ptr, h_a_ptr, size_a_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_ptr, h_b_ptr, size_b_in_bytes, cudaMemcpyHostToDevice);

    float *h_c_ref_ptr, *h_c_ptr, *d_c_ref_ptr, *d_c_ptr;
    h_c_ref_ptr = (float*) malloc(size_c_in_bytes);
    h_c_ptr     = (float*) malloc(size_c_in_bytes);
    cudaCheck(cudaMalloc(&d_c_ref_ptr, size_c_in_bytes));
    cudaCheck(cudaMalloc(&d_c_ptr,     size_c_in_bytes));

    ///////////// compute cuBLAS reference result once /////////////
    std::cout << "\n===== Computing cuBLAS reference =====" << std::endl;
    runKernel(0, M, N, K, d_a_ptr, d_b_ptr, d_c_ref_ptr);
    cudaCheck(cudaDeviceSynchronize());
    cudaMemcpy(h_c_ref_ptr, d_c_ref_ptr, size_c_in_bytes, cudaMemcpyDeviceToHost);
    std::cout << "Reference ready.\n" << std::endl;

    ///////////// bench config /////////////
    int repeat = 8;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ///////////// loop over all kernels /////////////
    for (int idx = 0; idx < num_kernels; idx++) {
        int kernel_num = all_kernels[idx];
        std::cout << "========================================" << std::endl;
        std::cout << "KERNEL " << kernel_num << " (" << kernelName(kernel_num) << ")" << std::endl;
        std::cout << "========================================" << std::endl;

        // Give the GPU some rest to avoid thermal throttling
        sleep(3);

        // ---- Step 1: Correctness verification ----
        if (kernel_num == -1) {
            // CPU kernel: use host memory directly
            memset(h_c_ptr, 0, size_c_in_bytes);
            runKernel(kernel_num, M, N, K, h_a_ptr, h_b_ptr, h_c_ptr);
            bool passed = verifyMatrixSilent(h_c_ref_ptr, h_c_ptr, M, N, 1e-2);
            if (!passed) {
                std::cout << "  [FAIL] Verification FAILED!" << std::endl;
            } else {
                std::cout << "  [PASS] Verification passed." << std::endl;
            }
        } else {
            // GPU kernels: use device memory
            cudaCheck(cudaMemset(d_c_ptr, 0, size_c_in_bytes));
            runKernel(kernel_num, M, N, K, d_a_ptr, d_b_ptr, d_c_ptr);
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run

            // Copy result back and verify against cuBLAS reference
            cudaMemcpy(h_c_ptr, d_c_ptr, size_c_in_bytes, cudaMemcpyDeviceToHost);
            bool passed = verifyMatrixSilent(h_c_ref_ptr, h_c_ptr, M, N, 1e-4);
            if (!passed) {
                std::cout << "  [FAIL] Verification FAILED!" << std::endl;
            } else {
                std::cout << "  [PASS] Verification passed." << std::endl;
            }
        }

        // Skip benchmark for CPU kernel (too slow for repeated runs)
        if (kernel_num == -1) {
            std::cout << "  (Skipping performance benchmark for CPU)\n" << std::endl;
            continue;
        }

        // ---- Step 2: Performance benchmark ----
        cudaEventRecord(start);
        for (int r = 0; r < repeat; r++) {
            runKernel(kernel_num, M, N, K, d_a_ptr, d_b_ptr, d_c_ptr);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        long flops = (2LL * M) * (N * K);
        printf(
            "Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n\n",
            elapsed_time / 1000.0 / repeat,
            (repeat * flops * 1e-9) / elapsed_time, M
        );
    }

    ///////////// cleanup /////////////
    free(h_a_ptr); free(h_b_ptr);
    free(h_c_ref_ptr); free(h_c_ptr);
    cudaFree(d_a_ptr); cudaFree(d_b_ptr);
    cudaFree(d_c_ptr); cudaFree(d_c_ref_ptr);

    return 0;
}
