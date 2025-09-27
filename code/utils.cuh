#pragma once

#include <random>
#include <cstdio>
#include <string>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda/barrier>

using bf16 = __nv_bfloat16;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

void cudaCheck(cudaError_t error) {
    if(error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s line %d:\n%s\n", __FILE__, __LINE__,
            cudaGetErrorString(error));
        exit(1);
    }
}

__device__ uint64_t matrix_descriptor_encode(u_int64_t x) {
    return (((x) & 0x3FFFF) >> 4);
}

template <int box_width>
__device__ uint64_t matrix_descriptor_format(bf16 *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;

    size_t size_of_a_row = sizeof(bf16) * box_width; // be careful
    switch (size_of_a_row)
    {
    case 32:
        desc |= matrix_descriptor_encode((uint64_t)256) << 32; 
        desc |= 3llu << 62; // 32B swizzle
        break;
    case 64:
        desc |= matrix_descriptor_encode((uint64_t)512) << 32;
        desc |= 2llu << 62; // 64B swizzle
    case 128:
        desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
        desc |= 1llu << 62; // 128B swizzle
    default:
        break;
    }

    return desc;
}

template <typename T>
void createTensorMap(T *gmem_ptr, size_t height, size_t width, 
    uint32_t box_height, uint32_t box_width, CUtensorMap *tensor_map) {

    constexpr uint32_t rank = 2;
    uint64_t gmem_shape[rank] = {width, height};
    uint64_t gmem_stride[rank - 1] = {width * sizeof(T)};
    uint32_t smem_shape[rank] = {box_width, box_height};
    uint32_t smem_elt_stride[rank] = {1, 1};

    CUtensorMapSwizzle_enum swizzle;
    size_t size_of_a_row = sizeof(T) * box_width;
    switch (size_of_a_row)
    {
    case 32:
        swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
        std::cout << "[INFO] using CU_TENSOR_MAP_SWIZZLE_32B\n";
        break;
    case 64:
        swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
        std::cout << "[INFO] using CU_TENSOR_MAP_SWIZZLE_64B\n";
        break;
    case 128:
        swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
        std::cout << "[INFO] using CU_TENSOR_MAP_SWIZZLE_128B\n";
        break;
    default:
        break;
    }

    CUresult res = cuTensorMapEncodeTiled(
        tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        gmem_ptr,
        gmem_shape,
        gmem_stride,
        smem_shape,
        smem_elt_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, // be careful
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

template <typename T>
void randMatrix(T *in_ptr, size_t num_elts) {
    std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<float> dis{0.0, 1.0}; // μ=0, σ=1
    for(int i = 0; i < num_elts; i ++) {
        in_ptr[i] = static_cast<T>(dis(gen));
    }
}

template <typename T>
void verifyMatrix(T *ref, T *in, size_t row, size_t col, float tol) {
    for(int i = 0; i < row; i ++) {
        for(int j = 0; j < col; j ++) {
            float diff = static_cast<float>(ref[i * col + j] - in[i * col + j]);
            if(diff > tol || diff < -tol) {
                std::cout << "position ("  << i << ", " << j << ") has a diff " << diff <<
                ", ref = " << ref[i * col + j] << ", in = " << in[i * col + j]  << std::endl;
                return;
            }
        }
    }
    std::cout << "[SUCCESS] pass verify!" << std::endl;
}

template <typename T>
__host__ __device__ void printMatrix(T *in_ptr, size_t row, size_t col, const char *str) {
    printf("%s\n", str);
    for(int i = 0; i < row; i ++) {
        for(int j = 0; j < col; j ++) {
            printf("%8.2f ", static_cast<float>(in_ptr[i * col + j]));
        }
        printf("\n");
    }
}
