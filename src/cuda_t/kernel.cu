#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

extern "C" void add_cuda_launcher(const float* a, const float* b, float* c, int64_t size) {
    const int threads = 256;
    const int blocks = (int)((size + threads - 1) / threads);
    add_kernel<<<blocks, threads>>>(a, b, c, size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}