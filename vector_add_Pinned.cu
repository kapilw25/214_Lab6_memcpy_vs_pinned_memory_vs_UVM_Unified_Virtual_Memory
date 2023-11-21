#include <cuda.h>
#include <iostream>
#include <chrono>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 50000; // Size of the vectors
    size_t size = numElements * sizeof(float);
    float *h_A, *h_B, *h_C; // Host vectors
    float *d_A, *d_B, *d_C; // Device vectors

    // Allocate pinned host memory
    cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, size, cudaHostAllocDefault);

    // Initialize host vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Start recording the execution time
    auto start = std::chrono::high_resolution_clock::now();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Stop recording the execution time
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();

    // Copy result vector from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Calculate and print the execution time
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free pinned host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
