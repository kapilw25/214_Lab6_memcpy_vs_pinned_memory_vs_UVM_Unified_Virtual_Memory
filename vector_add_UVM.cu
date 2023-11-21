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
    float *h_A, *h_B, *h_C; // Unified Memory pointers

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&h_A, size);
    cudaMallocManaged(&h_B, size);
    cudaMallocManaged(&h_C, size);

    // Optional: Set Memory Advise Hints
    int device = -1;
    cudaGetDevice(&device);
    cudaMemAdvise(h_A, size, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(h_B, size, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(h_C, size, cudaMemAdviseSetPreferredLocation, device);

    // Initialize vectors in Unified Memory
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Start recording the execution time
    auto start = std::chrono::high_resolution_clock::now();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(h_A, h_B, h_C, numElements);

    // Stop recording the execution time
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();

    // Calculate and print the execution time
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    // Free Unified Memory
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);

    return 0;
}
