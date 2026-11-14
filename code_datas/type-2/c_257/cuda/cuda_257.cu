#include <stdio.h>

// CUDA kernel function
__global__ void HammingDistance(int* c, const int* a, const int* b, const long int* size) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < *size; i += stride) {
        if (a[i] != b[i]) {
            atomicAdd(c, 1);
        }
    }
}

int main() {
    // Array size
    long int size = 1000; // Change this according to your requirements

    // Host arrays
    int* h_a = (int*)malloc(size * sizeof(int));
    int* h_b = (int*)malloc(size * sizeof(int));
    int* h_c = (int*)malloc(sizeof(int));

    // Initialize host input arrays
    for (int i = 0; i < size; ++i) {
        h_a[i] = i % 2; // Example data, you can modify this accordingly
        h_b[i] = (i + 1) % 2; // Example data, you can modify this accordingly
    }

    *h_c = 0;

    // Device arrays
    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    // Copy host input arrays to device
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((size + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    HammingDistance<<<grid_size, block_size>>>(d_c, d_a, d_b, &size);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Hamming Distance: %d\n", *h_c);

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
 
