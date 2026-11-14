#include <stdio.h>
#include <math.h>

// CUDA kernel function
__global__ void pythagoras(unsigned char* a, unsigned char* b, unsigned char* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float af = static_cast<float>(a[idx]);
        float bf = static_cast<float>(b[idx]);
        c[idx] = static_cast<unsigned char>(sqrtf(af * af + bf * bf));
    }
}

int main() {
    // Array size
    int size = 1024;  // Change this according to your requirements

    // Host arrays
    unsigned char* h_a = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* h_b = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* h_c = (unsigned char*)malloc(size * sizeof(unsigned char));

    // Initialize host input arrays
    for (int i = 0; i < size; ++i) {
        h_a[i] = 100;  // Example data for a, you can modify this accordingly
        h_b[i] = 150;  // Example data for b, you can modify this accordingly
    }

    // Device arrays
    unsigned char* d_a;
    unsigned char* d_b;
    unsigned char* d_c;
    cudaMalloc((void**)&d_a, size * sizeof(unsigned char));
    cudaMalloc((void**)&d_b, size * sizeof(unsigned char));
    cudaMalloc((void**)&d_c, size * sizeof(unsigned char));

    // Copy host input arrays to device
    cudaMemcpy(d_a, h_a, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Launch the CUDA kernel
    pythagoras<<<grid_size, block_size>>>(d_a, d_b, d_c, size);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Results:\n");
    for (int i = 0; i < size; ++i) {
        printf("%u ", h_c[i]);
    }
    printf("\n");

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
 
