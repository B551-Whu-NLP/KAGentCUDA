#include <stdio.h>

// CUDA kernel function
__global__ void mathKernel1(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

int main() {
    // Array size
    int size = 10; // Change this according to your requirements

    // Host array
    float* h_c = (float*)malloc(size * sizeof(float));

    // Device array
    float* d_c;
    cudaMalloc((void**)&d_c, size * sizeof(float));

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((size + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    mathKernel1<<<grid_size, block_size>>>(d_c);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < size; ++i) {
        printf("%f ", h_c[i]);
    }
    printf("\n");

    // Clean up
    free(h_c);
    cudaFree(d_c);

    return 0;
}
 
