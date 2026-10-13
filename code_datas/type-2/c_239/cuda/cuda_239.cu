#include <stdio.h>

// CUDA kernel function
__global__ void sumRowKernel(int* d_in, int* d_out, int DIM) {
    for (int bid = blockIdx.x; bid < DIM; bid += gridDim.x) {
        int sum = 0;
        for (int tid = threadIdx.x; tid < DIM; tid += blockDim.x) {
            sum += d_in[tid + bid * DIM];
        }
        atomicAdd(&d_out[bid], sum);
    }
}

int main() {
    // Matrix dimensions
    int DIM = 5; // Change this according to your requirements

    // Host matrices
    int* h_in = (int*)malloc(DIM * DIM * sizeof(int));
    int* h_out = (int*)malloc(DIM * sizeof(int));

    // Initialize host input matrix
    for (int i = 0; i < DIM * DIM; ++i) {
        h_in[i] = i + 1; // Example data, you can modify this accordingly
    }

    // Device matrices
    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in, DIM * DIM * sizeof(int));
    cudaMalloc((void**)&d_out, DIM * sizeof(int));

    // Copy host input matrix to device
    cudaMemcpy(d_in, h_in, DIM * DIM * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    dim3 block_size(256);
    dim3 grid_size((DIM + block_size.x - 1) / block_size.x);
    sumRowKernel<<<grid_size, block_size>>>(d_in, d_out, DIM);

    // Copy the result back to the host
    cudaMemcpy(h_out, d_out, DIM * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Sum of Rows: ");
    for (int i = 0; i < DIM; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
 
