#include <stdio.h>

// CUDA kernel function
__global__ void add_vec_scalaire_gpu(int* vec, int* res, int a, long N) {
    long i = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (i < N) {
        res[i] = vec[i] + a;
    }
}

int main() {
    // Vector size
    long N = 100; // Change this according to your requirements

    // Host vectors
    int* h_vec;
    int* h_res;
    h_vec = (int*)malloc(N * sizeof(int));
    h_res = (int*)malloc(N * sizeof(int));

    // Initialize host vectors
    for (long i = 0; i < N; ++i) {
        h_vec[i] = i; // Example data, you can modify this accordingly
    }

    // Device vectors
    int* d_vec;
    int* d_res;
    cudaMalloc((void**)&d_vec, N * sizeof(int));
    cudaMalloc((void**)&d_res, N * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_vec, h_vec, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((N + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    add_vec_scalaire_gpu<<<grid_size, block_size>>>(d_vec, d_res, 5, N); // Example scalar 'a' is set to 5

    // Copy the result back to the host
    cudaMemcpy(h_res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    for (long i = 0; i < N; ++i) {
        printf("%d ", h_res[i]);
    }
    printf("\n");

    // Clean up
    free(h_vec);
    free(h_res);
    cudaFree(d_vec);
    cudaFree(d_res);

    return 0;
}
 
 
