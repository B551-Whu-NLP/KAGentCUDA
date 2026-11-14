#include <stdio.h>

// CUDA kernel function
__global__ void shiftIndices(long* vec_out, const long by, const long imageSize, const long N) {
    long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        vec_out[idx] = (imageSize + ((idx - N / 2 + by) % imageSize)) % imageSize;
    }
}

int main() {
    // Vector size and parameters
    long N = 10;
    long imageSize = 8;
    long shiftAmount = 3;

    // Host array
    long* h_vec_out = (long*)malloc(N * sizeof(long));

    // Device array
    long* d_vec_out;
    cudaMalloc((void**)&d_vec_out, N * sizeof(long));

    // Launch the CUDA kernel
    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    shiftIndices<<<grid_size, block_size>>>(d_vec_out, shiftAmount, imageSize, N);

    // Copy the result back to the host
    cudaMemcpy(h_vec_out, d_vec_out, N * sizeof(long), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result:\n");
    for (long i = 0; i < N; ++i) {
        printf("%ld ", h_vec_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_vec_out);
    cudaFree(d_vec_out);

    return 0;
}
 
