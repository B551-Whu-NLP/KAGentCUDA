#include <stdio.h>

// CUDA kernel function
__global__ void setOffset(int* offset, int dims, int batchSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) {
        return;
    }

    offset[0] = 0;

    for (int i = 1; i < batchSize + 1; i++) {
        offset[i] = i * dims;
    }
}

int main() {
    // Batch size and dimensions
    int batchSize = 5;
    int dims = 3;

    // Host array for offsets
    int* h_offset = (int*)malloc((batchSize + 1) * sizeof(int));

    // Device array for offsets
    int* d_offset;
    cudaMalloc((void**)&d_offset, (batchSize + 1) * sizeof(int));

    // Launch the CUDA kernel
    setOffset<<<1, 1>>>(d_offset, dims, batchSize);

    // Copy the result back to the host
    cudaMemcpy(h_offset, d_offset, (batchSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Offsets: ");
    for (int i = 0; i <= batchSize; i++) {
        printf("%d ", h_offset[i]);
    }
    printf("\n");

    // Clean up
    free(h_offset);
    cudaFree(d_offset);

    return 0;
}
 
