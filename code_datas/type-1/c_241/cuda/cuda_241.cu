#include <stdio.h>

// CUDA kernel function
__global__ void setIndexYolov3(int* input, int dims, int batchSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }

    for (int i = 0; i < batchSize; i++) {
        input[i * dims + tid] = tid;
    }
}

int main() {
    // Dimensions and batch size
    int dims = 5;
    int batchSize = 3;

    // Host array
    int* h_input = (int*)malloc(batchSize * dims * sizeof(int));

    // Device array
    int* d_input;
    cudaMalloc((void**)&d_input, batchSize * dims * sizeof(int));

    // Launch the CUDA kernel
    dim3 block_size(256);
    dim3 grid_size((dims + block_size.x - 1) / block_size.x);

    setIndexYolov3<<<grid_size, block_size>>>(d_input, dims, batchSize);

    // Copy the result back to the host
    cudaMemcpy(h_input, d_input, batchSize * dims * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result:\n");
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < dims; ++j) {
            printf("%d ", h_input[i * dims + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_input);
    cudaFree(d_input);

    return 0;
}
 
