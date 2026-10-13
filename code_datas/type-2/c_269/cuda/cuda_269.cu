#include <stdio.h>

// CUDA kernel function
__global__ void gpuSearchPosShmem1(int key, int* gpu_key_arr, int* gpu_pos, int size) {
    int globalTx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalTx < size - 1) {
        if (key >= gpu_key_arr[globalTx] && key < gpu_key_arr[globalTx + 1]) {
            *gpu_pos = globalTx;
        }
    }
}

int main() {
    // Array size
    int size = 10; // Change this according to your requirements

    // Host arrays
    int* h_gpu_key_arr = (int*)malloc(size * sizeof(int));
    int* h_gpu_pos = (int*)malloc(sizeof(int));

    // Initialize host input array (sorted keys)
    for (int i = 0; i < size; ++i) {
        h_gpu_key_arr[i] = i * 10; // Example data for gpu_key_arr, you can modify this accordingly
    }

    // Device arrays
    int* d_gpu_key_arr;
    int* d_gpu_pos;
    cudaMalloc((void**)&d_gpu_key_arr, size * sizeof(int));
    cudaMalloc((void**)&d_gpu_pos, sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_gpu_key_arr, h_gpu_key_arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((size + block_size - 1) / block_size, 1);

    // Search key
    int key = 15; // Change this according to your requirements

    // Launch the CUDA kernel
    gpuSearchPosShmem1<<<grid_size, block_size>>>(key, d_gpu_key_arr, d_gpu_pos, size);

    // Copy the result back to the host
    cudaMemcpy(h_gpu_pos, d_gpu_pos, sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("gpu_key_arr Array:\n");
    for (int i = 0; i < size; ++i) {
        printf("%d ", h_gpu_key_arr[i]);
    }

    printf("\nSearch Key: %d\n", key);
    printf("Position: %d\n", *h_gpu_pos);

    // Clean up
    free(h_gpu_key_arr);
    free(h_gpu_pos);
    cudaFree(d_gpu_key_arr);
    cudaFree(d_gpu_pos);

    return 0;
}
 
