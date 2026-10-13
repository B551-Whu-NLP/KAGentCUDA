#include <stdio.h>

// CUDA kernel for odd-even sort
__global__ void oddevenSort(int* d_in, int size, int oe_flag, int& d_ch_flag) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int p = 2 * idx + oe_flag;

    if (p + 1 < size) {
        if (d_in[p] > d_in[p + 1]) {
            int temp = d_in[p];
            d_in[p] = d_in[p + 1];
            d_in[p + 1] = temp;
            d_ch_flag = 1;
        }
    }
}

int main() {
    // Set your desired parameters
    int size = 512;

    // Allocate memory on the host
    int* h_d_in = (int*)malloc(size * sizeof(int));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    int* d_d_in;
    cudaMalloc((void**)&d_d_in, size * sizeof(int));

    // Copy data to device memory

    // Set the flag on the host
    int h_d_ch_flag = 0;

    // Allocate memory for the flag on the device
    int* d_d_ch_flag;
    cudaMalloc((void**)&d_d_ch_flag, sizeof(int));

    // Copy the flag to the device
    cudaMemcpy(d_d_ch_flag, &h_d_ch_flag, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 gridSize((size + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for odd-even sort
    oddevenSort<<<gridSize, blockSize>>>(d_d_in, size, 0, *d_d_ch_flag);

    // Copy the result and flag back to the host
    cudaMemcpy(&h_d_ch_flag, d_d_ch_flag, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_d_in);
    cudaFree(d_d_ch_flag);

    // Free host memory
    free(h_d_in);

    return 0;
}
