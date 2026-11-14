#include <stdio.h>

// CUDA kernel function
__global__ void cuda_ReLU_forward_kernel(float* d_in_data, bool* d_mask, const long unsigned int datasize, bool training) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= datasize)
        return;

    bool keep = d_in_data[i] > 0;

    if (training)
        d_mask[i] = keep;

    if (!keep)
        d_in_data[i] = 0;
}

int main() {
    // Array size
    long unsigned int datasize = 100; // Change this according to your requirements

    // Host arrays
    float* h_d_in_data = (float*)malloc(datasize * sizeof(float));
    bool* h_d_mask = (bool*)malloc(datasize * sizeof(bool));

    // Initialize host input array
    for (int i = 0; i < datasize; ++i) {
        h_d_in_data[i] = i - datasize / 2; // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_d_in_data;
    bool* d_d_mask;
    cudaMalloc((void**)&d_d_in_data, datasize * sizeof(float));
    cudaMalloc((void**)&d_d_mask, datasize * sizeof(bool));

    // Copy host input array to device
    cudaMemcpy(d_d_in_data, h_d_in_data, datasize * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((datasize + block_size - 1) / block_size, 1);

    // Training flag
    bool training = true; // Change this according to your requirements

    // Launch the CUDA kernel
    cuda_ReLU_forward_kernel<<<grid_size, block_size>>>(d_d_in_data, d_d_mask, datasize, training);

    // Copy the result back to the host
    cudaMemcpy(h_d_in_data, d_d_in_data, datasize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_mask, d_d_mask, datasize * sizeof(bool), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Input Data:\n");
    for (int i = 0; i < datasize; ++i) {
        printf("%.2f ", h_d_in_data[i]);
    }

    printf("\nMask (for training):\n");
    for (int i = 0; i < datasize; ++i) {
        printf("%d ", h_d_mask[i] ? 1 : 0);
    }
    printf("\n");

    // Clean up
    free(h_d_in_data);
    free(h_d_mask);
    cudaFree(d_d_in_data);
    cudaFree(d_d_mask);

    return 0;
}
 
