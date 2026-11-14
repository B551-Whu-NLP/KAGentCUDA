#include <stdio.h>

// CUDA kernel function
__global__ void HistogramKernel(int* hist, unsigned char* img_in, int img_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < img_size) {
        atomicAdd(&(hist[img_in[i]]), 1);
    }
    __syncthreads();
}

int main() {
    // Image size
    int img_size = 100; // Change this according to your requirements

    // Host arrays
    unsigned char* h_img_in = (unsigned char*)malloc(img_size * sizeof(unsigned char));
    int* h_hist = (int*)malloc(256 * sizeof(int));

    // Initialize host input array (image)
    for (int i = 0; i < img_size; ++i) {
        h_img_in[i] = i % 256; // Example data for img_in, you can modify this accordingly
    }

    // Device arrays
    unsigned char* d_img_in;
    int* d_hist;
    cudaMalloc((void**)&d_img_in, img_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_hist, 256 * sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_img_in, h_img_in, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((img_size + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    HistogramKernel<<<grid_size, block_size>>>(d_hist, d_img_in, img_size);

    // Copy the result back to the host
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Image Array:\n");
    for (int i = 0; i < img_size; ++i) {
        printf("%d ", h_img_in[i]);
    }

    printf("\nHistogram:\n");
    for (int i = 0; i < 256; ++i) {
        printf("Value %d: %d\n", i, h_hist[i]);
    }

    // Clean up
    free(h_img_in);
    free(h_hist);
    cudaFree(d_img_in);
    cudaFree(d_hist);

    return 0;
}
 
