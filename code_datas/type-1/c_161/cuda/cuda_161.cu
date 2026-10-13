#include <stdio.h>

// Define the CUDA kernel
__global__ void rgb2yuv_kernel(int img_size, unsigned char *gpu_img_in_r, unsigned char *gpu_img_in_g, unsigned char *gpu_img_in_b,
                                unsigned char *gpu_img_out_y, unsigned char *gpu_img_out_u, unsigned char *gpu_img_out_v) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < img_size) {
        unsigned char r = gpu_img_in_r[index];
        unsigned char g = gpu_img_in_g[index];
        unsigned char b = gpu_img_in_b[index];

        gpu_img_out_y[index] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        gpu_img_out_u[index] = (unsigned char)(-0.169 * r - 0.331 * g + 0.499 * b + 128);
        gpu_img_out_v[index] = (unsigned char)(0.499 * r - 0.418 * g - 0.0813 * b + 128);
    }
}

int main() {
    // Example usage
    int img_size = 1000;  // Set your image size accordingly
    unsigned char *gpu_img_in_r, *gpu_img_in_g, *gpu_img_in_b, *gpu_img_out_y, *gpu_img_out_u, *gpu_img_out_v;
    // Assuming these arrays are allocated and initialized

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    unsigned char *d_gpu_img_in_r, *d_gpu_img_in_g, *d_gpu_img_in_b, *d_gpu_img_out_y, *d_gpu_img_out_u, *d_gpu_img_out_v;
    cudaMalloc((void **)&d_gpu_img_in_r, img_size * sizeof(unsigned char));
    cudaMalloc((void **)&d_gpu_img_in_g, img_size * sizeof(unsigned char));
    cudaMalloc((void **)&d_gpu_img_in_b, img_size * sizeof(unsigned char));
    cudaMalloc((void **)&d_gpu_img_out_y, img_size * sizeof(unsigned char));
    cudaMalloc((void **)&d_gpu_img_out_u, img_size * sizeof(unsigned char));
    cudaMalloc((void **)&d_gpu_img_out_v, img_size * sizeof(unsigned char));

    // Copy data from host to device
    cudaMemcpy(d_gpu_img_in_r, gpu_img_in_r, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu_img_in_g, gpu_img_in_g, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu_img_in_b, gpu_img_in_b, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (img_size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    rgb2yuv_kernel<<<blocksPerGrid, threadsPerBlock>>>(img_size, d_gpu_img_in_r, d_gpu_img_in_g, d_gpu_img_in_b,
                                                       d_gpu_img_out_y, d_gpu_img_out_u, d_gpu_img_out_v);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(gpu_img_out_y, d_gpu_img_out_y, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_img_out_u, d_gpu_img_out_u, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_img_out_v, d_gpu_img_out_v, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_gpu_img_in_r);
    cudaFree(d_gpu_img_in_g);
    cudaFree(d_gpu_img_in_b);
    cudaFree(d_gpu_img_out_y);
    cudaFree(d_gpu_img_out_u);
    cudaFree(d_gpu_img_out_v);

    return 0;
}
