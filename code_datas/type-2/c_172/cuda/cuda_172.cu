#include <stdio.h>

// Define the CUDA kernel
__global__ void col2im_gpu_kernel(const int n, const float *data_col, const int height, const int width, const int ksize,
                                  const int pad, const int stride, const int height_col, const int width_col,
                                  float *data_im) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < n; index += blockDim.x * gridDim.x) {
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);

        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }

        data_im[index] += val;
    }
}

int main() {
    // Example usage
    int n = 1000; // Set your value of n accordingly
    int height = 64; // Set your value of height accordingly
    int width = 64; // Set your value of width accordingly
    int ksize = 3; // Set your value of ksize accordingly
    int pad = 1; // Set your value of pad accordingly
    int stride = 1; // Set your value of stride accordingly
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    float *data_col; // Assuming this array is allocated and initialized
    float *data_im; // Assuming this array is allocated and initialized

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_data_col, *d_data_im;

    cudaMalloc((void **)&d_data_col, n * sizeof(float));
    cudaMalloc((void **)&d_data_im, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data_col, data_col, n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    col2im_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_data_col, height, width, ksize, pad, stride,
                                                           height_col, width_col, d_data_im);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(data_im, d_data_im, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data_col);
    cudaFree(d_data_im);

    return 0;
}
