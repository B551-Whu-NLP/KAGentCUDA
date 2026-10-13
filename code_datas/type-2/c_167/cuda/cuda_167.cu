#include <stdio.h>

// Define the CUDA kernel
__global__ void im2col_gpu_kernel(const int n, const float *data_im, const int height, const int width, const int ksize,
                                  const int pad, const int stride, const int height_col, const int width_col,
                                  float *data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < n; index += blockDim.x * gridDim.x) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;

        float *data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;

        const float *data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;

        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

int main() {
    // Example usage
    int n = 1000;       // Set your value of n accordingly
    int height = 32;    // Set your value of height accordingly
    int width = 32;     // Set your value of width accordingly
    int ksize = 3;      // Set your value of ksize accordingly
    int pad = 1;        // Set your value of pad accordingly
    int stride = 1;     // Set your value of stride accordingly
    int height_col = 30; // Set your value of height_col accordingly
    int width_col = 30;  // Set your value of width_col accordingly
    float *data_im, *data_col; // Assuming these arrays are allocated and initialized

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_data_im, *d_data_col;
    cudaMalloc((void **)&d_data_im, height * width * sizeof(float));
    cudaMalloc((void **)&d_data_col, height_col * width_col * ksize * ksize * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data_im, data_im, height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    im2col_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_data_im, height, width, ksize, pad, stride,
                                                          height_col, width_col, d_data_col);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(data_col, d_data_col, height_col * width_col * ksize * ksize * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data_im);
    cudaFree(d_data_col);

    return 0;
}
