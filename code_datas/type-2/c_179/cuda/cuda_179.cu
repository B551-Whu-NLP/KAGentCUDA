#include <stdio.h>

// Define the CUDA kernel
__global__ void nlf_filter_down_backward(const int n, const float *bottom_data, const float *top_data,
                                         const float *temp_diff, const int channel,
                                         const int height, const int width, const int wsize,
                                         float *filters_diff) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    int step = height * width;
    int base = index / step * step * channel + index % step;
    int fbase = index / step * step * wsize + index % step;
    int row = index % step / width;
    int col = index % step % width;

    for (int i = 0; i < channel; i++) {
        filters_diff[fbase] += temp_diff[base + i * step] * bottom_data[base + i * step];

        if (row - 1 >= 0)
            filters_diff[fbase + step] += temp_diff[base + i * step] * top_data[base - width + i * step];
        else
            filters_diff[fbase + step] += temp_diff[base + i * step] * bottom_data[base + i * step];

        if (row - 1 >= 0 && col - 1 >= 0)
            filters_diff[fbase + 2 * step] +=
                temp_diff[base + i * step] * top_data[base - width - 1 + i * step];
        else
            filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

        if (row - 1 >= 0 && col + 1 < width)
            filters_diff[fbase + 3 * step] +=
                temp_diff[base + i * step] * top_data[base - width + 1 + i * step];
        else
            filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

        if (col - 1 >= 0)
            filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * top_data[base - 1 + i * step];
        else
            filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];
    }
}

int main() {
    // Example usage
    int n = 1000;  // Set your value of n accordingly
    int channel = 3;  // Set your value of channel accordingly
    int height = 64;  // Set your value of height accordingly
    int width = 64;  // Set your value of width accordingly
    int wsize = 5;  // Set your value of wsize accordingly

    float *bottom_data, *top_data, *temp_diff, *filters_diff;

    // Assuming these arrays are allocated and initialized
    // ...

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_bottom_data, *d_top_data, *d_temp_diff, *d_filters_diff;

    cudaMalloc((void **)&d_bottom_data, n * height * width * channel * sizeof(float));
    cudaMalloc((void **)&d_top_data, n * height * width * channel * sizeof(float));
    cudaMalloc((void **)&d_temp_diff, n * height * width * channel * sizeof(float));
    cudaMalloc((void **)&d_filters_diff, n * height * width * wsize * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_bottom_data, bottom_data, n * height * width * channel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_top_data, top_data, n * height * width * channel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_diff, temp_diff, n * height * width * channel * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    nlf_filter_down_backward<<<blocksPerGrid, threadsPerBlock>>>(n, d_bottom_data, d_top_data, d_temp_diff,
                                                                channel, height, width, wsize, d_filters_diff);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(filters_diff, d_filters_diff, n * height * width * wsize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_bottom_data);
    cudaFree(d_top_data);
    cudaFree
