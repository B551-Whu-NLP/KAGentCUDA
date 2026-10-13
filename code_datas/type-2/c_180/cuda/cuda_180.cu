#include <stdio.h>

// Define the CUDA kernel
__global__ void nlf_up_forward(const int n, const float *filters, const int channel,
                               const int height, const int width, const int wsize,
                               float *top_data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    int step = height * width;
    int base = index * step;
    int fbase = index / channel * wsize * step;

    for (int row = height - 1; row >= 0; row--) {
        for (int col = width - 1; col >= 0; col--) {
            float temp = 0;
            int r = row;
            int c = col;
            int shift = 0 * step + row * width + col;
            temp += top_data[base + r * width + c] * filters[fbase + shift];

            r = row + 1;
            c = col;
            shift = 1 * step + row * width + col;
            if (r < height)
                temp += top_data[base + r * width + c] * filters[fbase + shift];
            else
                temp += top_data[base + row * width + col] * filters[fbase + shift];

            r = row + 1;
            c = col - 1;
            shift = 2 * step + row * width + col;
            if (r < height && c >= 0)
                temp += top_data[base + r * width + c] * filters[fbase + shift];
            else
                temp += top_data[base + row * width + col] * filters[fbase + shift];

            r = row + 1;
            c = col + 1;
            shift = 3 * step + row * width + col;
            if (r < height && c < width)
                temp += top_data[base + r * width + c] * filters[fbase + shift];
            else
                temp += top_data[base + row * width + col] * filters[fbase + shift];

            r = row;
            c = col + 1;
            shift = 4 * step + row * width + col;
            if (c < width)
                temp += top_data[base + r * width + c] * filters[fbase + shift];
            else
                temp += top_data[base + row * width + col] * filters[fbase + shift];

            top_data[base + row * width + col] = temp;
        }
    }
}

int main() {
    // Example usage
    int n = 1000;  // Set your value of n accordingly
    int channel = 3;  // Set your value of channel accordingly
    int height = 64;  // Set your value of height accordingly
    int width = 64;  // Set your value of width accordingly
    int wsize = 5;  // Set your value of wsize accordingly

    float *filters, *top_data;

    // Assuming these arrays are allocated and initialized
    // ...

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_filters, *d_top_data;

    cudaMalloc((void **)&d_filters, n * height * width * channel * wsize * sizeof(float));
    cudaMalloc((void **)&d_top_data, n * height * width * channel * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_filters, filters, n * height * width * channel * wsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_top_data, top_data, n * height * width * channel * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    nlf_up_forward<<<blocksPerGrid, threadsPerBlock>>>(n, d_filters, channel, height, width, wsize, d_top_data);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(top_data, d_top_data, n * height * width * channel * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_filters);
    cudaFree(d_top_data);

    return 0;
}
