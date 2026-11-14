#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void pad_input(float *f_in, float *f_out, int H, int W, int D, int pad) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    int new_H = H + 2 * pad;
    int new_W = W + 2 * pad;

    int i = dep * new_H * new_W + col * new_W + row;
    int j = dep * H * W + (col - pad) * W + (row - pad);

    if (col < new_H && row < new_W && dep < D) {
        if ((col < pad || col > H + pad - 1) || (row < pad || row > W + pad - 1)) {
            f_out[i] = 0;
        } else {
            f_out[i] = f_in[j];
        }
    }
}

int main() {
    const int H = 128;
    const int W = 128;
    const int D = 64;
    const int pad = 2;

    float *h_input, *h_output;
    float *d_input, *d_output;

    size_t input_size = H * W * D * sizeof(float);
    size_t output_size = (H + 2 * pad) * (W + 2 * pad) * D * sizeof(float);

    // Allocate host memory
    h_input = (float *)malloc(input_size);
    h_output = (float *)malloc(output_size);

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < H * W * D; ++i) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, output_size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((H + block_size.x - 1) / block_size.x, (W + block_size.y - 1) / block_size.y, (D + block_size.z - 1) / block_size.z);

    // Launch the kernel
    pad_input<<<grid_size, block_size>>>(d_input, d_output, H, W, D, pad);

    // Copy data from device to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
 
