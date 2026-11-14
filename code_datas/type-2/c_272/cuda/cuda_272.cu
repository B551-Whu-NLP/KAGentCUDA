#include <stdio.h>

// Device function for parallel reduction
__device__ float sumreduce(float in) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    sdata[tid] = in;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    return sdata[0];
}

// CUDA kernel function
__global__ void reduceKernel(float* input, float* output, int size) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data to shared memory
    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();

    // Perform parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result to output array
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    // Array size
    int size = 1024;  // Change this according to your requirements

    // Host arrays
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output = (float*)malloc(size * sizeof(float));

    // Initialize host input array
    for (int i = 0; i < size; ++i) {
        h_input[i] = 1.0f;  // Example data for input, you can modify this accordingly
    }

    // Device arrays
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Launch the CUDA kernel with shared memory
    reduceKernel<<<grid_size, block_size, block_size * sizeof(float)>>>(d_input, d_output, size);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    float final_result = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        final_result += h_output[i];
    }

    // Display the result
    printf("Final Result: %.2f\n", final_result);

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
 
