#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void GatherKernel(const int* input, float* output, int input_size, const float* data, int count, int dim, int data_offset) {
    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_index < input_size * dim) {
        const int input_id = input[thread_index / dim];
        const int pos = thread_index % dim;

        if (input_id < count + data_offset && input_id >= data_offset) {
            output[thread_index] = data[input_id * dim + pos];
        }
    }
}

int main() {
    // Set the parameters
    const int input_size = 1000; // Change as needed
    const int count = 100; // Change as needed
    const int dim = 3; // Change as needed
    const int data_offset = 50; // Change as needed

    // Host arrays
    int* h_input = (int*)malloc(input_size * sizeof(int));
    float* h_output = (float*)malloc(input_size * dim * sizeof(float));
    float* h_data = (float*)malloc((count + data_offset) * dim * sizeof(float));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < input_size; ++i) {
        h_input[i] = i % (count + data_offset); // Example data, you can modify this accordingly
    }

    for (int i = 0; i < (count + data_offset) * dim; ++i) {
        h_data[i] = i; // Example data, you can modify this accordingly
    }

    // Device arrays
    int* d_input;
    float* d_output, * d_data;
    cudaMalloc((void**)&d_input, input_size * sizeof(int));
    cudaMalloc((void**)&d_output, input_size * dim * sizeof(float));
    cudaMalloc((void**)&d_data, (count + data_offset) * dim * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, (count + data_offset) * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (input_size * dim + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    GatherKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, input_size, d_data, count, dim, data_offset);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_output, d_output, input_size * dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Gathered data:\n");
    for (int i = 0; i < input_size; ++i) {
        printf("Input ID %d: ", h_input[i]);
        for (int j = 0; j < dim; ++j) {
            printf("%.2f ", h_output[i * dim + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_input);
    free(h_output);
    free(h_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_data);

    return 0;
}
 
