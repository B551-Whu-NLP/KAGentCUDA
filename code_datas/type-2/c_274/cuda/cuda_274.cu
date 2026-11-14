#include <stdio.h>

// CUDA kernel function
__global__ void avgpool(int n, float* input_im, float* output_im) {
    int class_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (class_index < n) {
        input_im += 169 * class_index;
        float tmp = 0.0f;

        for (int i = 0; i < 169; i++) {
            tmp += input_im[i];
        }

        output_im[class_index] = tmp / 169.0;
    }
}

int main() {
    // Number of classes
    int num_classes = 10;  // Change this according to your requirements

    // Host arrays
    float* h_input_im = (float*)malloc(169 * num_classes * sizeof(float));
    float* h_output_im = (float*)malloc(num_classes * sizeof(float));

    // Initialize host input array
    for (int i = 0; i < 169 * num_classes; ++i) {
        h_input_im[i] = static_cast<float>(i);  // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_input_im;
    float* d_output_im;
    cudaMalloc((void**)&d_input_im, 169 * num_classes * sizeof(float));
    cudaMalloc((void**)&d_output_im, num_classes * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_input_im, h_input_im, 169 * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (num_classes + block_size - 1) / block_size;

    // Launch the CUDA kernel
    avgpool<<<grid_size, block_size>>>(num_classes, d_input_im, d_output_im);

    // Copy the result back to the host
    cudaMemcpy(h_output_im, d_output_im, num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Results:\n");
    for (int i = 0; i < num_classes; ++i) {
        printf("%f ", h_output_im[i]);
    }
    printf("\n");

    // Clean up
    free(h_input_im);
    free(h_output_im);
    cudaFree(d_input_im);
    cudaFree(d_output_im);

    return 0;
}
 
