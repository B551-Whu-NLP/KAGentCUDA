#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void convertInstanceToLabel_Kernel(unsigned short* d_outputLabel, const unsigned char* d_inputInstance,
                                              const unsigned short* d_instanceToLabel, unsigned int width,
                                              unsigned int height) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        d_outputLabel[y * width + x] = d_instanceToLabel[d_inputInstance[y * width + x]];
    }
}

int main() {
    // Set the parameters
    const unsigned int width = 512;  // Change as needed
    const unsigned int height = 512; // Change as needed

    // Host arrays
    unsigned char* h_inputInstance = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned short* h_instanceToLabel = (unsigned short*)malloc(256 * sizeof(unsigned short)); // Adjust size as needed
    unsigned short* h_outputLabel = (unsigned short*)malloc(width * height * sizeof(unsigned short));

    // Initialize host arrays (example data, modify as needed)
    for (unsigned int i = 0; i < width * height; ++i) {
        h_inputInstance[i] = static_cast<unsigned char>(i % 256); // Example data, you can modify this accordingly
    }

    for (int i = 0; i < 256; ++i) {
        h_instanceToLabel[i] = static_cast<unsigned short>(i); // Example data, you can modify this accordingly
    }

    // Device arrays
    unsigned char* d_inputInstance;
    unsigned short* d_instanceToLabel;
    unsigned short* d_outputLabel;

    cudaMalloc((void**)&d_inputInstance, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_instanceToLabel, 256 * sizeof(unsigned short));
    cudaMalloc((void**)&d_outputLabel, width * height * sizeof(unsigned short));

    // Copy host data to device
    cudaMemcpy(d_inputInstance, h_inputInstance, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_instanceToLabel, h_instanceToLabel, 256 * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemset(d_outputLabel, 0, width * height * sizeof(unsigned short));

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the CUDA kernel
    convertInstanceToLabel_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_outputLabel, d_inputInstance, d_instanceToLabel,
                                                                      width, height);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_outputLabel, d_outputLabel, width * height * sizeof(unsigned short), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Output Label:\n");
    for (unsigned int i = 0; i < width * height; ++i) {
        printf("Element %u: %u\n", i, h_outputLabel[i]);
    }

    // Clean up
    free(h_inputInstance);
    free(h_instanceToLabel);
    free(h_outputLabel);
    cudaFree(d_inputInstance);
    cudaFree(d_instanceToLabel);
    cudaFree(d_outputLabel);

    return 0;
}
 
