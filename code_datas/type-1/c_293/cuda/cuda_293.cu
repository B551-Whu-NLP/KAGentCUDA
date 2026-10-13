#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void sumAndScale(float* noiseVariance, float* diffMag2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int batchJump = i * 347;
    float temp = 0;

    for (int sumIndex = 0; sumIndex < 347; sumIndex++)
        temp += diffMag2[batchJump + sumIndex];

    temp = 0.00161812 * temp;
    noiseVariance[i] = temp;
}

int main() {
    // Set the parameters
    const int n = 1000; // Change as needed

    // Host arrays
    float* h_diffMag2 = (float*)malloc(n * 347 * sizeof(float));
    float* h_noiseVariance = (float*)malloc(n * sizeof(float));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < n * 347; ++i) {
        h_diffMag2[i] = static_cast<float>(i % 100); // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_diffMag2;
    float* d_noiseVariance;

    cudaMalloc((void**)&d_diffMag2, n * 347 * sizeof(float));
    cudaMalloc((void**)&d_noiseVariance, n * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_diffMag2, h_diffMag2, n * 347 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_noiseVariance, 0, n * sizeof(float));

    // Define block and grid dimensions
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch the CUDA kernel
    sumAndScale<<<blocksPerGrid, threadsPerBlock>>>(d_noiseVariance, d_diffMag2, n);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_noiseVariance, d_noiseVariance, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Noise Variance:\n");
    for (int i = 0; i < n; ++i) {
        printf("Element %d: %f\n", i, h_noiseVariance[i]);
    }

    // Clean up
    free(h_diffMag2);
    free(h_noiseVariance);
    cudaFree(d_diffMag2);
    cudaFree(d_noiseVariance);

    return 0;
}
 
