#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void downsampleCuda(float* I, float* Q, unsigned int numDownsampledSamples,
                                float* downsampled_I, float* downsampled_Q, unsigned int factor) {
    int sampleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (sampleIndex < numDownsampledSamples) {
        unsigned int absoluteIndex = sampleIndex * factor;
        downsampled_I[sampleIndex] = I[absoluteIndex];
        downsampled_Q[sampleIndex] = Q[absoluteIndex];
    }
}

int main() {
    // Set the parameters
    const unsigned int numDownsampledSamples = 1024; // Change as needed
    const unsigned int factor = 2;                   // Change as needed

    // Host arrays
    float* h_I = (float*)malloc(numDownsampledSamples * factor * sizeof(float));
    float* h_Q = (float*)malloc(numDownsampledSamples * factor * sizeof(float));
    float* h_downsampled_I = (float*)malloc(numDownsampledSamples * sizeof(float));
    float* h_downsampled_Q = (float*)malloc(numDownsampledSamples * sizeof(float));

    // Initialize host arrays (example data, modify as needed)
    for (unsigned int i = 0; i < numDownsampledSamples * factor; ++i) {
        h_I[i] = static_cast<float>(i); // Example data, you can modify this accordingly
        h_Q[i] = static_cast<float>(i); // Example data, you can modify this accordingly
    }

    // Device arrays
    float *d_I, *d_Q, *d_downsampled_I, *d_downsampled_Q;

    cudaMalloc((void**)&d_I, numDownsampledSamples * factor * sizeof(float));
    cudaMalloc((void**)&d_Q, numDownsampledSamples * factor * sizeof(float));
    cudaMalloc((void**)&d_downsampled_I, numDownsampledSamples * sizeof(float));
    cudaMalloc((void**)&d_downsampled_Q, numDownsampledSamples * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_I, h_I, numDownsampledSamples * factor * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, numDownsampledSamples * factor * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((numDownsampledSamples + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch the CUDA kernel
    downsampleCuda<<<blocksPerGrid, threadsPerBlock>>>(d_I, d_Q, numDownsampledSamples,
                                                        d_downsampled_I, d_downsampled_Q, factor);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_downsampled_I, d_downsampled_I, numDownsampledSamples * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_downsampled_Q, d_downsampled_Q, numDownsampledSamples * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Downsampled I and Q:\n");
    for (unsigned int i = 0; i < numDownsampledSamples; ++i) {
        printf("Element %u: I=%f, Q=%f\n", i, h_downsampled_I[i], h_downsampled_Q[i]);
    }

    // Clean up
    free(h_I);
    free(h_Q);
    free(h_downsampled_I);
    free(h_downsampled_Q);
    cudaFree(d_I);
    cudaFree(d_Q);
    cudaFree(d_downsampled_I);
    cudaFree(d_downsampled_Q);

    return 0;
}
 
