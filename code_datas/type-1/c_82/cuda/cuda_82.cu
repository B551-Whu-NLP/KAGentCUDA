#include <stdio.h>

// CUDA kernel for filtering in FFT
__global__ void filterFFT(float* FFT, float* filter, int nxprj2, int nviews, float scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nviews && j < nxprj2) {
        FFT[i * nxprj2 + j] *= filter[i * nxprj2 + j] * scale;
    }
}

int main() {
    // Set your desired parameters
    int nxprj2 = 512;
    int nviews = 512;
    float scale = 0.5; // You can set your own value for scale

    // Allocate memory on the host
    float* h_FFT = (float*)malloc(nxprj2 * nviews * sizeof(float));
    float* h_filter = (float*)malloc(nxprj2 * nviews * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_FFT, * d_filter;
    cudaMalloc((void**)&d_FFT, nxprj2 * nviews * sizeof(float));
    cudaMalloc((void**)&d_filter, nxprj2 * nviews * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((nxprj2 + 15) / 16, (nviews + 15) / 16);
    dim3 blockSize(16, 16);

    // Launch the CUDA kernel for filtering in FFT
    filterFFT<<<gridSize, blockSize>>>(d_FFT, d_filter, nxprj2, nviews, scale);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_FFT);
    cudaFree(d_filter);

    // Free host memory
    free(h_FFT);
    free(h_filter);

    return 0;
}
