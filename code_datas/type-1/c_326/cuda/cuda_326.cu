#include <stdio.h>
#include <cuda_runtime.h>

__global__ void waterElevationToDepth(const int nx_, const int ny_, float *h_ptr_, int h_pitch_, float *Bm_ptr_, int Bm_pitch_) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (ti < nx_ && tj < ny_) {
        float *const h_row = (float *)((char *)h_ptr_ + h_pitch_ * tj);
        float *const Bm_row = (float *)((char *)Bm_ptr_ + Bm_pitch_ * tj);
        h_row[ti] -= Bm_row[ti];
    }
}

int main() {
    // Set array dimensions
    const int nx = 256;  // Set the appropriate value
    const int ny = 256;  // Set the appropriate value

    // Allocate host memory
    float *h_host, *Bm_host;
    size_t h_pitch, Bm_pitch;

    cudaMallocPitch((void **)&h_host, &h_pitch, nx * sizeof(float), ny);
    cudaMallocPitch((void **)&Bm_host, &Bm_pitch, nx * sizeof(float), ny);

    // Initialize input arrays (you may use your own initialization logic)
    // Note: You need to fill h_host and Bm_host with valid data

    // Allocate device memory
    float *h_device, *Bm_device;
    cudaMalloc((void **)&h_device, h_pitch * ny);
    cudaMalloc((void **)&Bm_device, Bm_pitch * ny);

    // Copy input arrays from host to device
    cudaMemcpy2D(h_device, h_pitch, h_host, h_pitch, nx * sizeof(float), ny, cudaMemcpyHostToDevice);
    cudaMemcpy2D(Bm_device, Bm_pitch, Bm_host, Bm_pitch, nx * sizeof(float), ny, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16);  // You may adjust the block size
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    waterElevationToDepth<<<gridSize, blockSize>>>(nx, ny, h_device, h_pitch / sizeof(float), Bm_device, Bm_pitch / sizeof(float));

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result array from device to host (if needed)

    // Cleanup
    cudaFree(h_device);
    cudaFree(Bm_device);
    cudaFreeHost(h_host);
    cudaFreeHost(Bm_host);

    return 0;
}
 
