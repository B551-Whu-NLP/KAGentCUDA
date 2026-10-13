#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void waterDepthToElevation(const int nx_, const int ny_, float *w_ptr_, int w_pitch_, float *h_ptr_, int h_pitch_, float *Bm_ptr_, int Bm_pitch_) {
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (ti < nx_ && tj < ny_) {
        float *const h_row = (float *)((char *)h_ptr_ + h_pitch_ * tj);
        float *const Bm_row = (float *)((char *)Bm_ptr_ + Bm_pitch_ * tj);
        float *const w_row = (float *)((char *)w_ptr_ + w_pitch_ * tj);

        w_row[ti] = h_row[ti] + Bm_row[ti];
    }
}

int main() {
    const int nx = 128;
    const int ny = 64;

    float *h_ptr, *Bm_ptr, *w_ptr;
    int h_pitch, Bm_pitch, w_pitch;

    size_t h_pitch_bytes = nx * sizeof(float);
    size_t Bm_pitch_bytes = nx * sizeof(float);
    size_t w_pitch_bytes = nx * sizeof(float);

    // Allocate host memory
    h_ptr = (float *)malloc(ny * h_pitch_bytes);
    Bm_ptr = (float *)malloc(ny * Bm_pitch_bytes);
    w_ptr = (float *)malloc(ny * w_pitch_bytes);

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < nx * ny; ++i) {
        h_ptr[i] = static_cast<float>(i);
        Bm_ptr[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    cudaMallocPitch((void **)&h_ptr, &h_pitch, nx * sizeof(float), ny);
    cudaMallocPitch((void **)&Bm_ptr, &Bm_pitch, nx * sizeof(float), ny);
    cudaMallocPitch((void **)&w_ptr, &w_pitch, nx * sizeof(float), ny);

    // Copy data from host to device
    cudaMemcpy2D(h_ptr, h_pitch, h_ptr, h_pitch_bytes, nx * sizeof(float), ny, cudaMemcpyHostToDevice);
    cudaMemcpy2D(Bm_ptr, Bm_pitch, Bm_ptr, Bm_pitch_bytes, nx * sizeof(float), ny, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);

    // Launch the kernel
    waterDepthToElevation<<<grid_size, block_size>>>(nx, ny, w_ptr, w_pitch / sizeof(float), h_ptr, h_pitch / sizeof(float), Bm_ptr, Bm_pitch / sizeof(float));

    // Copy data from device to host
    cudaMemcpy2D(w_ptr, w_pitch_bytes, w_ptr, w_pitch, nx * sizeof(float), ny, cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(h_ptr);
    cudaFree(Bm_ptr);
    cudaFree(w_ptr);
    free(h_ptr);
    free(Bm_ptr);
    free(w_ptr);

    return 0;
}
 
