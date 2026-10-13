#include <stdio.h>

// CUDA kernel
__global__ void add_sources_d(const float* const model, float* wfp, const float* const source_amplitude,
                               const int* const sources_z, const int* const sources_x, const int nz, const int nx,
                               const int nt, const int ns, const int it) {
    int x = threadIdx.x;
    int b = blockIdx.x;
    int i = sources_z[b * ns + x] * nx + sources_x[b * ns + x];
    int ib = b * nz * nx + i;
    wfp[ib] += source_amplitude[b * ns * nt + x * nt + it] * model[i];
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int nz = 100;    // Replace with your actual nz
    int nx = 100;    // Replace with your actual nx
    int nt = 100;    // Replace with your actual nt
    int ns = 10;     // Replace with your actual ns

    float* h_model = (float*)malloc(nz * nx * sizeof(float));
    float* h_wfp = (float*)malloc(nz * nx * ns * sizeof(float));
    float* h_source_amplitude = (float*)malloc(ns * nt * sizeof(float));
    int* h_sources_z = (int*)malloc(ns * sizeof(int));
    int* h_sources_x = (int*)malloc(ns * sizeof(int));

    float* d_model, * d_wfp, * d_source_amplitude;
    int* d_sources_z, * d_sources_x;

    cudaMalloc((void**)&d_model, nz * nx * sizeof(float));
    cudaMalloc((void**)&d_wfp, nz * nx * ns * sizeof(float));
    cudaMalloc((void**)&d_source_amplitude, ns * nt * sizeof(float));
    cudaMalloc((void**)&d_sources_z, ns * sizeof(int));
    cudaMalloc((void**)&d_sources_x, ns * sizeof(int));

    // Copy host memory to device
    cudaMemcpy(d_model, h_model, nz * nx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wfp, h_wfp, nz * nx * ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_amplitude, h_source_amplitude, ns * nt * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sources_z, h_sources_z, ns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sources_x, h_sources_x, ns * sizeof(int), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(16); // Adjust block dimensions based on your requirements
    dim3 gridSize(ns);

    int it = 0; // Replace with your actual it

    add_sources_d<<<gridSize, blockSize>>>(d_model, d_wfp, d_source_amplitude, d_sources_z, d_sources_x, nz, nx, nt, ns, it);

    // Copy device memory back to host
    cudaMemcpy(h_wfp, d_wfp, nz * nx * ns * sizeof(float), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_model);
    free(h_wfp);
    free(h_source_amplitude);
    free(h_sources_z);
    free(h_sources_x);

    cudaFree(d_model);
    cudaFree(d_wfp);
    cudaFree(d_source_amplitude);
    cudaFree(d_sources_z);
    cudaFree(d_sources_x);

    return 0;
}
