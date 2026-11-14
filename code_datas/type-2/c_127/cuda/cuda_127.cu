#include <stdio.h>

// CUDA kernel
__global__ void Backwardsub(double* U, double* RES, double* UN, double* UE, double* LPR, int NI, int NJ, int End, int J, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int IJ = ((End - i) * NI) + (J - (End - i));
        RES[IJ] = RES[IJ] - UN[IJ] * RES[IJ + 1] - UE[IJ] * RES[IJ + NJ];
        U[IJ] = U[IJ] + RES[IJ];
    }
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int NI = 100; // Replace with your actual values
    int NJ = 100;
    int End = 50;
    int J = 25;
    int n = 10;

    double* h_U = (double*)malloc(NI * NJ * sizeof(double));
    double* h_RES = (double*)malloc(NI * NJ * sizeof(double));
    double* h_UN = (double*)malloc(NI * NJ * sizeof(double));
    double* h_UE = (double*)malloc(NI * NJ * sizeof(double));
    double* h_LPR = (double*)malloc(NI * NJ * sizeof(double));

    double* d_U, * d_RES, * d_UN, * d_UE, * d_LPR;
    cudaMalloc((void**)&d_U, NI * NJ * sizeof(double));
    cudaMalloc((void**)&d_RES, NI * NJ * sizeof(double));
    cudaMalloc((void**)&d_UN, NI * NJ * sizeof(double));
    cudaMalloc((void**)&d_UE, NI * NJ * sizeof(double));
    cudaMalloc((void**)&d_LPR, NI * NJ * sizeof(double));

    // Copy host memory to device
    cudaMemcpy(d_U, h_U, NI * NJ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RES, h_RES, NI * NJ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_UN, h_UN, NI * NJ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_UE, h_UE, NI * NJ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPR, h_LPR, NI * NJ * sizeof(double), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(256); // Adjust block dimensions based on your requirements
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    Backwardsub<<<gridSize, blockSize>>>(d_U, d_RES, d_UN, d_UE, d_LPR, NI, NJ, End, J, n);

    // Copy device memory back to host
    cudaMemcpy(h_U, d_U, NI * NJ * sizeof(double), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_U);
    free(h_RES);
    free(h_UN);
    free(h_UE);
    free(h_LPR);
    cudaFree(d_U);
    cudaFree(d_RES);
    cudaFree(d_UN);
    cudaFree(d_UE);
    cudaFree(d_LPR);

    return 0;
}
