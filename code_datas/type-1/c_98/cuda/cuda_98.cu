#include <stdio.h>

// CUDA kernel for aging simulation
__global__ void envejecer_kernel(int* estado, int* edad, int* pupacion, int* N_mobil, int dia) {
    int N = N_mobil[0];
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N) {
        if (dia < 80 || dia > 320) {
            if (edad[id] > pupacion[id])
                edad[id]++;
        } else {
            edad[id]++;
        }
    }
}

int main() {
    // Set your desired parameters
    int N = 512; // Set your desired value for N
    int dia = 150; // Set your desired value for dia

    // Allocate memory on the host
    int* h_estado = (int*)malloc(N * sizeof(int));
    int* h_edad = (int*)malloc(N * sizeof(int));
    int* h_pupacion = (int*)malloc(N * sizeof(int));
    int* h_N_mobil = (int*)malloc(sizeof(int));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    int* d_estado, * d_edad, * d_pupacion, * d_N_mobil;
    cudaMalloc((void**)&d_estado, N * sizeof(int));
    cudaMalloc((void**)&d_edad, N * sizeof(int));
    cudaMalloc((void**)&d_pupacion, N * sizeof(int));
    cudaMalloc((void**)&d_N_mobil, sizeof(int));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((N + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for aging simulation
    envejecer_kernel<<<gridSize, blockSize>>>(d_estado, d_edad, d_pupacion, d_N_mobil, dia);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_estado);
    cudaFree(d_edad);
    cudaFree(d_pupacion);
    cudaFree(d_N_mobil);

    // Free host memory
    free(h_estado);
    free(h_edad);
    free(h_pupacion);
    free(h_N_mobil);

    return 0;
}
