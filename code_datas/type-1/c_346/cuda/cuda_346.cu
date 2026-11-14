#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void RyT(float *R, float *T, float *P, float *Q, int num_points) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_points) {
        Q[0 + i * 3] = R[0 + 0 * 3] * P[0 + i * 3] + R[0 + 1 * 3] * P[1 + i * 3] + R[0 + 2 * 3] * P[2 + i * 3] + T[0];
        Q[1 + i * 3] = R[1 + 0 * 3] * P[0 + i * 3] + R[1 + 1 * 3] * P[1 + i * 3] + R[1 + 2 * 3] * P[2 + i * 3] + T[1];
        Q[2 + i * 3] = R[2 + 0 * 3] * P[0 + i * 3] + R[2 + 1 * 3] * P[1 + i * 3] + R[2 + 2 * 3] * P[2 + i * 3] + T[2];
    }
}

int main() {
    const int num_points = 1024; // Adjust the number of points based on your data

    float *R, *T, *P, *Q;

    // Allocate host memory
    R = (float *)malloc(9 * sizeof(float));
    T = (float *)malloc(3 * sizeof(float));
    P = (float *)malloc(3 * num_points * sizeof(float));
    Q = (float *)malloc(3 * num_points * sizeof(float));

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < 9; ++i) {
        R[i] = static_cast<float>(i);
    }

    for (int i = 0; i < 3; ++i) {
        T[i] = static_cast<float>(i * 2);
    }

    for (int i = 0; i < 3 * num_points; ++i) {
        P[i] = static_cast<float>(i * 3);
    }

    // Allocate device memory
    float *d_R, *d_T, *d_P, *d_Q;
    cudaMalloc((void **)&d_R, 9 * sizeof(float));
    cudaMalloc((void **)&d_T, 3 * sizeof(float));
    cudaMalloc((void **)&d_P, 3 * num_points * sizeof(float));
    cudaMalloc((void **)&d_Q, 3 * num_points * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_R, R, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, T, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, 3 * num_points * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((num_points + block_size.x - 1) / block_size.x);

    // Launch the kernel
    RyT<<<grid_size, block_size>>>(d_R, d_T, d_P, d_Q, num_points);

    // Copy data from device to host
    cudaMemcpy(Q, d_Q, 3 * num_points * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_R);
    cudaFree(d_T);
    cudaFree(d_P);
    cudaFree(d_Q);
    free(R);
    free(T);
    free(P);
    free(Q);

    return 0;
}
 
