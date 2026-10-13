#include <stdio.h>

// Define the CUDA kernel
__global__ void Match(float *P, float *Q, int q_points, int *idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float min = 100000;
    float d;
    float xp = P[0 + i * 3];
    float yp = P[1 + i * 3];
    float zp = P[2 + i * 3];
    float xq, yq, zq;
    int j;

    for (j = 0; j < q_points / 2; j++) {
        xq = Q[0 + j * 3];
        yq = Q[1 + j * 3];
        zq = Q[2 + j * 3];
        d = (xp - xq) * (xp - xq) + (yp - yq) * (yp - yq) + (zp - zq) * (zp - zq);
        if (d < min) {
            min = d;
            idx[i] = j;
        }
    }

    for (j = j; j < q_points; j++) {
        xq = Q[0 + j * 3];
        yq = Q[1 + j * 3];
        zq = Q[2 + j * 3];
        d = (xp - xq) * (xp - xq) + (yp - yq) * (yp - yq) + (zp - zq) * (zp - zq);
        if (d < min) {
            min = d;
            idx[i] = j;
        }
    }
}

int main() {
    // Example usage
    int q_points = 100; // Set your value of q_points accordingly
    float *P, *Q; // Assuming these arrays are allocated and initialized
    int *idx; // Assuming this array is allocated

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_P, *d_Q;
    int *d_idx;

    cudaMalloc((void **)&d_P, 3 * sizeof(float));
    cudaMalloc((void **)&d_Q, 3 * q_points * sizeof(float));
    cudaMalloc((void **)&d_idx, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_P, P, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Q, 3 * q_points * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = 1; // Assuming a single block for simplicity

    // Launch the CUDA kernel
    Match<<<blocksPerGrid, threadsPerBlock>>>(d_P, d_Q, q_points, d_idx);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_idx);

    return 0;
}
