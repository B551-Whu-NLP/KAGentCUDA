#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mat_mul_kernel(int *m_A, int *m_B, int *m_C, int A_rows, int A_cols, int B_rows, int B_cols) {
    int sum = 0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < A_rows && col < B_cols) {
        for (int i = 0; i < A_cols; i++) {
            sum += m_A[row * A_cols + i] * m_B[i * B_cols + col];
        }
        m_C[row * B_cols + col] = sum;
    }
}

int main() {
    // Set matrix dimensions
    int A_rows = 4;  // Set the appropriate value
    int A_cols = 3;  // Set the appropriate value
    int B_rows = 3;  // Set the appropriate value
    int B_cols = 5;  // Set the appropriate value

    // Allocate host memory
    int *h_A, *h_B, *h_C;
    h_A = (int *)malloc(A_rows * A_cols * sizeof(int));
    h_B = (int *)malloc(B_rows * B_cols * sizeof(int));
    h_C = (int *)malloc(A_rows * B_cols * sizeof(int));

    // Initialize matrices (you may use your own initialization logic)
    for (int i = 0; i < A_rows * A_cols; i++) {
        h_A[i] = i + 1;
    }

    for (int i = 0; i < B_rows * B_cols; i++) {
        h_B[i] = i + 1;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, A_rows * A_cols * sizeof(int));
    cudaMalloc((void **)&d_B, B_rows * B_cols * sizeof(int));
    cudaMalloc((void **)&d_C, A_rows * B_cols * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, A_rows * A_cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_rows * B_cols * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16);  // You may adjust the block size
    dim3 gridSize((B_cols + blockSize.x - 1) / blockSize.x, (A_rows + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    mat_mul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, A_rows, A_cols, B_rows, B_cols);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, A_rows * B_cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result matrix (you may modify this part based on your needs)
    printf("Result Matrix:\n");
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            printf("%d ", h_C[i * B_cols + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
 
