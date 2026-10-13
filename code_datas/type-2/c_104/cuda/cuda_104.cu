#include <stdio.h>

__global__ void cuda_SparseMatmul_forward_kernel(float* a_in, float* b_in, float* c_in, int* indptr, int* indices, int p) {
    int i = blockIdx.x;
    int k = threadIdx.x;

    for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
        int j = indices[jj];
        c_in[i * p + k] += a_in[jj] * b_in[j * p + k];
    }
}

int main() {
    // Set your desired parameters
    int numRows = 512;  // Set your desired value for numRows
    int numCols = 256;  // Set your desired value for numCols
    int numNonZeros = 1024;  // Set your desired value for numNonZeros

    // Allocate memory on the host
    float* h_a_in = (float*)malloc(numNonZeros * sizeof(float));
    float* h_b_in = (float*)malloc(numCols * p * sizeof(float));
    float* h_c_in = (float*)malloc(numRows * p * sizeof(float));
    int* h_indptr = (int*)malloc((numRows + 1) * sizeof(int));
    int* h_indices = (int*)malloc(numNonZeros * sizeof(int));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_a_in, * d_b_in, * d_c_in;
    int* d_indptr, * d_indices;
    cudaMalloc((void**)&d_a_in, numNonZeros * sizeof(float));
    cudaMalloc((void**)&d_b_in, numCols * p * sizeof(float));
    cudaMalloc((void**)&d_c_in, numRows * p * sizeof(float));
    cudaMalloc((void**)&d_indptr, (numRows + 1) * sizeof(int));
    cudaMalloc((void**)&d_indices, numNonZeros * sizeof(int));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize(numRows, 1, 1);
    dim3 blockSize(p, 1, 1);

    // Launch the CUDA kernel for sparse matrix multiplication
    cuda_SparseMatmul_forward_kernel<<<gridSize, blockSize>>>(d_a_in, d_b_in, d_c_in, d_indptr, d_indices, p);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_a_in);
    cudaFree(d_b_in);
    cudaFree(d_c_in);
    cudaFree(d_indptr);
    cudaFree(d_indices);

    // Free host memory
    free(h_a_in);
    free(h_b_in);
    free(h_c_in);
    free(h_indptr);
    free(h_indices);

    return 0;
}
