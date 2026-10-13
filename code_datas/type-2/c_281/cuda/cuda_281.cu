#include <stdio.h>

// CUDA kernel function
__global__ void matrixTranspose(int* in_mat, int* out_mat, int dim_rows, int dim_cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < dim_rows && col < dim_cols) {
        unsigned int new_pos = col * dim_rows + row;
        out_mat[new_pos] = in_mat[row * dim_cols + col];
    }
}

int main() {
    // Matrix dimensions
    int dim_rows = 3;  // Number of rows
    int dim_cols = 4;  // Number of columns

    // Host arrays
    int* h_in_mat = (int*)malloc(dim_rows * dim_cols * sizeof(int));
    int* h_out_mat = (int*)malloc(dim_cols * dim_rows * sizeof(int));

    // Initialize host input matrix
    for (int i = 0; i < dim_rows * dim_cols; ++i) {
        h_in_mat[i] = i;  // Example data for in_mat, you can modify this accordingly
    }

    // Device arrays
    int* d_in_mat;
    int* d_out_mat;
    cudaMalloc((void**)&d_in_mat, dim_rows * dim_cols * sizeof(int));
    cudaMalloc((void**)&d_out_mat, dim_cols * dim_rows * sizeof(int));

    // Copy host input matrix to device
    cudaMemcpy(d_in_mat, h_in_mat, dim_rows * dim_cols * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block_size(2, 2);  // Change this according to your requirements
    dim3 grid_size((dim_cols + block_size.x - 1) / block_size.x, (dim_rows + block_size.y - 1) / block_size.y);

    // Launch the CUDA kernel
    matrixTranspose<<<grid_size, block_size>>>(d_in_mat, d_out_mat, dim_rows, dim_cols);

    // Copy the result back to the host
    cudaMemcpy(h_out_mat, d_out_mat, dim_cols * dim_rows * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Original Matrix:\n");
    for (int i = 0; i < dim_rows; ++i) {
        for (int j = 0; j < dim_cols; ++j) {
            printf("%d ", h_in_mat[i * dim_cols + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Matrix:\n");
    for (int i = 0; i < dim_cols; ++i) {
        for (int j = 0; j < dim_rows; ++j) {
            printf("%d ", h_out_mat[i * dim_rows + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_in_mat);
    free(h_out_mat);
    cudaFree(d_in_mat);
    cudaFree(d_out_mat);

    return 0;
}
 
