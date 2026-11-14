#include <stdio.h>

// CUDA kernel function
__global__ void zero_centroid_vals(int k, double* Cx_sum, double* Cy_sum, int* Csize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < k) {
        Cx_sum[index] = 0.0;
        Cy_sum[index] = 0.0;
        Csize[index] = 0;
    }
}

int main() {
    // Number of centroids (k)
    int k = 10; // Change this according to your requirements

    // Host arrays
    double* h_Cx_sum = (double*)malloc(k * sizeof(double));
    double* h_Cy_sum = (double*)malloc(k * sizeof(double));
    int* h_Csize = (int*)malloc(k * sizeof(int));

    // Device arrays
    double* d_Cx_sum;
    double* d_Cy_sum;
    int* d_Csize;
    cudaMalloc((void**)&d_Cx_sum, k * sizeof(double));
    cudaMalloc((void**)&d_Cy_sum, k * sizeof(double));
    cudaMalloc((void**)&d_Csize, k * sizeof(int));

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((k + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    zero_centroid_vals<<<grid_size, block_size>>>(k, d_Cx_sum, d_Cy_sum, d_Csize);

    // Copy the result back to the host
    cudaMemcpy(h_Cx_sum, d_Cx_sum, k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Cy_sum, d_Cy_sum, k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Csize, d_Csize, k * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result:\n");
    for (int i = 0; i < k; ++i) {
        printf("Centroid %d: Cx_sum=%f, Cy_sum=%f, Csize=%d\n", i, h_Cx_sum[i], h_Cy_sum[i], h_Csize[i]);
    }

    // Clean up
    free(h_Cx_sum);
    free(h_Cy_sum);
    free(h_Csize);
    cudaFree(d_Cx_sum);
    cudaFree(d_Cy_sum);
    cudaFree(d_Csize);

    return 0;
}
 
