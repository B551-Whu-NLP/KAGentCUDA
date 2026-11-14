#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void update_clusters(int n, int k, double* Cx, double* Cy, double* Cx_sum, double* Cy_sum, int* Csize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < k && Csize[index]) {
        Cx[index] = Cx_sum[index] / Csize[index];
        Cy[index] = Cy_sum[index] / Csize[index];
    }
}

int main() {
    // Set the parameters
    const int k = 10; // Number of clusters (change as needed)
    const int n = 1000; // Number of data points (change as needed)

    // Host arrays
    double* h_Cx = (double*)malloc(k * sizeof(double));
    double* h_Cy = (double*)malloc(k * sizeof(double));
    double* h_Cx_sum = (double*)malloc(k * sizeof(double));
    double* h_Cy_sum = (double*)malloc(k * sizeof(double));
    int* h_Csize = (int*)malloc(k * sizeof(int));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < k; ++i) {
        h_Cx[i] = i; // Example data, you can modify this accordingly
        h_Cy[i] = i;
        h_Cx_sum[i] = i;
        h_Cy_sum[i] = i;
        h_Csize[i] = i + 1;
    }

    // Device arrays
    double* d_Cx, * d_Cy, * d_Cx_sum, * d_Cy_sum;
    int* d_Csize;
    cudaMalloc((void**)&d_Cx, k * sizeof(double));
    cudaMalloc((void**)&d_Cy, k * sizeof(double));
    cudaMalloc((void**)&d_Cx_sum, k * sizeof(double));
    cudaMalloc((void**)&d_Cy_sum, k * sizeof(double));
    cudaMalloc((void**)&d_Csize, k * sizeof(int));

    // Copy host data to device
    cudaMemcpy(d_Cx, h_Cx, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cy, h_Cy, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cx_sum, h_Cx_sum, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cy_sum, h_Cy_sum, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Csize, h_Csize, k * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (k + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    update_clusters<<<blocksPerGrid, threadsPerBlock>>>(n, k, d_Cx, d_Cy, d_Cx_sum, d_Cy_sum, d_Csize);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_Cx, d_Cx, k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Cy, d_Cy, k * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Updated cluster centers:\n");
    for (int i = 0; i < k; ++i) {
        printf("Cluster %d: Cx=%.2f, Cy=%.2f\n", i, h_Cx[i], h_Cy[i]);
    }

    // Clean up
    free(h_Cx);
    free(h_Cy);
    free(h_Cx_sum);
    free(h_Cy_sum);
    free(h_Csize);
    cudaFree(d_Cx);
    cudaFree(d_Cy);
    cudaFree(d_Cx_sum);
    cudaFree(d_Cy_sum);
    cudaFree(d_Csize);

    return 0;
}
 
