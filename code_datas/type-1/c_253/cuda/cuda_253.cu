#include <stdio.h>

// CUDA kernel function
__global__ void multMat(int n, int* arrForce_d, int* arrDistance_d, int* arrAnswer_d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        arrAnswer_d[i] = arrForce_d[i] * arrDistance_d[i];
    }
}

int main() {
    // Array size
    int n = 1000; // Change this according to your requirements

    // Host arrays
    int* h_arrForce = (int*)malloc(n * sizeof(int));
    int* h_arrDistance = (int*)malloc(n * sizeof(int));
    int* h_arrAnswer = (int*)malloc(n * sizeof(int));

    // Initialize host input arrays
    for (int i = 0; i < n; ++i) {
        h_arrForce[i] = i;           // Example data, you can modify this accordingly
        h_arrDistance[i] = i * 2;    // Example data, you can modify this accordingly
    }

    // Device arrays
    int* d_arrForce;
    int* d_arrDistance;
    int* d_arrAnswer;
    cudaMalloc((void**)&d_arrForce, n * sizeof(int));
    cudaMalloc((void**)&d_arrDistance, n * sizeof(int));
    cudaMalloc((void**)&d_arrAnswer, n * sizeof(int));

    // Copy host input arrays to device
    cudaMemcpy(d_arrForce, h_arrForce, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrDistance, h_arrDistance, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((n + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    multMat<<<grid_size, block_size>>>(n, d_arrForce, d_arrDistance, d_arrAnswer);

    // Copy the result back to the host
    cudaMemcpy(h_arrAnswer, d_arrAnswer, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_arrAnswer[i]);
    }
    printf("\n");

    // Clean up
    free(h_arrForce);
    free(h_arrDistance);
    free(h_arrAnswer);
    cudaFree(d_arrForce);
    cudaFree(d_arrDistance);
    cudaFree(d_arrAnswer);

    return 0;
}
 
