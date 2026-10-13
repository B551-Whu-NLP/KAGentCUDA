#include <stdio.h>

// CUDA kernel function
__global__ void kernelIsFirst(int* head, int* first_pts, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        if (head[i] == 1)
            first_pts[i] = i;
        else
            first_pts[i] = 0;
    }
}

int main() {
    // Vector size
    int n = 10; // Change this according to your requirements

    // Host arrays
    int* h_head = (int*)malloc(n * sizeof(int));
    int* h_first_pts = (int*)malloc(n * sizeof(int));

    // Initialize host input array
    for (int i = 0; i < n; ++i) {
        h_head[i] = (i % 2 == 0) ? 1 : 0;  // Example data, you can modify this accordingly
    }

    // Device arrays
    int* d_head;
    int* d_first_pts;
    cudaMalloc((void**)&d_head, n * sizeof(int));
    cudaMalloc((void**)&d_first_pts, n * sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_head, h_head, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((n + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    kernelIsFirst<<<grid_size, block_size>>>(d_head, d_first_pts, n);

    // Copy the result back to the host
    cudaMemcpy(h_first_pts, d_first_pts, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("First Points: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_first_pts[i]);
    }
    printf("\n");

    // Clean up
    free(h_head);
    free(h_first_pts);
    cudaFree(d_head);
    cudaFree(d_first_pts);

    return 0;
}
 
