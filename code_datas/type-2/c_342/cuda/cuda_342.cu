#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 4

__global__ void opL21(float *vec, float *vec1, long depth, long rows, long cols) {
    unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
    unsigned long long i = z * rows * cols + y * cols + x;
    unsigned long long j = z * rows * cols + x;
    unsigned long size2d = cols;
    unsigned long size3d = depth * rows * cols + rows * cols + cols;

    if (x >= cols || y >= rows || z >= depth)
        return;

    if (i + cols + 1 >= size3d)
        return;

    vec[i + cols] = 0.25 * (vec1[i + 1] + vec1[i] + vec1[i + cols + 1] + vec1[i + cols]);

    if (j + 1 >= size2d)
        return;

    vec[j] = (vec1[j] + vec1[j + 1]) / 4;
}

int main() {
    const long depth = 64;
    const long rows = 128;
    const long cols = 64;

    float *vec, *vec1;

    size_t size3d = depth * rows * cols * sizeof(float);
    size_t size2d = rows * cols * sizeof(float);

    // Allocate host memory
    vec = (float *)malloc(size3d);
    vec1 = (float *)malloc(size3d);

    // Initialize host data (you may need to modify this based on your use case)
    for (long i = 0; i < depth * rows * cols; ++i) {
        vec[i] = static_cast<float>(i);
        vec1[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_vec, *d_vec1;
    cudaMalloc((void **)&d_vec, size3d);
    cudaMalloc((void **)&d_vec1, size3d);

    // Copy data from host to device
    cudaMemcpy(d_vec, vec, size3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec1, vec1, size3d, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid_size((cols + block_size.x - 1) / block_size.x, (rows + block_size.y - 1) / block_size.y, (depth + block_size.z - 1) / block_size.z);

    // Launch the kernel
    opL21<<<grid_size, block_size>>>(d_vec, d_vec1, depth, rows, cols);

    // Copy data from device to host
    cudaMemcpy(vec, d_vec, size3d, cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_vec);
    cudaFree(d_vec1);
    free(vec);
    free(vec1);

    return 0;
}
 
