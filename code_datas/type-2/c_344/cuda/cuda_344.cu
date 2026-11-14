#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

__global__ void maxValExtract(float *normM_c, float *normM1_c, long int image_size, float *d_projections, int *d_index, float a) {
    __shared__ int pos[BLOCK_SIZE * 2];
    __shared__ float val[BLOCK_SIZE * 2];

    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    float faux, faux2;
    faux = ((a - normM_c[id]) / a);
    faux2 = ((a - normM_c[id + blockDim.x]) / a);

    if (id < image_size && faux <= 1.0e-6) {
        val[tid] = normM1_c[id];
        pos[tid] = id;
    } else {
        val[tid] = -1;
    }

    if (id + blockDim.x < image_size && faux2 <= 1.0e-6) {
        val[tid + blockDim.x] = normM1_c[id + blockDim.x];
        pos[tid + blockDim.x] = id + blockDim.x;
    } else {
        val[tid + blockDim.x] = -1;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
        if (tid < s) {
            if (val[tid] <= val[tid + s]) {
                val[tid] = val[tid + s];
                pos[tid] = pos[tid + s];
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        d_projections[blockIdx.x] = val[0];
        d_index[blockIdx.x] = (int)pos[0];
    }

    __syncthreads();
}

int main() {
    const long int image_size = 1024; // Adjust the size based on your data
    const float a = 1.0; // Adjust the value based on your requirement

    float *normM_c, *normM1_c, *d_projections;
    int *d_index;

    size_t size = image_size * sizeof(float);

    // Allocate host memory
    normM_c = (float *)malloc(size * 2); // Assuming normM_c is a 1D array with size * 2 elements
    normM1_c = (float *)malloc(size);
    d_projections = (float *)malloc((image_size / BLOCK_SIZE) * sizeof(float));
    d_index = (int *)malloc((image_size / BLOCK_SIZE) * sizeof(int));

    // Initialize host data (you may need to modify this based on your use case)
    for (long int i = 0; i < size * 2; ++i) {
        normM_c[i] = static_cast<float>(i);
    }

    for (long int i = 0; i < size; ++i) {
        normM1_c[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_normM_c, *d_normM1_c;
    int *d_d_index;
    cudaMalloc((void **)&d_normM_c, size * 2);
    cudaMalloc((void **)&d_normM1_c, size);
    cudaMalloc((void **)&d_d_index, (image_size / BLOCK_SIZE) * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_normM_c, normM_c, size * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_normM1_c, normM1_c, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((image_size / BLOCK_SIZE + 1), 1);

    // Launch the kernel
    maxValExtract<<<grid_size, block_size>>>(d_normM_c, d_normM1_c, image_size, d_projections, d_index, a);

    // Copy data from device to host
    cudaMemcpy(d_projections, d_projections, (image_size / BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_index, d_d_index, (image_size / BLOCK_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_normM_c);
    cudaFree(d_normM1_c);
    cudaFree(d_d_index);
    free(normM_c);
    free(normM1_c);
    free(d_projections);
    free(d_index);

    return 0;
}
 
