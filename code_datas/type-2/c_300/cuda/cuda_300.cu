#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int width) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float pvalue = 0;

    for (int k = 0; k < width; ++k) {
        float Mdelement = Md[ty * width + k];
        float Ndelement = Nd[ty * width + k];
        pvalue += Mdelement * Ndelement;
    }

    Pd[ty * width + tx] = pvalue;
}

int main() {
    // Set your problem dimensions
    const int width = 4;  // Set your actual matrix width

    // Allocate host memory
    float *h_Md = (float *)malloc(width * width * sizeof(float));
    float *h_Nd = (float *)malloc(width * width * sizeof(float));
    float *h_Pd = (float *)malloc(width * width * sizeof(float));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < width * width; i++) {
        h_Md[i] = (float)rand() / RAND_MAX;
        h_Nd[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_Md, *d_Nd, *d_Pd;
    cudaMalloc((void **)&d_Md, width * width * sizeof(float));
    cudaMalloc((void **)&d_Nd, width * width * sizeof(float));
    cudaMalloc((void **)&d_Pd, width * width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_Md, h_Md, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Nd, h_Nd, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(width, width);
    dim3 gridSize(1, 1);

    // Launch the kernel
    MatrixMulKernel<<<gridSize, blockSize>>>(d_Md, d_Nd, d_Pd, width);

    // Copy result back to host
    cudaMemcpy(h_Pd, d_Pd, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Result Matrix:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", h_Pd[i * width + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(h_Md);
    free(h_Nd);
    free(h_Pd);
    cudaFree(d_Md);
    cudaFree(d_Nd);
    cudaFree(d_Pd);

    return 0;
}
 
