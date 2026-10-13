#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void invalidateFlow_kernel(float *modFlowX, float *modFlowY, const float *constFlowX, const float *constFlowY, int width, int height, float cons_thres) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int ind = y * width + x;
        float mFX = modFlowX[ind];
        float mFY = modFlowY[ind];
        float cFX = constFlowX[ind];
        float cFY = constFlowY[ind];

        float err = (mFX - cFX) * (mFX - cFX) + (mFY - cFY) * (mFY - cFY);

        if (err > cons_thres) {
            mFX = 0;
            mFY = 0;
        }

        modFlowX[ind] = mFX;
        modFlowY[ind] = mFY;
    }
}

int main() {
    const int width = 128;
    const int height = 64;
    const float cons_thres = 0.1; // Adjust the threshold as needed

    float *modFlowX, *modFlowY, *constFlowX, *constFlowY;

    size_t flow_size = width * height * sizeof(float);

    // Allocate host memory
    modFlowX = (float *)malloc(flow_size);
    modFlowY = (float *)malloc(flow_size);
    constFlowX = (float *)malloc(flow_size);
    constFlowY = (float *)malloc(flow_size);

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < width * height; ++i) {
        modFlowX[i] = static_cast<float>(i);
        modFlowY[i] = static_cast<float>(i * 2);
        constFlowX[i] = static_cast<float>(i * 3);
        constFlowY[i] = static_cast<float>(i * 4);
    }

    // Allocate device memory
    float *d_modFlowX, *d_modFlowY, *d_constFlowX, *d_constFlowY;
    cudaMalloc((void **)&d_modFlowX, flow_size);
    cudaMalloc((void **)&d_modFlowY, flow_size);
    cudaMalloc((void **)&d_constFlowX, flow_size);
    cudaMalloc((void **)&d_constFlowY, flow_size);

    // Copy data from host to device
    cudaMemcpy(d_modFlowX, modFlowX, flow_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_modFlowY, modFlowY, flow_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_constFlowX, constFlowX, flow_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_constFlowY, constFlowY, flow_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Launch the kernel
    invalidateFlow_kernel<<<grid_size, block_size>>>(d_modFlowX, d_modFlowY, d_constFlowX, d_constFlowY, width, height, cons_thres);

    // Copy data from device to host
    cudaMemcpy(modFlowX, d_modFlowX, flow_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(modFlowY, d_modFlowY, flow_size, cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_modFlowX);
    cudaFree(d_modFlowY);
    cudaFree(d_constFlowX);
    cudaFree(d_constFlowY);
    free(modFlowX);
    free(modFlowY);
    free(constFlowX);
    free(constFlowY);

    return 0;
}
 
