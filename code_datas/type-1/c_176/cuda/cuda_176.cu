#include <stdio.h>

// Define the CUDA kernel
__global__ void decode(const float *anchor, const float *locData, float *predictBox,
                       int dims, float scaleClamp, int batchSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    for (int i = 0; i < batchSize; i++) {
        float anchorW = anchor[i * dims * 4 + tid * 4 + 2] - anchor[i * dims * 4 + tid * 4];
        float anchorH = anchor[i * dims * 4 + tid * 4 + 3] - anchor[i * dims * 4 + tid * 4 + 1];
        float anchorCx = anchor[i * dims * 4 + tid * 4] + 0.5 * anchorW;
        float anchorCy = anchor[i * dims * 4 + tid * 4 + 1] + 0.5 * anchorH;

        float dx = locData[i * dims * 4 + tid * 4];
        float dy = locData[i * dims * 4 + tid * 4 + 1];
        float dw = locData[i * dims * 4 + tid * 4 + 2];
        float dh = locData[i * dims * 4 + tid * 4 + 3];

        if (dw > scaleClamp) {
            dw = scaleClamp;
        }

        if (dh > scaleClamp) {
            dh = scaleClamp;
        }

        float preCx = dx * anchorW + anchorCx;
        float preCy = dy * anchorH + anchorCy;
        float preW = anchorW * 0.5;
        float preH = anchorH * 0.5;

        predictBox[i * dims * 4 + tid * 4] = preCx - 0.5 * preW;
        predictBox[i * dims * 4 + tid * 4 + 1] = preCy - 0.5 * preH;
        predictBox[i * dims * 4 + tid * 4 + 2] = preCx + 0.5 * preW;
        predictBox[i * dims * 4 + tid * 4 + 3] = preCy + 0.5 * preH;
    }
}

int main() {
    // Example usage
    int dims = 1000;  // Set your value of dims accordingly
    float scaleClamp = 1.0;  // Set your value of scaleClamp accordingly
    int batchSize = 1;  // Set your value of batchSize accordingly

    float *anchor, *locData, *predictBox;

    // Assuming these arrays are allocated and initialized
    // ...

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_anchor, *d_locData, *d_predictBox;

    cudaMalloc((void **)&d_anchor, dims * 4 * batchSize * sizeof(float));
    cudaMalloc((void **)&d_locData, dims * 4 * batchSize * sizeof(float));
    cudaMalloc((void **)&d_predictBox, dims * 4 * batchSize * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_anchor, anchor, dims * 4 * batchSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_locData, locData, dims * 4 * batchSize * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (dims + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    decode<<<blocksPerGrid, threadsPerBlock>>>(d_anchor, d_locData, d_predictBox, dims, scaleClamp, batchSize);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(predictBox, d_predictBox, dims * 4 * batchSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_anchor);
    cudaFree(d_locData);
    cudaFree(d_predictBox);

    return 0;
}
