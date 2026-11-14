#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cuda_Adam_step_kernel(float *grad, float *data, float *m, float *v, bool decay, float weight_decay, float beta1, float beta2, float eps, float step_size, int varsize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= varsize)
        return;

    float g = grad[i];
    if (decay)
        g += weight_decay * data[i];

    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
    data[i] -= step_size * m[i] / (sqrtf(v[i]) + eps);
}

int main() {
    // Set array dimensions
    const int varsize = 1000;  // Set the appropriate value

    // Allocate host memory
    float *grad_host = (float *)malloc(varsize * sizeof(float));
    float *data_host = (float *)malloc(varsize * sizeof(float));
    float *m_host = (float *)malloc(varsize * sizeof(float));
    float *v_host = (float *)malloc(varsize * sizeof(float));

    // Initialize input arrays (you may use your own initialization logic)
    // Note: You need to fill grad_host, data_host, m_host, and v_host with valid data

    // Allocate device memory
    float *grad_device, *data_device, *m_device, *v_device;
    cudaMalloc((void **)&grad_device, varsize * sizeof(float));
    cudaMalloc((void **)&data_device, varsize * sizeof(float));
    cudaMalloc((void **)&m_device, varsize * sizeof(float));
    cudaMalloc((void **)&v_device, varsize * sizeof(float));

    // Copy input arrays from host to device
    cudaMemcpy(grad_device, grad_host, varsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_device, data_host, varsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_device, m_host, varsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v_device, v_host, varsize * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((varsize + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    cuda_Adam_step_kernel<<<gridSize, blockSize>>>(grad_device, data_device, m_device, v_device, true, 0.001, 0.9, 0.999, 1e-8, 0.001, varsize);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result array from device to host (if needed)

    // Cleanup
    cudaFree(grad_device);
    cudaFree(data_device);
    cudaFree(m_device);
    cudaFree(v_device);
    free(grad_host);
    free(data_host);
    free(m_host);
    free(v_host);

    return 0;
}
 
