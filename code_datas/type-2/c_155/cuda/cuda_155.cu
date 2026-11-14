#include <stdio.h>

// CUDA kernel
__global__ void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch,
                                 int w1, int h1, int c1, float* add, int w2, int h2, int c2, float* out) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (id >= size)
        return;

    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
    int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));

    atomicAdd(&out[out_index], add[add_index]);
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    float* h_add = /* Your initialization */;
    float* h_out = /* Your initialization */;

    float* d_add, *d_out;

    cudaMalloc((void**)&d_add, /* Size in bytes */);
    cudaMalloc((void**)&d_out, /* Size in bytes */);

    // Copy host memory to device
    cudaMemcpy(d_add, h_add, /* Size in bytes */, cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(/* Set your block size */);
    dim3 gridSize(/* Set your grid size */);

    int size = /* Set size */;
    int minw = /* Set minw */;
    int minh = /* Set minh */;
    int minc = /* Set minc */;
    int stride = /* Set stride */;
    int sample = /* Set sample */;
    int batch = /* Set batch */;
    int w1 = /* Set w1 */;
    int h1 = /* Set h1 */;
    int c1 = /* Set c1 */;
    int w2 = /* Set w2 */;
    int h2 = /* Set h2 */;
    int c2 = /* Set c2 */;

    shortcut_kernel<<<gridSize, blockSize>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, d_add, w2, h2, c2, d_out);

    // Copy device memory back to host
    cudaMemcpy(h_out, d_out, /* Size in bytes */, cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    cudaFree(d_add);
    cudaFree(d_out);

    return 0;
}
