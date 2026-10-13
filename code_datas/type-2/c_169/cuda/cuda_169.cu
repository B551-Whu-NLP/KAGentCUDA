#include <stdio.h>

// Define the CUDA kernel
__global__ void fractal(const int width, const int frames, unsigned char *const pic) {
    const long i = threadIdx.x + blockIdx.x * (long)blockDim.x;

    if (i > width * width * frames) {
        return;
    }

    const float Delta = 0.00304f;
    const float xMid = -0.055846456f;
    const float yMid = -0.668311119f;

    const int frame = i / (width * width);
    float delta = Delta * powf(0.975f, frame);

    const int col = i % width;
    const float xMin = xMid - delta;
    const float yMin = yMid - delta;

    const float dw = 2.0f * delta / width;
    const int row = (i / width) % width;

    const float cy = yMin + row * dw;
    const float cx = xMin + col * dw;

    float x = cx;
    float y = cy;
    float x2, y2;
    int count = 256;

    do {
        x2 = x * x;
        y2 = y * y;
        y = 2.0 * x * y + cy;
        x = x2 - y2 + cx;
        count--;
    } while ((count > 0) && ((x2 + y2) <= 5.0));

    pic[frame * width * width + row * width + col] = (unsigned char)count;
}

int main() {
    // Example usage
    int width = 800;  // Set your value of width accordingly
    int frames = 100; // Set your value of frames accordingly
    unsigned char *pic; // Assuming this array is allocated and initialized

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    unsigned char *d_pic;
    cudaMalloc((void **)&d_pic, frames * width * width * sizeof(unsigned char));

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * width * frames + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    fractal<<<blocksPerGrid, threadsPerBlock>>>(width, frames, d_pic);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(pic, d_pic, frames * width * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pic);

    return 0;
}
