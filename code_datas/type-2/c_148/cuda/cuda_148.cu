#include <stdio.h>

// CUDA kernel
__global__ void normalizacion(float* image_c, int bands, long int image_size, float* normM_c, float* normM1_c) {
    long int j, i;
    float norm_val = 0, aux = 0, pixel = 0;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < image_size) {
        for (j = 0; j < bands; j++) {
            norm_val += image_c[j * image_size + i];
        }

        norm_val = 1.0 / (norm_val + 1.0e-16);

        for (j = 0; j < bands; j++) {
            pixel = image_c[j * image_size + i] * norm_val;
            image_c[j * image_size + i] = pixel;
            aux += pixel * pixel;
        }

        normM_c[i] = aux;
        normM1_c[i] = aux;
    }
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    float* h_image = /* Your initialization */;
    float* h_normM = /* Your initialization */;
    float* h_normM1 = /* Your initialization */;

    float* d_image, *d_normM, *d_normM1;

    cudaMalloc((void**)&d_image, /* Size in bytes */);
    cudaMalloc((void**)&d_normM, /* Size in bytes */);
    cudaMalloc((void**)&d_normM1, /* Size in bytes */);

    // Copy host memory to device
    cudaMemcpy(d_image, h_image, /* Size in bytes */, cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(/* Set your block size */);
    dim3 gridSize(/* Set your grid size */);

    normalizacion<<<gridSize, blockSize>>>(d_image, /* Pass your parameters */, d_normM, d_normM1);

    // Copy device memory back to host
    cudaMemcpy(h_normM, d_normM, /* Size in bytes */, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normM1, d_normM1, /* Size in bytes */, cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    cudaFree(d_image);
    cudaFree(d_normM);
    cudaFree(d_normM1);

    return 0;
}
