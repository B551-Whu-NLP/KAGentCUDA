#include <stdio.h>
#include <math.h>

__global__ void subsample_ind_and_labels_GPU(int* d_ind_sub, const int* d_ind, unsigned int* d_label_sub, const unsigned int* d_label, int n_out, float inv_sub_factor) {
    unsigned int ind_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind_out < n_out) {
        int ind_in = (int)floorf((float)(ind_out) * inv_sub_factor);
        d_ind_sub[ind_out] = d_ind[ind_in];
        d_label_sub[ind_out] = d_label[ind_in];
    }
}

int main() {
    // Set your desired parameters
    int n_out = 512;  // Set your desired value for n_out
    float inv_sub_factor = 0.5f;  // Set your desired value for inv_sub_factor

    // Allocate memory on the host
    int* h_ind = nullptr;  // Add initialization or copy data to h_ind
    unsigned int* h_label = nullptr;  // Add initialization or copy data to h_label

    // Allocate memory on the device
    int* d_ind, *d_ind_sub;
    unsigned int* d_label, *d_label_sub;
    cudaMalloc((void**)&d_ind, sizeof(int));  // Add appropriate size
    cudaMalloc((void**)&d_ind_sub, sizeof(int));  // Add appropriate size
    cudaMalloc((void**)&d_label, sizeof(unsigned int));  // Add appropriate size
    cudaMalloc((void**)&d_label_sub, sizeof(unsigned int));  // Add appropriate size

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((n_out + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for subsampling indices and labels
    subsample_ind_and_labels_GPU<<<gridSize, blockSize>>>(d_ind_sub, d_ind, d_label_sub, d_label, n_out, inv_sub_factor);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_ind);
    cudaFree(d_ind_sub);
    cudaFree(d_label);
    cudaFree(d_label_sub);

    // Free host memory
    // Add code to free host memory if needed

    return 0;
}
