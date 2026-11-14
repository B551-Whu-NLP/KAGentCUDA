#include <stdio.h>

// CUDA kernel function
__global__ void get_conf_inds(const float* mlvl_conf, const float conf_thr, int* conf_inds, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dims) {
        conf_inds[tid] = (mlvl_conf[tid] >= conf_thr) ? 1 : -1;
    }
}

int main() {
    // Array size
    int dims = 10; // Change this according to your requirements

    // Host arrays
    float* h_mlvl_conf = (float*)malloc(dims * sizeof(float));
    int* h_conf_inds = (int*)malloc(dims * sizeof(int));

    // Initialize host input array (confidence values)
    for (int i = 0; i < dims; ++i) {
        h_mlvl_conf[i] = i * 0.1; // Example data for mlvl_conf, you can modify this accordingly
    }

    // Device arrays
    float* d_mlvl_conf;
    int* d_conf_inds;
    cudaMalloc((void**)&d_mlvl_conf, dims * sizeof(float));
    cudaMalloc((void**)&d_conf_inds, dims * sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_mlvl_conf, h_mlvl_conf, dims * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((dims + block_size - 1) / block_size, 1);

    // Set confidence threshold
    float conf_thr = 0.5; // Change this according to your requirements

    // Launch the CUDA kernel
    get_conf_inds<<<grid_size, block_size>>>(d_mlvl_conf, conf_thr, d_conf_inds, dims);

    // Copy the result back to the host
    cudaMemcpy(h_conf_inds, d_conf_inds, dims * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("mlvl_conf Array:\n");
    for (int i = 0; i < dims; ++i) {
        printf("%.2f ", h_mlvl_conf[i]);
    }

    printf("\nconf_inds Array:\n");
    for (int i = 0; i < dims; ++i) {
        printf("%d ", h_conf_inds[i]);
    }
    printf("\n");

    // Clean up
    free(h_mlvl_conf);
    free(h_conf_inds);
    cudaFree(d_mlvl_conf);
    cudaFree(d_conf_inds);

    return 0;
}
 
