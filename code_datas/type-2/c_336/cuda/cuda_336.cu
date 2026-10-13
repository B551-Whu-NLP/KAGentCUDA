 #include <iostream>

// Define CUDA kernel
__global__ void conv1x1(int input_channels, int input_size, int n, float *input_im, float *filter_weight, float *filter_bias, float *output_im) {
    int filter_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (filter_index < n) {
        filter_weight += filter_index * input_channels;
        float bias = filter_bias[filter_index];
        output_im += filter_index * input_size * input_size;

        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < input_size; j++) {
                float tmp = bias;

                for (int k = 0; k < input_channels; k++) {
                    tmp += input_im[k * input_size * input_size + i * input_size + j] * filter_weight[k];
                }

                output_im[i * input_size + j] = (tmp > 0.0) ? tmp : 0.0;
            }
        }
    }
}

int main() {
    // Example usage
    int input_channels = 3;
    int input_size = 5;
    int n = 64;  // Number of filters
    int total_elements = n * input_size * input_size;

    // Allocate device memory
    float *d_input_im, *d_filter_weight, *d_filter_bias, *d_output_im;
    cudaMalloc((void**)&d_input_im, input_channels * input_size * input_size * sizeof(float));
    cudaMalloc((void**)&d_filter_weight, n * input_channels * sizeof(float));
    cudaMalloc((void**)&d_filter_bias, n * sizeof(float));
    cudaMalloc((void**)&d_output_im, total_elements * sizeof(float));

    // Initialize data on host (you should replace this with your actual data)
    float *h_input_im = new float[input_channels * input_size * input_size];
    float *h_filter_weight = new float[n * input_channels];
    float *h_filter_bias = new float[n];
    float *h_output_im = new float[total_elements];

    // Copy data from host to device
    cudaMemcpy(d_input_im, h_input_im, input_channels * input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_weight, h_filter_weight, n * input_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_bias, h_filter_bias, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    // Launch the CUDA kernel
    conv1x1<<<gridSize, blockSize>>>(input_channels, input_size, n, d_input_im, d_filter_weight, d_filter_bias, d_output_im);

    // Copy the result back to host
    cudaMemcpy(h_output_im, d_output_im, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
    delete[] h_input_im;
    delete[] h_filter_weight;
    delete[] h_filter_bias;
    delete[] h_output_im;

    cudaFree(d_input_im);
    cudaFree(d_filter_weight);
    cudaFree(d_filter_bias);
    cudaFree(d_output_im);

    return 0;
}

