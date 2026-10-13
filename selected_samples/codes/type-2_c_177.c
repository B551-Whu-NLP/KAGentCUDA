#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void conv2(float* A, float* kernel, int inputSize, int depth, int kernelSize, int stride, int pad, float* B, int outputSize) {
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) { // 增加外层循环以延长运行时间
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                int Ai = i * stride;
                int Aj = j * stride;
                int startk = (pad - Ai) < 0 ? 0 : pad - Ai;
                int endk = kernelSize < (inputSize + pad - Ai) ? kernelSize : (inputSize + pad - Ai);
                int startl = (pad - Aj) < 0 ? 0 : pad - Aj;
                int endl = kernelSize < (inputSize + pad - Aj) ? kernelSize : (inputSize + pad - Aj);
                float sum = 0;
                for (int d = 0; d < depth; d++) {
                    for (int k = startk; k < endk; k++) {
                        for (int l = startl; l < endl; l++) {
                            sum += A[d * inputSize * inputSize + (Ai + k - pad) * inputSize + Aj + l - pad] * kernel[d * kernelSize * kernelSize + k * kernelSize + l];
                        }
                    }
                    B[d * outputSize * outputSize + i * outputSize + j] = sum;
                }
                B[i * outputSize + j] = sum;
            }
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char* argv[]) {
    int inputSize = 50, depth = 10, kernelSize = 5, stride = 1, pad = 2;
    int outputSize = (inputSize - kernelSize + 2 * pad) / stride + 1;

    int A_size = depth * inputSize * inputSize;
    int kernel_size = depth * kernelSize * kernelSize;
    int B_size = depth * outputSize * outputSize;

    float* h_A = (float*)malloc(A_size * sizeof(float));
    float* h_kernel = (float*)malloc(kernel_size * sizeof(float));
    float* h_B = (float*)malloc(B_size * sizeof(float));

    for (int i = 0; i < A_size; i++) {
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < kernel_size; i++) {
        h_kernel[i] = 1.0f;
    }

    conv2(h_A, h_kernel, inputSize, depth, kernelSize, stride, pad, h_B, outputSize);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_177_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int d = 0; d < depth; d++) {
                fprintf(file, "Depth %d:\n", d);
                for (int i = 0; i < outputSize; i++) {
                    for (int j = 0; j < outputSize; j++) {
                        fprintf(file, "%.2f ", h_B[d * outputSize * outputSize + i * outputSize + j]);
                    }
                    fprintf(file, "\n");
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_A);
    free(h_kernel);
    free(h_B);
    return 0;
}