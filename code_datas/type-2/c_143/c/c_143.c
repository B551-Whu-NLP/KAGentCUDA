#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sgemm_kernel(const float* host_inputArray1, const float* host_inputArray2, float* host_inputArray3, int M, int N, int K, float alpha, float beta) {
    for (int row = 0; row < M; row++) {
        for (int column = 0; column < N; column++) {
            float element_c = 0.f;
            for (int e = 0; e < K; e++) {
                element_c += host_inputArray1[row * K + e] * host_inputArray2[e * N + column];
            }
            host_inputArray3[row * N + column] = alpha * element_c + beta * host_inputArray3[row * N + column];
        }
    }
}

int main(int argc, char* argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int M = 1024;
    int N = 1024;
    int K = 512;
    float* h_inputArray1 = (float*)malloc(M * K * sizeof(float));
    float* h_inputArray2 = (float*)malloc(K * N * sizeof(float));
    float* h_inputArray3 = (float*)malloc(M * N * sizeof(float));

    srand(0);
    for (int i = 0; i < M * K; ++i) {
        h_inputArray1[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < K * N; ++i) {
        h_inputArray2[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < M * N; ++i) {
        h_inputArray3[i] = (float)(rand() % 10);
    }

    clock_t start_time = clock();

    sgemm_kernel(h_inputArray1, h_inputArray2, h_inputArray3, M, N, K, 1.0, 0.0);

    clock_t end_time = clock();

    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", elapsed_time * 1000);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_143_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    fprintf(file, "%.2f ", h_inputArray3[i * N + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_inputArray1);
    free(h_inputArray2);
    free(h_inputArray3);
    return 0;
}