#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void MatrMult(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rowsA = 1024;
    int colsA = 512;
    int colsB = 256;

    float* h_Ad = (float*)malloc(rowsA * colsA * sizeof(float));
    float* h_Bd = (float*)malloc(colsA * colsB * sizeof(float));
    float* h_Cd = (float*)malloc(rowsA * colsB * sizeof(float));

    for (int i = 0; i < rowsA * colsA; ++i) {
        h_Ad[i] = i % 100 / 10.0f;
    }
    for (int i = 0; i < colsA * colsB; ++i) {
        h_Bd[i] = i % 100 / 10.0f;
    }
    for (int i = 0; i < rowsA * colsB; ++i) {
        h_Cd[i] = 0;
    }

    clock_t start = clock();

    // 重复计算以延长运行时间
    for (int rep = 0; rep < 100; ++rep) {
        MatrMult(h_Ad, h_Bd, h_Cd, rowsA, colsA, colsB);
    }

    clock_t end = clock();
    
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_135_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < 1000; ++j) {
                    fprintf(file, "%0.2f ", h_Cd[i * colsB + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_Ad);
    free(h_Bd);
    free(h_Cd);
    return 0;
}