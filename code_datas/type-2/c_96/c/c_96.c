#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sgemm(float *C, float *A, float *B, long size) {
    for (long i = 0; i < size; i++) {
        for (long k = 0; k < size; k++) {
            for (long j = 0; j < size; j++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    long size = 1000;
    float* h_C = (float*)malloc(size * size * sizeof(float));
    float* h_A = (float*)malloc(size * size * sizeof(float));
    float* h_B = (float*)malloc(size * size * sizeof(float));
    
    for (long i = 0; i < size; i++) {
        for (long j = 0; j < size; j++) {
            h_A[i * size + j] = 1.0f;
            h_B[i * size + j] = 1.0f;
            h_C[i * size + j] = 0.0f;
        }
    }

    clock_t start = clock();
    sgemm(h_C, h_A, h_B, size);
    clock_t end = clock();

    double time_taken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_96_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (long i = 0; i < 50; i++) {
                for (long j = 0; j < 50; j++) {
                    fprintf(fp, "%.2f ", h_C[i * size + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }


    free(h_C);
    free(h_A);
    free(h_B);

    return 0;
}