#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void multiply_matrices(float* A, float* B, float* C, int N) {
    int i, j, k;
    float sum;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0;
            for (k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 1000; 
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 10 + 1); 
        B[i] = (float)(i % 10 + 2); 
    }

    clock_t start = clock();

    multiply_matrices(A, B, C, N);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_311_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    fprintf(fp, "%.2f ", C[i * N + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(A);
    free(B);
    free(C);

    return 0;
}