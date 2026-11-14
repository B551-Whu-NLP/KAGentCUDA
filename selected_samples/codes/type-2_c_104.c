#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void SparseMatmul_forward(float* a, float* b, float* c, int* indptr, int* indices, int p, int size) {
    clock_t start = clock();

    for (int iter = 0; iter < 100; iter++) {
        for (int i = 0; i < size; i++) {      
            for (int k = 0; k < p; k++) {   
                for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) { 
                    int j = indices[jj];
                    c[i * p + k] += a[jj] * b[j * p + k];
                }
            }
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    int n = 10000;
    int numNonZeros = 50000;
    int p = 10000;

    float* h_a_in = (float*)malloc(numNonZeros * sizeof(float));
    float* h_b_in = (float*)malloc(n * p * sizeof(float));
    float* h_c_in = (float*)malloc(n * p * sizeof(float));
    int* h_indptr = (int*)malloc((n + 1) * sizeof(int));
    int* h_indices = (int*)malloc(numNonZeros * sizeof(int));

    srand(42);
    for (int i = 0; i < numNonZeros; i++) {
        h_a_in[i] = (float)(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < n * p; i++) {
        h_b_in[i] = (float)(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < n * p; i++) {
        h_c_in[i] = 0;
    }
    for (int i = 0; i <= n; i++) {
        h_indptr[i] = i * (numNonZeros / n);
    }
    for (int i = 0; i < numNonZeros; i++) {
        h_indices[i] = rand() % n;
    }

    SparseMatmul_forward(h_a_in, h_b_in, h_c_in, h_indptr, h_indices, p, n);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_104_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.6f ", h_c_in[i * p + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_a_in);
    free(h_b_in);
    free(h_c_in);
    free(h_indptr);
    free(h_indices);
    return 0;
}