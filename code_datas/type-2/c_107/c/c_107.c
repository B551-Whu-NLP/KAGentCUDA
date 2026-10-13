#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void SparseMatmul_backward(float* a, float* b_grad, float* c_grad, int* indptr, int* indices, int p, int size) {
    clock_t start = clock();

    for (int rep = 0; rep < 100; rep++) {
        for (int i = 0; i < size - 1; i++) {
            for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
                int j = indices[jj];
                for (int k = 0; k < p; k++) {
                    b_grad[j * p + k] += c_grad[i * p + k] * a[jj];
                }
            }
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char* argv[]) {
    const int size = 10000;
    const int p = 1000;
    float* h_a = (float*)malloc(size * p * sizeof(float));
    float* h_b = (float*)malloc(size * p * sizeof(float));
    float* h_c = (float*)malloc(size * p * sizeof(float));

    srand(42);

    for (int i = 0; i < size * p; i++) {
        h_a[i] = (float)(rand() % 100) / 10.0f;
        h_c[i] = (float)(rand() % 100) / 10.0f;
        h_b[i] = 0.0f;
    }

    int* h_indptr = (int*)malloc((size + 1) * sizeof(int));
    int* h_indices = (int*)malloc(size * sizeof(int));

    h_indptr[0] = 0;
    for (int i = 0; i < size; i++) {
        h_indptr[i + 1] = h_indptr[i] + 1;
        h_indices[i] = i;
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    SparseMatmul_backward(h_a, h_b, h_c, h_indptr, h_indices, p, size);

    if (verbose) {
        FILE* file = fopen("c_107_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.6f ", h_b[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_indptr);
    free(h_indices);
    return 0;
}