#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void residual(double* out, double* x, double* b, double* cotans, int* neighbors, double* diag, int meshStride, int n) {
    clock_t start = clock();
    
    for (int iter = 0; iter < 100000; iter++) {
        for (int i = 0; i < n; i++) {
            out[i] = diag[i] * x[i] - b[i];
            for (int iN = 0; iN < meshStride; ++iN) {
                int neighbor = neighbors[i * meshStride + iN];
                double weight = cotans[i * meshStride + iN];
                out[i] -= weight * x[neighbor];
            }
        }
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char** argv) {
    int n = 100000;
    int meshStride = 6;
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    double* h_out = (double*)malloc(n * sizeof(double));
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_b = (double*)malloc(n * sizeof(double));
    double* h_cotans = (double*)malloc(n * meshStride * sizeof(double));
    int* h_neighbors = (int*)malloc(n * meshStride * sizeof(int));
    double* h_diag = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        h_x[i] = i % 10;
        h_b[i] = i % 10;
        h_diag[i] = 1.0 + i % 5;
        for (int j = 0; j < meshStride; ++j) {
            h_neighbors[i * meshStride + j] = (i + j) % n;
            h_cotans[i * meshStride + j] = 0.1 * (i % 10);
        }
    }

    residual(h_out, h_x, h_b, h_cotans, h_neighbors, h_diag, meshStride, n);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_132_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.6f ", h_out[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }
    
    free(h_out);
    free(h_x);
    free(h_b);
    free(h_cotans);
    free(h_neighbors);
    free(h_diag);
    return 0;
}