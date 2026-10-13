#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void GraphSum_backward(float* in_grad, float* out_grad, int* indptr, int* indices, int dim, int numNodes) {
    clock_t start = clock();

    for (int src = 0; src < numNodes; src++) {
        for (int i = indptr[src]; i < indptr[src + 1]; i++) {
            int dst = indices[i];
            float coef = 1.0f / sqrtf((indptr[src + 1] - indptr[src]) * (indptr[dst + 1] - indptr[dst]));
            for (int j = 0; j < dim; j++) {
                in_grad[src * dim + j] += coef * out_grad[dst * dim + j];
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char** argv) {
    int dim = 102400;
    int numNodes = 512;

    float* h_d_in_grad = (float*)malloc(numNodes * dim * sizeof(float));
    float* h_d_out_grad = (float*)malloc(numNodes * dim * sizeof(float));
    int* h_d_indptr = (int*)malloc((numNodes + 1) * sizeof(int));
    int* h_d_indices = (int*)malloc(numNodes * 20 * sizeof(int));

    srand(42);

    for (int i = 0; i < numNodes * dim; ++i) {
        h_d_in_grad[i] = (float)(rand() % 10);
        h_d_out_grad[i] = (float)(rand() % 10);
    }

    for (int i = 0; i < numNodes + 1; ++i) {
        h_d_indptr[i] = i * 20;
    }

    int num_indices = h_d_indptr[numNodes] - h_d_indptr[0];
    for (int i = 0; i < num_indices; ++i) {
        h_d_indices[i] = i % numNodes;
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    GraphSum_backward(h_d_in_grad, h_d_out_grad, h_d_indptr, h_d_indices, dim, numNodes);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_144_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < numNodes; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(file, "%.6f ", h_d_in_grad[i * dim + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_d_in_grad);
    free(h_d_out_grad);
    free(h_d_indptr);
    free(h_d_indices);
    return 0;
}