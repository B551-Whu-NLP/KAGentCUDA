#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void l2(int n, float* pred, float* truth, float* delta, float* error) {
    int i;
    for (i = 0; i < n; ++i) {
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

int main(int argc, char** argv) {
    int n = 1000000; // Increased from 1000 to 1000000
    float* h_pred = (float*)malloc(n * sizeof(float));
    float* h_truth = (float*)malloc(n * sizeof(float));
    float* h_delta = (float*)malloc(n * sizeof(float));
    float* h_error = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        h_pred[i] = i;
        h_truth[i] = 2 * i;
    }

    clock_t start = clock();

    // Repeat the computation multiple times to increase runtime
    for (int rep = 0; rep < 500; rep++) {
        l2(n, h_pred, h_truth, h_delta, h_error);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_266_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_pred[i]);
            }
            fprintf(file, "\nTruth Array:\n");
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_truth[i]);
            }
            fprintf(file, "\nDelta Array:\n");
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_delta[i]);
            }
            fprintf(file, "\nError Array:\n");
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_error[i]);
            }
            fclose(file);
        }
    }

    free(h_pred);
    free(h_truth);
    free(h_delta);
    free(h_error);

    return 0;
}