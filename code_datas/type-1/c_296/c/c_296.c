#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void logistic_x_ent(int n, float* pred, float* truth, float* delta, float* error) {
    for (int i = 0; i < n; ++i) {
        float t = truth[i];
        float p = pred[i];
        error[i] = -t * logf(p) - (1 - t) * logf(1 - p);
        delta[i] = t - p;
    }
}

int main(int argc, char** argv) {
    const int n = 1000;
    float* h_pred = (float*)malloc(n * sizeof(float));
    float* h_truth = (float*)malloc(n * sizeof(float));
    float* h_delta = (float*)malloc(n * sizeof(float));
    float* h_error = (float*)malloc(n * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < n; ++i) {
        h_pred[i] = 0.7f;
        h_truth[i] = 0.5f;
    }

    clock_t start = clock();

    logistic_x_ent(n, h_pred, h_truth, h_delta, h_error);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_296_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "Element %d: Delta=%f, Error=%f\n", i, h_delta[i], h_error[i]);
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