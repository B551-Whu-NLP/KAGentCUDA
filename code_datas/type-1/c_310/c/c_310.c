#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void smooth_l1(int n, float* pred, float* truth, float* delta, float* error) {
    int i;

    for (i = 0; i < n; ++i) {
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);

        if (abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2 * abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

int main(int argc, char *argv[]) {
    int n = 1000000;
    float *pred = (float *)malloc(n * sizeof(float));
    float *truth = (float *)malloc(n * sizeof(float));
    float *delta = (float *)malloc(n * sizeof(float));
    float *error = (float *)malloc(n * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < n; i++) {
        pred[i] = i * 0.001 + 1.0;
        truth[i] = i * 0.001 + 1.5;
    }

    clock_t start = clock();
    smooth_l1(n, pred, truth, delta, error);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char output_filename[256];
        sprintf(output_filename, "%s_output.txt", argv[0]);
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", error[i]);
            }
            fprintf(fp, "\nTime: %.2f(ms)\n", time_taken);
            fclose(fp);
        }
    }

    free(pred);
    free(truth);
    free(delta);
    free(error);

    return 0;
}