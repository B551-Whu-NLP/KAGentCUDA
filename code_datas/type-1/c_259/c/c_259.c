#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

double compareDoubleArrayToThreshold(double* d_in, int* d_out, int length, double threshold) {
    clock_t start = clock();

    for (int idx = 0; idx < length; idx++) {
        double abs_value = (d_in[idx] > 0) ? d_in[idx] : -d_in[idx];
        d_out[idx] = (abs_value < threshold) ? 1 : 0;
    }

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char* argv[]) {
    int length = 10000000;
    double* h_d_in = (double*)malloc(length * sizeof(double));
    int* h_d_out = (int*)malloc(length * sizeof(int));
    double threshold = 1.0;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < length; ++i) {
        h_d_in[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    double time_taken = compareDoubleArrayToThreshold(h_d_in, h_d_out, length, threshold);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_259_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%d ", h_d_out[i]);
            }
            fclose(file);
        }
    }

    free(h_d_in);
    printf("Time: %.2f(ms)\n", time_taken);
    free(h_d_out);

    return 0;
}