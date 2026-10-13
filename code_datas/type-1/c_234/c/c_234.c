#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void doubleArraySign(double* d_in, double* d_out, int length) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = (0 < d_in[idx]) - (d_in[idx] < 0);
    }
}

int main(int argc, char** argv) {
    int length = 10000000;
    double* h_d_in;
    double* h_d_out;
    h_d_in = (double*)malloc(length * sizeof(double));
    h_d_out = (double*)malloc(length * sizeof(double));

    for (int i = 0; i < length; ++i) {
        h_d_in[i] = i - 50.0;
    }

    clock_t start = clock();

    doubleArraySign(h_d_in, h_d_out, length);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_234_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_d_out[i]);
            }
            fclose(file);
        }
    }

    free(h_d_in);
    free(h_d_out);

    return 0;
}