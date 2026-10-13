#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void doubleArrayElementwiseSquare(double* d_in, double* d_out, int length, double* time_taken) {
    clock_t start, end;
    start = clock();

    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }

    end = clock();
    *time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char *argv[]) {
    const int array_size = 1000000;
    double* d_in_host = (double*)malloc(array_size * sizeof(double));
    double* d_out_result = (double*)malloc(array_size * sizeof(double));
    int verbose = 0;
    double time_taken;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < array_size; ++i) {
        d_in_host[i] = rand() % 100 + 1.0;
    }

    doubleArrayElementwiseSquare(d_in_host, d_out_result, array_size, &time_taken);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "%s_output.txt", "c_216");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", d_out_result[i]);
            }
            fclose(fp);
        }
    }

    printf("Time: %.2f(ms)\n", time_taken);
    free(d_in_host);
    free(d_out_result);

    return 0;
}