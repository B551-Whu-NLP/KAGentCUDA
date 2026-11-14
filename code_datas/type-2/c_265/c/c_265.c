#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void ParTrans(float* d_in, float* d_out, int x_size, int y_size) {
    for (int gidx = 0; gidx < x_size; gidx++) {
        for (int gidy = 0; gidy < y_size; gidy++) {
            d_out[gidx * y_size + gidy] = d_in[gidy * x_size + gidx];
        }
    }
}

int main(int argc, char** argv) {
    int x_size = 10000;
    int y_size = 8000;
    float* h_in = (float*)malloc(x_size * y_size * sizeof(float));
    float* h_out = (float*)malloc(y_size * x_size * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < x_size * y_size; ++i) {
        h_in[i] = i + 1.0;
    }

    clock_t start = clock();
    ParTrans(h_in, h_out, x_size, y_size);
    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", "c_265");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_out[i]);
                if ((i + 1) % x_size == 0) {
                    fprintf(file, "\n");
                }
            }
            fclose(file);
        }
    }

    free(h_in);
    free(h_out);

    return 0;
}