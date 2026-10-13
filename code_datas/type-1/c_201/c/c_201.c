#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add(int n, float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char** argv) {
    const int data_size = 1000000;
    float* x_host = (float*)malloc(data_size * sizeof(float));
    float* y_host = (float*)malloc(data_size * sizeof(float));
    
    for (int i = 0; i < data_size; ++i) {
        x_host[i] = i + 1.0;
        y_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();
    add(data_size, x_host, y_host);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* output_file = fopen("c_201_output.txt", "w");
        if (output_file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(output_file, "%.2f ", y_host[i]);
            }
            fclose(output_file);
        }
    }


    free(x_host);
    free(y_host);

    return 0;
}