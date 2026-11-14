#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void add(float* c, float* a, float* b, int n) {
    for (int k = 0; k < n; k++) {
        c[k] = a[k] + b[k];
    }
}

int main(int argc, char** argv) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    const int data_size = 10000000; // Increased from 1000 to 10,000,000
    float* a_host = (float*)malloc(data_size * sizeof(float));
    float* b_host = (float*)malloc(data_size * sizeof(float));
    float* c_result = (float*)malloc(data_size * sizeof(float));

    srand(42);
    for (int i = 0; i < data_size; ++i) {
        a_host[i] = rand() % 1000 / 100.0;
        b_host[i] = rand() % 1000 / 100.0;
    }

    clock_t start = clock();

    add(c_result, a_host, b_host, data_size);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_202_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", c_result[i]);
            }
            fclose(file);
        }
    } 

    free(a_host);
    free(b_host);
    free(c_result);

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    return 0;
}