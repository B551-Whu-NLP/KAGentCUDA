#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char** argv) {
    const int data_size = 100000000;
    double* arr_host = (double*)malloc(data_size * sizeof(double));
    double* buf_host = (double*)malloc(data_size * sizeof(double));
    for (int i = 0; i < data_size; ++i) {
        arr_host[i] = i + 1.0;
    }

    clock_t start = clock();

    for (int i = 0; i < data_size; i++) {
        buf_host[i] = arr_host[i] / 2.0;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_198_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", buf_host[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }

    free(arr_host);
    free(buf_host);

    return 0;
}