#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void allMulInplace(double* arr, double alpha, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] *= alpha;
    }
}

int main(int argc, char* argv[]) {
    const int data_size = 100000000;  // 增加100倍
    const double alpha = 2.0;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    double* arr_host = (double*)malloc(data_size * sizeof(double));
    for (int i = 0; i < data_size; ++i) {
        arr_host[i] = i + 1.0;
    }

    clock_t start = clock();

    allMulInplace(arr_host, alpha, data_size);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_188_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", arr_host[i]);
            }
            fprintf(file, "\n");
            fprintf(file, "Time: %.2f(ms)\n", time_taken * 1000);
            fclose(file);
        }
    }

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    free(arr_host);

    return 0;
}