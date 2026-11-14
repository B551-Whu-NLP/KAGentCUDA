#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void allDivInplace(double* arr, double alpha, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] /= alpha;
    }
}

int main(int argc, char** argv) {
    const int data_size = 1000000;
    const double alpha = 2.0;
    double* arr_host = (double*)malloc(data_size * sizeof(double));
    for (int i = 0; i < data_size; ++i) {
        arr_host[i] = i + 1.0; 
    }

    clock_t start = clock();

    allDivInplace(arr_host, alpha, data_size);

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    printf("Time: %.2fms\n", time_taken * 1000);
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_186_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", arr_host[i]);
            }
            fprintf(fp, "\n");
            fprintf(fp, "Time: %.2fms\n", time_taken * 1000);
            fclose(fp);
        }
    }

    free(arr_host);

    return 0;
}