#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void allExp2Inplace(double* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = exp2(arr[i]);
    }
}

int main(int argc, char *argv[]) {
    const int data_size = 1000;
    double* arr_host = (double*)malloc(data_size * sizeof(double));
    for (int i = 0; i < data_size; ++i) {
        arr_host[i] = i + 1.0;
    }

    clock_t start = clock();

    allExp2Inplace(arr_host, data_size);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_191_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < data_size; i++) {
                fprintf(fp, "%.2f ", arr_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(arr_host);

    return 0;
}