#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int size = 1000000;
    double* h_old_arr = (double*)malloc(size * sizeof(double));
    double* h_new_arr = (double*)malloc(size * sizeof(double));

    for (int i = 0; i < size; ++i) {
        h_old_arr[i] = (double)i;
    }

    clock_t start = clock();
    for (int tid = 0; tid < size; tid++) {
        h_new_arr[tid] = h_old_arr[tid];
    }
    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_2_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_new_arr[i]);
            }
            fclose(fp);
        }
    }
    printf("\nTime: %.2f(ms)\n", time_taken);

    free(h_old_arr);
    free(h_new_arr);

    return 0;
}