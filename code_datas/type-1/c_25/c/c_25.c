#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rows = 100000;
    int cols = 100000;
    double* h_buf = (double*)malloc(cols * sizeof(double));
    double* h_tmp = (double*)malloc(cols * sizeof(double));

    for (int i = 0; i < cols; ++i) {
        h_tmp[i] = (double)(i + 1);
    }

    clock_t start = clock();

    for (int iter = 0; iter < 10000; iter++) {
        for (int i = 0; i < cols; i++) {
            h_buf[i] = h_tmp[i] / rows;
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_25_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_buf[i]);
            }
            fclose(fp);
        }
    }

    free(h_buf);
    free(h_tmp);
    return 0;
}