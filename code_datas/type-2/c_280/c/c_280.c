#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void rowSumSquare(const double* mat, double* buf, int m, int n, int verbose, const char* file_name) {
    clock_t start = clock();

    for (int i = 0; i < m; i++) {
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            double a = mat[i * n + j];
            sum += a * a;
        }

        buf[i] = sum;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "%s_output.txt", file_name);
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", buf[i]);
            }
            fclose(fp);
        }
    }
}

int main(int argc, char* argv[]) {
    int m = 5000;
    int n = 5000;
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_buf = (double*)malloc(m * sizeof(double));

    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = i % 100;
    }

    int verbose = 0;
    const char* file_name = "c_280";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    rowSumSquare(h_mat, h_buf, m, n, verbose, file_name);

    free(h_mat);
    free(h_buf);

    return 0;
}