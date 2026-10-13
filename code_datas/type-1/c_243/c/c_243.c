#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void doubleArrayVectorSubtract(double* d_in_a, double* d_in_b, double* d_out, int length) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in_a[idx] - d_in_b[idx];
    }
}

int main(int argc, char *argv[]) {
    int length = 10000000; // 增大数组长度
    double* h_in_a = (double*)malloc(length * sizeof(double));
    double* h_in_b = (double*)malloc(length * sizeof(double));
    double* h_out = (double*)malloc(length * sizeof(double));

    for (int i = 0; i < length; ++i) {
        h_in_a[i] = (double)(i + 1);
        h_in_b[i] = (double)(i);
    }

    clock_t start = clock();

    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 100; iter++) {
        doubleArrayVectorSubtract(h_in_a, h_in_b, h_out, length);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_243_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_out[i]);
            }
            fclose(fp);
        }
    }

    free(h_in_a);
    printf("Time: %.2f(ms)\n", time_taken);
    free(h_in_b);
    free(h_out);

    return 0;
}