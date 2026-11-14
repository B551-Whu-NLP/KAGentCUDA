#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void doubleArrayScalarAdd(double* d_in, double* d_out, int length, double scalar) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in[idx] + scalar;
    }
}

int main(int argc, char** argv) {
    const int array_size = 10000000; // 增大数组大小
    double scalar = 5.0;
    double* d_in_host = (double*)malloc(array_size * sizeof(double));
    double* d_out_result = (double*)malloc(array_size * sizeof(double));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < array_size; ++i) {
        d_in_host[i] = i + 1.0;
    }

    clock_t start = clock();

    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 100; iter++) {
        doubleArrayScalarAdd(d_in_host, d_out_result, array_size, scalar);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_filename[256];
        sprintf(output_filename, "%s_output.txt", "c_206");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", d_out_result[i]);
            }
            fclose(file);
        }
    }


    free(d_in_host);
    free(d_out_result);

    return 0;
}