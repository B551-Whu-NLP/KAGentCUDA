#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add(int N, float ALPHA, float* X, int INCX) {
    for (int i = 0; i < N; i++) {
        X[i * INCX] += ALPHA;
        if (X[i * INCX] == -128.0f) {
            X[i * INCX] = -127.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 10000000; // 增加数组大小
    float* h_X;
    h_X = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_X[i] = i * 0.5f;
    }

    clock_t start = clock();

    // 增加迭代次数
    for (int iter = 0; iter < 100; iter++) {
        add(N, 2.0f, h_X, 1);
    }

    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_233_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_X[i]);
            }
            fclose(fp);
        }
    }

    free(h_X);
    printf("Time: %.2f(ms)\n", time_spent * 1000);
    return 0;
}