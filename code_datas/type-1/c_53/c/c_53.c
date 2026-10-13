#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    clock_t start = clock();

    for (int i = 0; i < N; ++i) {
        Y[i * INCY] = powf(X[i * INCX], ALPHA);
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    const int N = 10000;
    float* h_X = (float*)malloc(N * sizeof(float));
    float* h_Y = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; ++i) {
        h_X[i] = (float)(i + 1);
    }

    pow_cpu(N, 2.0, h_X, 1, h_Y, 1);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_53_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Y[%d]: %f\n", i, h_Y[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_X);
    free(h_Y);
    return 0;
}