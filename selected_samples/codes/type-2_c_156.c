#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void dot(int N, float* X, int INCX, float* Y, int INCY, float* result) {
    int i;
    float sum = 0.0;

    for (i = 0; i < N; ++i) {
        sum += X[i * INCX] * Y[i * INCY];
    }
    
    result[0] = sum;
}

int main(int argc, char** argv) {
    int N = 1000000;  
    int INCX = 1;
    int INCY = 1;
    float* result = (float*)malloc(sizeof(float));
    result[0] = 0.0f;

    float* X = (float*)malloc(N * INCX * sizeof(float));
    float* Y = (float*)malloc(N * INCY * sizeof(float));

    for (int i = 0; i < N; i++) {
        X[i * INCX] = i + 1;
        Y[i * INCY] = i + 2;
    }

    clock_t start = clock();
    dot(N, X, INCX, Y, INCY, result);
    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_156_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            fprintf(file, "%f\n", result[0]);
            fclose(file);
        }
    }

    printf("%f\n", result[0]);
    free(X);
    free(Y);
    free(result);
    return 0;
}