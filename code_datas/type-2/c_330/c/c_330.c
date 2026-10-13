#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void Softmax(float* x, const int r, const int c) {
    float temp1, temp2;
    for (int i = 0; i < r; i++) {
        temp1 = 0.;
        temp2 = 0.;
        for (int j = 0; j < c; j++) {
            temp1 = fmaxf(x[i * c + j], temp1);
        }
        for (int j = 0; j < c; j++) {
            x[i * c + j] = expf(x[i * c + j] - temp1);
            temp2 += x[i * c + j];
        }
        for (int j = 0; j < c; j++) {
            x[i * c + j] /= temp2;
        }
    }
}

int main(int argc, char** argv) {
    const int r = 10000;
    const int c = 10000;

    float* x = (float*)malloc(r * c * sizeof(float));

    srand(42);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            x[i * c + j] = (float)(rand() % 1000) / 100.0f;
        }
    }

    clock_t start = clock();
    Softmax(x, r, c);
    clock_t end = clock();
    double time_spent = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_spent);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_330_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(file, "%.4f ", x[i * c + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(x);

    return 0;
}