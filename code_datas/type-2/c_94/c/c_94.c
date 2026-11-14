#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matmul(float* a, float* b, float* c, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float result = 0;
            for (int k = 0; k < width; k++) {
                result += a[row * width + k] * b[k * width + col];
            }
            c[row * width + col] = result;
        }
    }
}

int main(int argc, char** argv) {
    int width = 100;
    size_t size = width * width * sizeof(float);
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)calloc(width * width, sizeof(float));

    for (int i = 0; i < width * width; i++) {
        h_a[i] = (float)(i % 10 + 1);
        h_b[i] = (float)((i + 1) % 10 + 1);
    }

    clock_t start = clock();

    matmul(h_a, h_b, h_c, width);

    clock_t end = clock();

    double time_taken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_94_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < width; j++) {
                    fprintf(file, "%.2f ", h_c[i * width + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}