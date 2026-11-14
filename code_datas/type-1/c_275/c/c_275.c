#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void weighted_sum(int n, float* a, float* b, float* s, float* c) {
    for (int i = 0; i < n; ++i) {
        c[i] = s[i] * a[i] + (1 - s[i]) * (b ? b[i] : 0);
    }
}

int main(int argc, char** argv) {
    int size = 10000000; // 增大数组大小
    float* h_a = (float*)malloc(size * sizeof(float));
    float* h_b = (float*)malloc(size * sizeof(float));
    float* h_s = (float*)malloc(size * sizeof(float));
    float* h_c = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_s[i] = 0.5f;
    }

    clock_t start = clock();

    weighted_sum(size, h_a, h_b, h_s, h_c);

    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_spent * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_275_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_c[i]);
            }
            fclose(file);
        }
    }

    free(h_a);
    free(h_b);
    free(h_s);
    free(h_c);

    return 0;
}