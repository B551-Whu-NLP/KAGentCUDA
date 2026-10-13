#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mult_add_into(int n, float* a, float* b, float* c) {
    for (int i = 0; i < n; ++i) {
        c[i] += a[i] * b[i];
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 10000000;
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_b = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = (float)(i);
        h_b[i] = (float)(2 * i);
        h_c[i] = (float)(3 * i);
    }

    clock_t start = clock();
    
    mult_add_into(arraySize, h_a, h_b, h_c);
    
    clock_t end = clock();
    
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        FILE* file = fopen("c_40_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_c[i]);
            }
            fclose(file);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    printf("Time: %.2fms\n", timeElapsed * 1000);
    return 0;
}