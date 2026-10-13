#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add_arrays(int n, float* x, float* y, float* z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

int main(int argc, char** argv) {
    int arraySize = 100000000; // 进一步增大数组大小
    float* h_x = (float*)malloc(arraySize * sizeof(float));
    float* h_y = (float*)malloc(arraySize * sizeof(float));
    float* h_z = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_x[i] = (float)(i);
        h_y[i] = (float)(2 * i);
    }

    clock_t start = clock();

    add_arrays(arraySize, h_x, h_y, h_z);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_29_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_z[i]);
            }
            fclose(file);
        }
    }

    free(h_x);
    free(h_y);
    free(h_z);
    return 0;
}