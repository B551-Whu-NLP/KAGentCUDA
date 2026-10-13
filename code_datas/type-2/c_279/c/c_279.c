#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void transKernel(float* array1, float* array2, int width) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < width; y++) {
            int current_index = x * width + y;
            int replace = y * width + x;
            array2[replace] = array1[current_index];
        }
    }
}

int main(int argc, char *argv[]) {
    int width = 1000;
    float* h_array1 = (float*)malloc(width * width * sizeof(float));
    float* h_array2 = (float*)malloc(width * width * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < width * width; ++i) {
        h_array1[i] = i;
    }

    clock_t start_time = clock();

    // Repeat the operation multiple times to increase runtime
    for (int iter = 0; iter < 1000; iter++) {
        transKernel(h_array1, h_array2, width);
        // Swap arrays for next iteration
        float* temp = h_array1;
        h_array1 = h_array2;
        h_array2 = temp;
    }

    clock_t end_time = clock();

    double time_taken = (double)(end_time - start_time) * 1000 / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        FILE *fp = fopen("c_279_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < 1000; ++j) {
                    fprintf(fp, "%.2f ", h_array2[i * width + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_array1);
    free(h_array2);

    return 0;
}