#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void MMDSelfComputeWithSum(float* x_average, int size_x, float* distance_matrix) {
    clock_t start_time = clock();

    for (int i = 0; i < size_x; i++) {
        for (int j = i; j < size_x; j++) {
            distance_matrix[i * size_x + j] = x_average[i] * x_average[j];
            distance_matrix[j * size_x + i] = distance_matrix[i * size_x + j];
        }
    }

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", elapsed_time * 1000);
}

int main(int argc, char *argv[]) {
    int size_x = 1000; // Increased from 100 to 1000
    float* h_x_average = (float*)malloc(size_x * sizeof(float));
    float* h_distance_matrix = (float*)malloc(size_x * size_x * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < size_x; ++i) {
        h_x_average[i] = (float)(i + 1);
    }

    MMDSelfComputeWithSum(h_x_average, size_x, h_distance_matrix);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", "c_277");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < 1000; ++j) {
                    fprintf(file, "%.2f ", h_distance_matrix[i * size_x + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_x_average);
    free(h_distance_matrix);

    return 0;
}