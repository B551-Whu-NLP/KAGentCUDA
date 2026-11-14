#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void k_vec_divide(float* vec1, float* vec2, int max_size) {
    for (int i = 0; i < max_size; i++) {
        vec1[i] = vec1[i] / vec2[i];
    }
}

int main(int argc, char* argv[]) {
    const int array_size = 10000000; // Increased from 1000 to 10,000,000
    float* vec1_host = (float*)malloc(array_size * sizeof(float));
    float* vec2_host = (float*)malloc(array_size * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < array_size; ++i) {
        vec1_host[i] = i + 1.0;
        vec2_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();

    k_vec_divide(vec1_host, vec2_host, array_size);

    clock_t end = clock();

    double time_taken = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_218_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", vec1_host[i]);
            }
            fclose(file);
        }
    }

    free(vec1_host);
    free(vec2_host);

    return 0;
}