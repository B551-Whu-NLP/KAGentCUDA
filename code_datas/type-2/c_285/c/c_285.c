#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Avg(float* dev_arrayMax, float* dev_array, const int r, const int c) {
    float sum;
    for (int i = 0; i < r; i++) {
        sum = 0.0;
        for (int j = 0; j < c; j++) {
            sum += dev_array[i * c + j];
        }
        dev_arrayMax[i] = sum / c;
    }
}

int main(int argc, char* argv[]) {
    const int r = 10000;
    const int c = 10000;
    float* h_array = (float*)malloc(r * c * sizeof(float));
    float* h_arrayMax = (float*)malloc(r * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < r * c; ++i) {
        h_array[i] = i % 100;
    }

    clock_t start = clock();

    Avg(h_arrayMax, h_array, r, c);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_285_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f\t", h_arrayMax[i]);
            }
            fclose(file);
        }
    }

    free(h_array);
    free(h_arrayMax);

    return 0;
}