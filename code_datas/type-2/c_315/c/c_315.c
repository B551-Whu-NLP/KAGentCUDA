#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void binarize_input(float* input, int n, int size, float* binary) {
    int i, s;

    clock_t start = clock();

    for (s = 0; s < size; ++s) {
        float mean = 0;

        for (i = 0; i < n; ++i) {
            mean += fabs(input[i * size + s]);
        }

        mean = mean / n;

        for (i = 0; i < n; ++i) {
            binary[i * size + s] = (input[i * size + s] > 0) ? mean : -mean;
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_spent * 1000);
}

int main(int argc, char *argv[]) {
    int n = 5000; 
    int size = 5000; 

    float *input = (float *)malloc(n * size * sizeof(float));
    float *binary = (float *)malloc(n * size * sizeof(float));

    for (int i = 0; i < n * size; i++) {
        input[i] = (float)((i % 10) + 1);
    }

    binarize_input(input, n, size, binary);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_315_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.2f ", binary[i * size + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(input);
    free(binary);

    return 0;
}