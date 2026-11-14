#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void transpose(float* a, float* b, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            b[col * width + row] = a[row * width + col];
        }
    }
}

int main(int argc, char *argv[]) {
    int width = 1000;
    float* h_a = (float*)malloc(width * width * sizeof(float));
    float* h_b = (float*)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; ++i) {
        h_a[i] = (float)(i);
    }

    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        transpose(h_a, h_b, width);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_260_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j < width; ++j) {
                    fprintf(fp, "%.2f\t", h_b[i * width + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_a);
    free(h_b);

    return 0;
}