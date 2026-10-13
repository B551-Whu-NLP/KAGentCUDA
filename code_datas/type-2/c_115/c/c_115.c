#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Reduction(float* dev_c, const float* reduction, int r, int c, int n, int size_block) {
    clock_t start = clock();

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            float temp = 0;
            for (int k = 0; k < size_block; k++) {
                temp += reduction[i * (c * size_block) + j * size_block + k];
            }
            dev_c[i * c + j] = temp;
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char *argv[]) {
    int r = 1000;
    int c = 1000; 
    int n = 1000; 
    int size_block = 16; 

    float* h_reduction = (float*)malloc(r * c * n * sizeof(float));
    float* h_dev_c = (float*)malloc(r * c * sizeof(float));

    for (int i = 0; i < r * c * n; i++) {
        h_reduction[i] = (float)(i % 100);
    }

    Reduction(h_dev_c, h_reduction, r, c, n, size_block);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_115_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.6f\n", h_dev_c[i]);
            }
            fclose(fp);
        }
    }

    free(h_reduction);
    free(h_dev_c);

    return 0;
}