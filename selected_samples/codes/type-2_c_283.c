#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const int r = 1000;
    const int c = 800;
    float* h_dev_transposeArray = (float*)malloc(r * c * sizeof(float));
    float* h_dev_array = (float*)malloc(r * c * sizeof(float));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < r * c; ++i) {
        h_dev_array[i] = i;
    }

    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                h_dev_transposeArray[j * r + i] = h_dev_array[i * c + j];
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_283_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < c; ++i) {
                for (int j = 0; j < r; ++j) {
                    fprintf(fp, "%.2f\t", h_dev_transposeArray[i * r + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_dev_transposeArray);
    free(h_dev_array);

    return 0;
}