#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Argmax(int* dev_argMax, float* dev_array, const int r, const int c) {
    for (int i = 0; i < r; i++) {
        int idx = 0;
        float temp = 0.0;

        for (int j = 0; j < c; j++) {
            if (dev_array[i * c + j] > temp) {
                temp = dev_array[i * c + j];
                idx = j;
            }
        }
        dev_argMax[i] = idx;
    }
}

int main(int argc, char *argv[]) {
    const int r = 20000;
    const int c = 20000;
    float* h_array = (float*)malloc(r * c * sizeof(float));
    int* h_argMax = (int*)malloc(r * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < r * c; ++i) {
        h_array[i] = (float)(i % 100);
    }

    clock_t start = clock();

    Argmax(h_argMax, h_array, r, c);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_297_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Row %d: Argmax=%d\n", i, h_argMax[i]);
            }
            fclose(fp);
        }
    }

    free(h_array);
    free(h_argMax);

    return 0;
}