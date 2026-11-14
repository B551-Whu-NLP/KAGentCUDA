#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void flatten(float *x, float *swap, int size, int layers, int batch, int forward) {
    int i, c, b;
    for (b = 0; b < batch; ++b) {
        for (c = 0; c < layers; ++c) {
            for (i = 0; i < size; ++i) {
                int i1 = b * layers * size + c * size + i;
                int i2 = b * layers * size + i * layers + c;
                if (forward)
                    swap[i2] = x[i1];
                else
                    swap[i1] = x[i2];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int size = 1000;
    int layers = 150;
    int batch = 200;

    float *x = (float *)malloc(size * layers * batch * sizeof(float));
    float *swap = (float*)malloc(size * layers * batch * sizeof(float));

    for (int i = 0; i < size * layers * batch; ++i) {
        x[i] = i + 1;
        swap[i] = 0;
    }

    clock_t start = clock();
    flatten(x, swap, size, layers, batch, 1);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_329_output.txt");
        FILE *f = fopen(output_file, "w");
        if (f != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(f, "%.2f ", swap[i]);
            }
            fprintf(f, "\n");
            fclose(f);
        }
    }

    free(x);
    free(swap);

    return 0;
}