#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void CEE(float* x, int* t, float* loss, int r, int c) {
    clock_t start = clock();

    for (int i = 0; i < r; i++) {
        float temp;
        int loss_index = 0;
        for (int j = 0; j < c; j++) {
            if (t[i * c + j] == 1) {
                temp = logf(x[i * c + j] + 1e-7);
                loss[loss_index] += temp;
            }
        }
    }

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);
}

int main(int argc, char** argv) {
    const int r = 10000;
    const int c = 10000;
    float* h_x = (float*)malloc(r * c * sizeof(float));
    int* h_t = (int*)malloc(r * c * sizeof(int));
    float* h_loss = (float*)calloc(r, sizeof(float));

    for (int i = 0; i < r * c; i++) {
        h_x[i] = (float)(i % 10);
        h_t[i] = (i + 1) % 2;
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    CEE(h_x, h_t, h_loss, r, c);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_298_output.txt");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            fprintf(f, "Cross-Entropy Loss: %f\n", *h_loss);
            fclose(f);
        }
    }

    free(h_x);
    free(h_t);
    free(h_loss);

    return 0;
}