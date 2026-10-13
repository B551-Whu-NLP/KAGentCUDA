#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void primal_descent(float* y1, float* y2, float* xbar, float sigma, int w, int h, int nc) {
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int i;
            float x1, x2, val, norm;
            for (int z = 0; z < nc; z++) {
                i = x + w * y + w * h * z;
                val = xbar[i];
                x1 = (x + 1 < w) ? (xbar[(x + 1) + w * y + w * h * z] - val) : 0.f;
                x2 = (y + 1 < h) ? (xbar[x + w * (y + 1) + w * h * z] - val) : 0.f;
                x1 = y1[i] + sigma * x1;
                x2 = y2[i] + sigma * x2;
                norm = sqrtf(x1 * x1 + x2 * x2);
                y1[i] = x1 / fmaxf(1.f, norm);
                y2[i] = x2 / fmaxf(1.f, norm);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    const int w = 512;
    const int h = 512;  
    const int nc = 30;
    const float sigma = 0.01;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* y1, * y2, * xbar;
    y1 = (float*)malloc(w * h * nc * sizeof(float));
    y2 = (float*)malloc(w * h * nc * sizeof(float));
    xbar = (float*)malloc(w * h * nc * sizeof(float));

    srand(42);
    for (int i = 0; i < w * h * nc; ++i) {
        y1[i] = (float)(rand() % 1000) / 100.0f;
        y2[i] = (float)(rand() % 1000) / 100.0f;
        xbar[i] = (float)(rand() % 1000) / 100.0f;
    }

    clock_t start = clock();

    primal_descent(y1, y2, xbar, sigma, w, h, nc);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_344_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < w * h; ++i) {
                fprintf(fp, "%f ", y1[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < w * h; ++i) {
                fprintf(fp, "%f ", y2[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(y1);
    free(y2);
    free(xbar);
    return 0;
}