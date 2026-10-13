#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void SimpleCorrelator(float* xi, float* xq, float* sr, float* si, int sLength, float* L, int uLength) {
    clock_t start = clock();

    for (int u = 0; u < uLength; u++) {
        float real = 0;
        float imag = 0;
        float a, b, c, d;

        for (int n = u; n < u + sLength; n++) {
            a = xi[n];
            b = xq[n];
            c = sr[n - u];
            d = si[n - u] * (-1);
            real += (a * c) - (b * d);
            imag += (a * d) + (b * c);
        }

        L[u] = sqrtf(real * real + imag * imag);
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int sLength = 10000;
    int uLength = 20000;

    float* xi = (float*)malloc((uLength + sLength) * sizeof(float));
    float* xq = (float*)malloc((uLength + sLength) * sizeof(float));
    float* sr = (float*)malloc(uLength * sizeof(float));
    float* si = (float*)malloc(uLength * sizeof(float));
    float* L = (float*)malloc(uLength * sizeof(float));

    srand(42);

    for (int i = 0; i < uLength + sLength; i++) {
        xi[i] = rand() % 100;
        xq[i] = rand() % 100;
    }

    for (int i = 0; i < uLength; i++) {
        sr[i] = rand() % 100;
        si[i] = rand() % 100;
    }

    SimpleCorrelator(xi, xq, sr, si, sLength, L, uLength);

    if (verbose) {
        FILE *fp = fopen("c_150_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < uLength; i++) {
                fprintf(fp, "%f ", L[i]);
            }
            fclose(fp);
        }
    }

    free(xi);
    free(xq);
    free(sr);
    free(si);
    free(L);
    return 0;
}