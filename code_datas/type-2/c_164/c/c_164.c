#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void BYUSimplified(float* xi, float* xq, float* sr, float* si, int N, int Lq, float* L) {
    clock_t start = clock();

    for (int u = 0; u < N; u++) {
        float uSum = 0;
        float r_i, r_q, q_i, q_q;
        float realPart, imagPart;

        for (int k = 0; k <= 7; k++) {
            realPart = 0;
            imagPart = 0;

            for (int l = 0; l < Lq; l++) {
                r_i = xi[u + k * Lq + l];
                r_q = xq[u + k * Lq + l];
                q_i = sr[l];
                q_q = si[l] * (-1);

                realPart += (r_i * q_i) - (r_q * q_q);
                imagPart += (r_i * q_q) + (r_q * q_i);
            }

            uSum += (realPart * realPart) + (imagPart * imagPart);
        }

        L[u] = uSum;
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeElapsed);
}

int main(int argc, char* argv[]) {
    int N = 10000;
    int Lq = 2500;

    float* xi = (float*)malloc(N * 8 * Lq * sizeof(float));
    float* xq = (float*)malloc(N * 8 * Lq * sizeof(float));
    float* sr = (float*)malloc(Lq * sizeof(float));
    float* si = (float*)malloc(Lq * sizeof(float));
    float* L = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N * 8 * Lq; i++) {
        xi[i] = i + 1;
        xq[i] = i + 2;
    }

    for (int i = 0; i < Lq; i++) {
        sr[i] = i + 3;
        si[i] = i + 4;
    }

    for (int i = 0; i < N; i++) {
        L[i] = 0;
    }

    BYUSimplified(xi, xq, sr, si, N, Lq, L);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* fp = fopen("c_164_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", L[i]);
            }
            fprintf(fp, "\n");
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