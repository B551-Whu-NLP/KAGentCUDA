#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void ChoiLee(float* xi, float* xq, float* sr, float* si, int N, float* L) {
    clock_t start = clock();

    for (int u = 0; u < N; u++) {
        float uSum = 0;
        float r_i, r_q, rconj_i, rconj_q;
        float s_i, s_q, sconj_i, sconj_q;
        float rsum_i, rsum_q, ssum_i, ssum_q;
        float ksum_i, ksum_q;

        for (int i = 0; i < N; i++) {
            ksum_i = 0;
            ksum_q = 0;

            for (int k = 0; k < N - i; k++) {
                r_i = xi[u + k + i];
                r_q = xq[u + k + i];
                rconj_i = xi[u + k];
                rconj_q = -xq[u + k];
                s_i = sr[k];
                s_q = si[k];
                sconj_i = sr[k + i];
                sconj_q = -si[k + i];
                rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
                rsum_q = (r_i * rconj_q) + (r_q * rconj_i);
                ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
                ssum_q = (s_i * sconj_q) + (s_q * sconj_i);
                ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
                ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
            }

            uSum += sqrtf(ksum_i * ksum_i + ksum_q * ksum_q);
        }

        L[u] = uSum;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    int N = 500;
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* xi_host = (float*)malloc(N * sizeof(float));
    float* xq_host = (float*)malloc(N * sizeof(float));
    float* sr_host = (float*)malloc(N * sizeof(float));
    float* si_host = (float*)malloc(N * sizeof(float));
    float* L_host = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        xi_host[i] = sinf(i);
        xq_host[i] = cosf(i);
        sr_host[i] = sinf(i * 0.5f);
        si_host[i] = cosf(i * 0.5f);
    }

    ChoiLee(xi_host, xq_host, sr_host, si_host, N, L_host);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_163_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < N; i++) {
                fprintf(fp, "L[%d] = %f\n", i, L_host[i]);
            }
            fclose(fp);
        }
    }

    free(xi_host);
    free(xq_host);
    free(sr_host);
    free(si_host);
    free(L_host);
    return 0;
}