#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void fa(const float* q, const float* h, int nq, float* a, float* fa) {
    for (int iq = 0; iq < (nq - 1); iq++) {
        float dq = q[iq + 1] - q[iq];
        a[iq] = (h[iq + 1] * q[iq + 1] - h[iq] * q[iq]) / dq;
        fa[iq] = q[iq] * (a[iq] - h[iq]) + 1.0;
    }
}

int main(int argc, char* argv[]) {
    int nq = 500000;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* q = (float*)malloc(nq * sizeof(float));
    float* h = (float*)malloc(nq * sizeof(float));
    float* a = (float*)malloc((nq - 1) * sizeof(float));
    float* fa_result = (float*)malloc((nq - 1) * sizeof(float));

    for (int i = 0; i < nq; i++) {
        q[i] = (float)(i + 1);
        h[i] = 0.5 * (float)(i + 1);
    }

    clock_t start = clock();
    for (int iter = 0; iter < 5000; iter++) {
        fa(q, h, nq, a, fa_result);
    }
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_306_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", a[i]);
            }
            fprintf(file, "\n");
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", fa_result[i]);
            }
            fclose(file);
        }
    }

    free(q);
    free(h);
    free(a);
    free(fa_result);
    return 0;
}