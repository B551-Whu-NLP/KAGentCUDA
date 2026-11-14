#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void cross_correlate(float* Isg, float* Iss, float* sp, float* gp, int npml, int nnz, int nnx) {
    for (int i1 = npml; i1 < nnz - npml; i1++) {
        for (int i2 = npml; i2 < nnx - npml; i2++) {
            int id = i1 + i2 * nnz;
            float ps = sp[id];
            float pg = gp[id];
            Isg[id] += ps * pg;
            Iss[id] += ps * ps;
        }
    }
}

int main(int argc, char* argv[]) {
    int npml = 1;
    int nnz = 1000;
    int nnx = 1000;

    float* h_Isg = (float*)malloc(nnz * nnx * sizeof(float));
    float* h_Iss = (float*)malloc(nnz * nnx * sizeof(float));
    float* h_sp = (float*)malloc(nnz * nnx * sizeof(float));
    float* h_gp = (float*)malloc(nnz * nnx * sizeof(float));

    for (int i = 0; i < nnz * nnx; i++) {
        h_Isg[i] = 0.0f;
        h_Iss[i] = 0.0f;
        h_sp[i] = (float)(i);
        h_gp[i] = (float)((i + 5) % 10);
    }

    clock_t start = clock();

    for (int iter = 0; iter < 100; iter++) {
        cross_correlate(h_Isg, h_Iss, h_sp, h_gp, npml, nnz, nnx);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* file = fopen("c_118_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "Isg[%d] = %f\n", i, h_Isg[i]);
            }
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "Iss[%d] = %f\n", i, h_Iss[i]);
            }
            fclose(file);
        }
    }

    free(h_Isg);
    free(h_Iss);
    free(h_sp);
    free(h_gp);
    return 0;
}