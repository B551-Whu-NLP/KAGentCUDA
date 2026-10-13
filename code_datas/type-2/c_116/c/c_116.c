#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Forwardsub(double* RES, double* LS, double* LW, double* LPR, int NI, int NJ, int Start, int J, int n) {
    for (int i = 0; i < n; i++) {
        int IJ = ((Start + i) * NI) + (J - (Start + i));
        RES[IJ] = (RES[IJ] - LS[IJ] * RES[IJ - 1] - LW[IJ] * RES[IJ - NJ]) * LPR[IJ];
    }
}

int main(int argc, char** argv) {
    int NI = 1000;
    int NJ = 1000;
    int Start = 0;
    int J = 200;
    int n = 1000;

    double* h_RES = (double*)malloc(NI * NJ * sizeof(double));
    double* h_LS = (double*)malloc(NI * NJ * sizeof(double));
    double* h_LW = (double*)malloc(NI * NJ * sizeof(double));
    double* h_LPR = (double*)malloc(NI * NJ * sizeof(double));

    for (int i = 0; i < NI * NJ; i++) {
        h_RES[i] = (double)(i % 1000);
        h_LS[i] = 0.5;
        h_LW[i] = 0.3;
        h_LPR[i] = 0.1;
    }

    clock_t start_time = clock();

    for (int iter = 0; iter < 100; iter++) {
        Forwardsub(h_RES, h_LS, h_LW, h_LPR, NI, NJ, Start, J, n);
    }

    clock_t end_time = clock();

    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", elapsed_time);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_116_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.6f ", h_RES[i]);
            }
            fclose(file);
        }
    }

    free(h_RES);
    free(h_LS);
    free(h_LW);
    free(h_LPR);
    return 0;
}