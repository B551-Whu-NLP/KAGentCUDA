#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void Backwardsub(double* U, double* RES, double* UN, double* UE, double* LPR, int NI, int NJ, int End, int J, int n);

int main(int argc, char *argv[]) {
    int NI = 5000;
    int NJ = 5000;
    int End = 2500;
    int J = 1000;
    int n = 500;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    double* h_U = (double*)malloc(NI * NJ * sizeof(double));
    double* h_RES = (double*)malloc(NI * NJ * sizeof(double));
    double* h_UN = (double*)malloc(NI * NJ * sizeof(double));
    double* h_UE = (double*)malloc(NI * NJ * sizeof(double));
    double* h_LPR = (double*)malloc(NI * NJ * sizeof(double));

    for (int i = 0; i < NI * NJ; i++) {
        h_U[i] = 1.0;
        h_RES[i] = 2.0;
        h_UN[i] = 3.0;
        h_UE[i] = 4.0;
        h_LPR[i] = 5.0;
    }

    clock_t start = clock();

    for (int iter = 0; iter < 100000; iter++) {
        Backwardsub(h_U, h_RES, h_UN, h_UE, h_LPR, NI, NJ, End, J, n);
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        FILE *fp = fopen("c_127_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.2f ", h_U[i * NJ + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_U);
    free(h_RES);
    free(h_UN);
    free(h_UE);
    free(h_LPR);
    return 0;
}

void Backwardsub(double* U, double* RES, double* UN, double* UE, double* LPR, int NI, int NJ, int End, int J, int n) {
    for (int i = 0; i < n; i++) {
        int IJ = ((End - i) * NI) + (J - (End - i));
        RES[IJ] = RES[IJ] - UN[IJ] * RES[IJ + 1] - UE[IJ] * RES[IJ + NJ];
        U[IJ] = U[IJ] + RES[IJ];
    }
}