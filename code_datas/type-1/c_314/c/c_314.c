#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void deinter(int NX, float* X, int NY, float* Y, int B, float* OUT) {
    int i, j;
    int index = 0;

    clock_t start = clock();

    for (j = 0; j < B; ++j) {
        for (i = 0; i < NX; ++i) {
            if (X) X[j * NX + i] += OUT[index];
            ++index;
        }

        for (i = 0; i < NY; ++i) {
            if (Y) Y[j * NY + i] += OUT[index];
            ++index;
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    // Removed unused variable warning by using the variable
    (void)time_taken;
}

int main(int argc, char *argv[]) {
    int NX = 1000;
    int NY = 800;
    int B = 500;

    float *X = (float *)malloc(B * NX * sizeof(float));
    float *Y = (float *)malloc(B * NY * sizeof(float));
    float *OUT = (float *)malloc((NX + NY) * B * sizeof(float));

    for (int i = 0; i < B * NX; i++) {
        X[i] = i + 1.0;
    }
    for (int i = 0; i < B * NY; i++) {
        Y[i] = i + 1.0;
    }
    for (int i = 0; i < (NX + NY) * B; i++) {
        OUT[i] = i * 0.1f; 
    }

    deinter(NX, X, NY, Y, B, OUT);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_314_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", X[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 800; i++) {
                fprintf(fp, "%.2f ", Y[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(X);
    free(Y);
    free(OUT);

    return 0;
}