#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void MatrixMul(float* M, float* N, float* P, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            double sum = 0;
            for (int k = 0; k < width; ++k) {
                double a = M[i * width + k];
                double b = N[k * width + j];
                sum += a * b;
            }
            P[i * width + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    const int width = 1000;
    float* h_Md = (float*)malloc(width * width * sizeof(float));
    float* h_Nd = (float*)malloc(width * width * sizeof(float));
    float* h_Pd = (float*)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; i++) {
        h_Md[i] = (float)(i % 10 + 1);
        h_Nd[i] = (float)(i % 7 + 1);
    }

    clock_t start = clock();

    MatrixMul(h_Md, h_Nd, h_Pd, width);

    clock_t end = clock();
    double time_taken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_300_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < width; j++) {
                    fprintf(fp, "%.2f ", h_Pd[i * width + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_Md);
    free(h_Nd);
    free(h_Pd);

    return 0;
}