#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mul(float* M, float* N, float* K, float height_M, float width_N, float width_M) {
    for (int i = 0; i < height_M; i++) {
        for (int j = 0; j < width_N; j++) {
            float sum = 0;
            for (int k = 0; k < width_M; k++) {
                float a = M[i * (int)width_M + k];
                float b = N[k * (int)width_N + j];
                sum += a * b;
            }
            K[i * (int)width_N + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int height_M = 256;
    int width_N = 256;
    int width_M = 256;

    float* h_Md = (float*)malloc(width_M * height_M * sizeof(float));
    float* h_Nd = (float*)malloc(width_N * height_M * sizeof(float));
    float* h_K = (float*)malloc(width_N * height_M * sizeof(float));

    for (int i = 0; i < width_M * height_M; i++) {
        h_Md[i] = i % 100;
    }
    for (int i = 0; i < width_N * height_M; i++) {
        h_Nd[i] = i % 100;
    }

    clock_t start = clock();
    for(int iter = 0; iter < 100; iter++) {
        mul(h_Md, h_Nd, h_K, height_M, width_N, width_M);
    }
    clock_t end = clock();

    double time_taken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_146_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < height_M; i++) {
                for (int j = 0; j < width_N; j++) {
                    fprintf(fp, "%.2f ", h_K[i * (int)width_N + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }


    free(h_Md);
    free(h_Nd);
    free(h_K);
    return 0;
}