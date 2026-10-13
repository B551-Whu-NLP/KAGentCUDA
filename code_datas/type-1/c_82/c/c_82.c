#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void filterFFT(float* FFT, float* filter, int nxprj2, int nviews, float scale) {
    for (int i = 0; i < nviews; i++) {
        for (int j = 0; j < nxprj2; j++) {
            FFT[i * nxprj2 + j] *= filter[i * nxprj2 + j] * scale;
        }
    }
}

int main(int argc, char *argv[]) {
    const int nxprj2 = 8192;
    const int nviews = 2000;
    float scale = 0.5f;
    float* h_FFT = (float*)malloc(nxprj2 * nviews * sizeof(float));
    float* h_filter = (float*)malloc(nxprj2 * nviews * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);

    for (int i = 0; i < nviews * nxprj2; ++i) {
        h_FFT[i] = (float)(rand() % 100);
        h_filter[i] = (float)(rand() % 10) / 10.0f;
    }

    clock_t start = clock();

    filterFFT(h_FFT, h_filter, nxprj2, nviews, scale);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_82_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < 1000; ++j) {
                    fprintf(fp, "%.2f ", h_FFT[i * nxprj2 + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }


    printf("Time: %.2f(ms)\n", time_taken);
    free(h_FFT);
    free(h_filter);

    return 0;
}