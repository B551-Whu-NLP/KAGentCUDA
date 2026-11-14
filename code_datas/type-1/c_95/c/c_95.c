#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

double estimateSnr(const float* corrSum, const int* corrValidCount, const float* maxval, float* snrValue, const int size) {
    clock_t start = clock();

    for (int idx = 0; idx < size; idx++) {
        if (corrValidCount[idx] - 1 == 0) {
            snrValue[idx] = 0.0f;
            continue;
        }

        float mean = (corrSum[idx] - maxval[idx] * maxval[idx]) / (corrValidCount[idx] - 1);
        snrValue[idx] = maxval[idx] * maxval[idx] / mean;
    }

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char* argv[]) {
    int size = 1000000; // 增大数组大小
    float* h_corrSum = (float*)malloc(size * sizeof(float));
    int* h_corrValidCount = (int*)malloc(size * sizeof(int));
    float* h_maxval = (float*)malloc(size * sizeof(float));
    float* h_snrValue = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        h_corrSum[i] = (float)(i + 1);
        h_corrValidCount[i] = i % 100 + 1;
        h_maxval[i] = (float)(i);
    }

    double time_taken = estimateSnr(h_corrSum, h_corrValidCount, h_maxval, h_snrValue, size);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* file = fopen("c_95_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "SNR Value at index %d: %f\n", i, h_snrValue[i]);
            }
            fclose(file);
        }
    }

    free(h_corrSum);
    free(h_corrValidCount);
    free(h_maxval);
    free(h_snrValue);

    printf("Time: %.2f(ms)\n", time_taken);
    return 0;
}