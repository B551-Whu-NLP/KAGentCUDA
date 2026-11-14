#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void RunComplexFilter(float* I, float* Q, int samplesLength, float* hr, float* hi,
    int filterLength, float* filtered_I, float* filtered_Q, int convLength) {
    clock_t start = clock();

    for (int sampleIndex = 0; sampleIndex < convLength; sampleIndex++) {
        int index;
        float sumI, sumQ;
        sumI = 0;
        sumQ = 0;
        for (int j = sampleIndex - filterLength + 1; j <= sampleIndex; j++) {
            index = sampleIndex - j;
            if ((j < samplesLength) && (j >= 0)) {
                sumI += (I[j] * hr[index]) - (Q[j] * hi[index]);
                sumQ += (I[j] * hi[index]) + (Q[j] * hr[index]);
            }
        }
        filtered_I[sampleIndex] = sumI;
        filtered_Q[sampleIndex] = sumQ;
    }

    clock_t end = clock();
    double timeElapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f\n", timeElapsed);
}

int main(int argc, char *argv[]) {
    const int samplesLength = 1000;
    const int filterLength = 16;
    const int convLength = samplesLength - filterLength + 1;
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* I, * Q, * hr, * hi, * filtered_I, * filtered_Q;

    I = (float*)malloc(samplesLength * sizeof(float));
    Q = (float*)malloc(samplesLength * sizeof(float));
    hr = (float*)malloc(filterLength * sizeof(float));
    hi = (float*)malloc(filterLength * sizeof(float));
    filtered_I = (float*)malloc(convLength * sizeof(float));
    filtered_Q = (float*)malloc(convLength * sizeof(float));

    for (int i = 0; i < samplesLength; ++i) {
        I[i] = (float)(i % 100);
        Q[i] = (float)(i * 2 % 100);
    }

    for (int i = 0; i < filterLength; ++i) {
        hr[i] = (float)(i % 50);
        hi[i] = (float)(i * 2 % 50);
    }

    RunComplexFilter(I, Q, samplesLength, hr, hi, filterLength, filtered_I, filtered_Q, convLength);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_341_output.txt");
        FILE *file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < convLength; i++) {
                fprintf(file, "filtered_I[%d] = %.2f, filtered_Q[%d] = %.2f\n", i, filtered_I[i], i, filtered_Q[i]);
            }
            fclose(file);
        }
    }

    free(I);
    free(Q);
    free(hr);
    free(hi);
    free(filtered_I);
    free(filtered_Q);
    return 0;
}