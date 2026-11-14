#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void downsample(float* I, float* Q, unsigned int numDownsampledSamples,
    float* downsampled_I, float* downsampled_Q, unsigned int factor) {
    for (int sampleIndex = 0; sampleIndex < numDownsampledSamples; sampleIndex++) {
        unsigned int absoluteIndex = sampleIndex * factor;
        downsampled_I[sampleIndex] = I[absoluteIndex];
        downsampled_Q[sampleIndex] = Q[absoluteIndex];
    }
}

int main(int argc, char* argv[]) {
    const unsigned int numDownsampledSamples = 1000000; // 增大数组大小
    const unsigned int factor = 5; // 增大下采样因子
    float* h_I = (float*)malloc(numDownsampledSamples * factor * sizeof(float));
    float* h_Q = (float*)malloc(numDownsampledSamples * factor * sizeof(float));
    float* h_downsampled_I = (float*)malloc(numDownsampledSamples * sizeof(float));
    float* h_downsampled_Q = (float*)malloc(numDownsampledSamples * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (unsigned int i = 0; i < numDownsampledSamples * factor; ++i) {
        h_I[i] = (float)(i * 2);
        h_Q[i] = (float)(i * 3);
    }

    clock_t start = clock();

    downsample(h_I, h_Q, numDownsampledSamples, h_downsampled_I, h_downsampled_Q, factor);

    clock_t end = clock();

    double timeElapsed = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    if (verbose) {
        char outputFileName[256];
        snprintf(outputFileName, sizeof(outputFileName), "c_295_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (unsigned int i = 0; i < 1000; ++i) {
                fprintf(file, "Element %u: I=%.2f, Q=%.2f\n", i, h_downsampled_I[i], h_downsampled_Q[i]);
            }
            fclose(file);
        }
    }

    free(h_I);
    free(h_Q);
    free(h_downsampled_I);
    free(h_downsampled_Q);
    
    printf("Time: %.2f(ms)\n", timeElapsed);

    return 0;
}