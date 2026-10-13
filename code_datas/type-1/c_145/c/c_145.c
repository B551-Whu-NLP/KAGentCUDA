#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void CDFfunction(float* median, float* stdvLogNormalFrame, float* MeanLogNormalFrame, unsigned char* currentFrame, int pixelsPerFrame) {
    int pixel;
    for (pixel = 0; pixel < pixelsPerFrame; pixel++) {
        float newvalue;
        float x = currentFrame[pixel];
        newvalue = -((logf(x) - median[pixel]) - MeanLogNormalFrame[pixel]) / (sqrtf(2.0f) * stdvLogNormalFrame[pixel]);
        float summ = 0.5f + 0.5f * erff(newvalue);
        if (summ >= 0.3f) {
            currentFrame[pixel] = (unsigned char)255;
        }
        else {
            currentFrame[pixel] = (unsigned char)0;
        }
    }
}

int main(int argc, char** argv) {
    int pixelsPerFrame = 409600;
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_median = (float*)malloc(pixelsPerFrame * sizeof(float));
    float* h_stdvLogNormalFrame = (float*)malloc(pixelsPerFrame * sizeof(float));
    float* h_MeanLogNormalFrame = (float*)malloc(pixelsPerFrame * sizeof(float));
    unsigned char* h_currentFrame = (unsigned char*)malloc(pixelsPerFrame * sizeof(unsigned char));

    for (int i = 0; i < pixelsPerFrame; ++i) {
        h_median[i] = 5.0f;
        h_stdvLogNormalFrame[i] = 1.0f;
        h_MeanLogNormalFrame[i] = 0.0f;
        h_currentFrame[i] = i % 256;
    }

    clock_t start = clock();

    double timeElapsed = 0.0;
    CDFfunction(h_median, h_stdvLogNormalFrame, h_MeanLogNormalFrame, h_currentFrame, pixelsPerFrame);

    clock_t end = clock();

    timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_145_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%d ", h_currentFrame[i]);
                if ((i + 1) % 32 == 0) {
                    fprintf(file, "\n");
                }
            }
            fclose(file);
        }
    }

    free(h_median);
    free(h_stdvLogNormalFrame);
    free(h_MeanLogNormalFrame);
    free(h_currentFrame);
    return 0;
}