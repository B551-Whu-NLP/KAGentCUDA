#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void distanceMatCalc(long int totalPixels, int availablePixels, int outPixelOffset, int patchSize, float* distMat, float* data, float filtSig) {
    clock_t start = clock();

    for (long int i = 0; i < availablePixels * totalPixels; i++) {
        int data_i = i / totalPixels + outPixelOffset;
        int data_j = i % totalPixels;
        float tmp = 0.0;

        if (data_i != data_j) {
            for (int elem = 0; elem < patchSize * patchSize; elem++) {
                float diff = data[data_i * patchSize * patchSize + elem] - data[data_j * patchSize * patchSize + elem];
                tmp += diff * diff;
            }
            tmp = expf(-tmp / filtSig);
        }

        distMat[i] = tmp;
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);
}

int main(int argc, char *argv[]) {
    long int totalPixels = 10000;
    int availablePixels = 5000;
    int outPixelOffset = 1000;
    int patchSize = 5;

    float* distMat = (float*)malloc(availablePixels * totalPixels * sizeof(float));
    float* data = (float*)malloc(totalPixels * patchSize * patchSize * sizeof(float));

    for (long int i = 0; i < totalPixels * patchSize * patchSize; i++) {
        data[i] = (float)(i % 100 + 1);
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    distanceMatCalc(totalPixels, availablePixels, outPixelOffset, patchSize, distMat, data, 1.0);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_154_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (long int i = 0; i < 1000; i++) {
                fprintf(fp, "%f ", distMat[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(distMat);
    free(data);
    return 0;
}