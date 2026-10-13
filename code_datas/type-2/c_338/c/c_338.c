#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void convolution(float* outputMatrix, float* inputMatrix, float* filter, int imageRows, int imageColumns, int filterSize) {
    clock_t start = clock();

    for (int eachRowOfImage = 0; eachRowOfImage < (int)imageRows; ++eachRowOfImage) {
        for (int eachColumnOfImage = 0; eachColumnOfImage < (int)imageColumns; ++eachColumnOfImage) {
            float convolvedValue = 0.f;

            for (int eachRowOfFilter = -filterSize / 2; eachRowOfFilter <= filterSize / 2; ++eachRowOfFilter) {
                for (int eachColumnOfFilter = -filterSize / 2; eachColumnOfFilter <= filterSize / 2; ++eachColumnOfFilter) {
                    int imageRow = eachRowOfImage + eachRowOfFilter;
                    int imageColumn = eachColumnOfImage + eachColumnOfFilter;

                    float pixelValue = (imageRow >= 0 && imageRow < imageRows && imageColumn >= 0 && imageColumn < imageColumns)
                        ? inputMatrix[imageRow * imageColumns + imageColumn]
                        : 0.f;

                    float filterValue = filter[(eachRowOfFilter + filterSize / 2) * filterSize + eachColumnOfFilter + filterSize / 2];

                    convolvedValue += pixelValue * filterValue;
                }
            }

            outputMatrix[eachRowOfImage * imageColumns + eachColumnOfImage] = convolvedValue;
        }
    }

    clock_t end = clock();
    double timeElapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", timeElapsed * 1000);
}

int main(int argc, char* argv[]) {
    const int imageRows = 128;
    const int imageColumns = 256;
    const int filterSize = 3;

    float* host_outputMatrix = (float*)malloc(imageRows * imageColumns * sizeof(float));
    float* host_inputMatrix = (float*)malloc(imageRows * imageColumns * sizeof(float));
    float* host_filter = (float*)malloc(filterSize * filterSize * sizeof(float));

    srand(0);
    for (int i = 0; i < imageRows * imageColumns; ++i) {
        host_inputMatrix[i] = (float)(rand() % 100);
    }

    for (int i = 0; i < filterSize * filterSize; ++i) {
        host_filter[i] = (float)(1);
    }

    convolution(host_outputMatrix, host_inputMatrix, host_filter, imageRows, imageColumns, filterSize);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        snprintf(outputFileName, sizeof(outputFileName), "c_338_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < imageRows; i += 32) { 
                for (int j = 0; j < imageColumns; j += 32) { 
                    fprintf(file, "%f\n", host_outputMatrix[i * imageColumns + j]);
                }
            }
            fclose(file);
        }
    }

    free(host_outputMatrix);
    free(host_inputMatrix);
    free(host_filter);

    return 0;
}