#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void permuteData(const float* input, float* output, int num, int devideNum, int featureSize, int priorNum, int batchSize, int verbose) {
    clock_t start = clock();

    for (int tid = 0; tid < num; tid++) {
        int numPerbatch = num * devideNum * priorNum;
        for (int s = 0; s < batchSize; s++) {
            for (int i = 0; i < priorNum; i++) {
                for (int j = 0; j < devideNum; j++) {
                    output[s * numPerbatch + tid * priorNum * devideNum + i * devideNum + j] =
                        input[s * numPerbatch + (i * devideNum * featureSize) + (j * featureSize) + tid];
                }
            }
        }
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        char filename[256];
        snprintf(filename, sizeof(filename), "c_149_output.txt");
        FILE* file = fopen(filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.6f ", output[i]);
            }
            fclose(file);
        }
    }
}

int main(int argc, char** argv) {
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    int num = 256;
    int devideNum = 16;
    int featureSize = 64;
    int priorNum = 12;
    int batchSize = 8;

    int totalInputSize = batchSize * num * devideNum * featureSize * priorNum;
    int totalOutputSize = batchSize * num * devideNum * priorNum;

    float* h_input = (float*)malloc(totalInputSize * sizeof(float));
    float* h_output = (float*)malloc(totalOutputSize * sizeof(float));

    srand(42);
    for (int i = 0; i < totalInputSize; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }

    permuteData(h_input, h_output, num, devideNum, featureSize, priorNum, batchSize, verbose);

    free(h_input);
    free(h_output);

    return 0;
}