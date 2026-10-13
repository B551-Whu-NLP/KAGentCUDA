#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void getTopkNum(const float* inputScore, const int* inputIndex, float* outputScore, int* outputIndex, float threshold, const int dims, int* anchorIndex, int* classIndex, const int classNum, int batchSize, int totalScoreNum) {
    clock_t start = clock();

    for (int tid = 0; tid < dims; tid++) {
        for (int i = 0; i < batchSize; i++) {
            if (inputScore[i * totalScoreNum + tid] >= threshold) {
                outputScore[i * dims + tid] = inputScore[i * totalScoreNum + tid];
                outputIndex[i * dims + tid] = inputIndex[i * totalScoreNum + tid];
                anchorIndex[i * dims + tid] = outputIndex[i * dims + tid] / classNum;
                classIndex[i * dims + tid] = outputIndex[i * dims + tid] % classNum;
            }
            else {
                outputScore[i * dims + tid] = 0.0f;
                outputIndex[i * dims + tid] = -1;
                anchorIndex[i * dims + tid] = -1;
                classIndex[i * dims + tid] = -1;
            }
        }
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);
}

int main(int argc, char** argv) {
    int dims = 1000;
    int batchSize = 500;
    int totalScoreNum = 1500;
    int classNum = 100;
    float threshold = 0.5;

    float* inputScore = (float*)malloc(batchSize * totalScoreNum * sizeof(float));
    int* inputIndex = (int*)malloc(batchSize * totalScoreNum * sizeof(int));
    float* outputScore = (float*)malloc(batchSize * dims * sizeof(float));
    int* outputIndex = (int*)malloc(batchSize * dims * sizeof(int));
    int* anchorIndex = (int*)malloc(batchSize * dims * sizeof(int));
    int* classIndex = (int*)malloc(batchSize * dims * sizeof(int));

    srand(0);
    for (int i = 0; i < batchSize * totalScoreNum; i++) {
        inputScore[i] = (float)(rand() % 1000) / 1000.0f;
        inputIndex[i] = rand() % 1000;
    }

    getTopkNum(inputScore, inputIndex, outputScore, outputIndex, threshold, dims, anchorIndex, classIndex, classNum, batchSize, totalScoreNum);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_168_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f %d %d %d\n", outputScore[i], outputIndex[i], anchorIndex[i], classIndex[i]);
            }
            fclose(file);
        }
    } 

    free(inputScore);
    free(inputIndex);
    free(outputScore);
    free(outputIndex);
    free(anchorIndex);
    free(classIndex);

    return 0;
}