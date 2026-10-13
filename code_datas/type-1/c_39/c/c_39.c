#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void pathPlan(int* devSpeed, int* devSteer, int size) {
    for (int tid = 0; tid < size; tid++) {
        devSpeed[tid] += 1;
        devSteer[tid] += 1;
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 1000000;
    int* h_devSpeed = (int*)malloc(arraySize * sizeof(int));
    int* h_devSteer = (int*)malloc(arraySize * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < arraySize; ++i) {
        h_devSpeed[i] = i;
        h_devSteer[i] = 2 * i;
    }

    clock_t start = clock();

    pathPlan(h_devSpeed, h_devSteer, arraySize);

    clock_t end = clock();
    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_39_output.txt");
        FILE* outputFile = fopen(outputFileName, "w");
        if (outputFile != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(outputFile, "Speed: %d, Steer: %d\n", h_devSpeed[i], h_devSteer[i]);
            }
            fclose(outputFile);
        }
    }

    free(h_devSpeed);
    free(h_devSteer);
    printf("Time: %.2f(ms)\n", timeTaken * 1000);
    return 0;
}