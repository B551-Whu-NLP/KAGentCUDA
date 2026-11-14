#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void runFilter(float* I, float* Q, int samplesLength, float* filter, int filterLength, float* filtered_I, float* filtered_Q, int convLength) {
    for (int sampleIndex = 0; sampleIndex < convLength; sampleIndex++) {
        int index;
        float sumI, sumQ;
        sumI = 0;
        sumQ = 0;

        for (int j = sampleIndex - filterLength + 1; j <= sampleIndex; j++) {
            index = sampleIndex - j;

            if ((j < samplesLength) && (j >= 0)) {
                sumI += filter[index] * I[j];
                sumQ += filter[index] * Q[j];
            }
        }

        filtered_I[sampleIndex] = sumI;
        filtered_Q[sampleIndex] = sumQ;
    }
}

int main(int argc, char *argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int samplesLength = 1000000;
    int filterLength = 10000;
    int convLength = samplesLength - filterLength + 1;

    float* I = (float*)malloc(samplesLength * sizeof(float));
    float* Q = (float*)malloc(samplesLength * sizeof(float));
    float* filter = (float*)malloc(filterLength * sizeof(float));
    float* filtered_I = (float*)malloc(convLength * sizeof(float));
    float* filtered_Q = (float*)malloc(convLength * sizeof(float));

    srand(42);
    for (int i = 0; i < samplesLength; i++) {
        I[i] = rand() % 100;
        Q[i] = rand() % 100;
    }

    for (int i = 0; i < filterLength; i++) {
        filter[i] = rand() % 10;
    }

    clock_t start = clock();

    runFilter(I, Q, samplesLength, filter, filterLength, filtered_I, filtered_Q, convLength);

    clock_t end = clock();
    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", timeTaken * 1000);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_152_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; i++) {
                fprintf(fp, "%.2f ", filtered_I[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 100; i++) {
                fprintf(fp, "%.2f ", filtered_Q[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(I);
    free(Q);
    free(filter);
    free(filtered_I);
    free(filtered_Q);

    return 0;
}