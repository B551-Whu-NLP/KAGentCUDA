#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add_kernel(float* inputleft, float* inputright, float* output, int count) {
    for (int idx = 0; idx < count; idx++) {
        output[idx] = inputleft[idx] + inputright[idx];
    }
}

int main(int argc, char** argv) {
    const int arraySize = 10000000;
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_inputleft = (float*)malloc(arraySize * sizeof(float));
    float* h_inputright = (float*)malloc(arraySize * sizeof(float));
    float* h_output = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_inputleft[i] = (float)(i);
        h_inputright[i] = (float)(2 * i);
    }

    clock_t start = clock();
    add_kernel(h_inputleft, h_inputright, h_output, arraySize);
    clock_t end = clock();
    
    double timeElapsed = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_37_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_output[i]);
            }
            fclose(file);
        }
    }

    free(h_inputleft);
    free(h_inputright);
    free(h_output);
    printf("Time: %.2f(ms)\n", timeElapsed);
    return 0;
}