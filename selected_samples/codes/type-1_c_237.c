#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void expandScoreFactors(const float* input, float* output, int dims, int clsNum) {
    for (int tid = 0; tid < dims; tid++) {
        int k = tid % clsNum;
        output[tid] = input[k];
    }
}

int main(int argc, char** argv) {
    int dims = 1000000;
    int clsNum = 3;
    float* h_input = (float*)malloc(clsNum * sizeof(float));
    float* h_output = (float*)malloc(dims * sizeof(float));
    
    for (int i = 0; i < clsNum; ++i) {
        h_input[i] = (float)(i + 1);
    }

    clock_t start = clock();
    
    expandScoreFactors(h_input, h_output, dims, clsNum);
    
    clock_t end = clock();
    double timeElapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_237_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.6f ", h_output[i]);
            }
            fclose(file);
        }
    } 
    
    free(h_input);
    
    printf("Time: %.2f(ms)\n", timeElapsed * 1000);free(h_output);
    
    return 0;
}