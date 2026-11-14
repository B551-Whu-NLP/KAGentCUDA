#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void fill(int N, float ALPHA, float* X, int INCX) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i * INCX] = ALPHA;
    }
}

int main(int argc, char** argv) {
    int arraySize = 100000000;
    float ALPHA = 2.0;
    int INCX = 2;
    float* h_X = (float*)malloc(arraySize * INCX * sizeof(float));
    
    clock_t start = clock();
    
    fill(arraySize, ALPHA, h_X, INCX);
    
    clock_t end = clock();
    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Time: %.2f(ms)\n", timeTaken * 1000);
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_15_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%f ", h_X[i]);
            }
            fclose(file);
        }
    }
    
    free(h_X);
    return 0;
}