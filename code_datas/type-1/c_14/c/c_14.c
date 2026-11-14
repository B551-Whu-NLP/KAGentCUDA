#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int arraySize = 100000000;
    float* h_L = (float*)malloc(arraySize * sizeof(float));
    float* h_r = (float*)malloc(arraySize * sizeof(float));
    
    for (int i = 0; i < arraySize; ++i) {
        h_L[i] = (float)(i);
        h_r[i] = (float)(2 * i);
    }

    clock_t start = clock();
    
    for (int u = 0; u < arraySize; u++) {
        h_L[u] -= h_r[u];
    }

    clock_t end = clock();
    double timeTaken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;
    
    printf("Time: %.2f(ms)\n", timeTaken);
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        FILE *fp = fopen("c_14_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%f ", h_L[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_L);
    free(h_r);
    return 0;
}