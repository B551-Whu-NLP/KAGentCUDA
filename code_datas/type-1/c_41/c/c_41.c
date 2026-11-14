#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const int voxelCount = 10000000;
    const int reductionSize = 102400000;
    int* h_flags = (int*)malloc(voxelCount * sizeof(int));
    int* h_reduction = (int*)malloc(reductionSize * sizeof(int));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < voxelCount; ++i) {
        h_flags[i] = i % 2 == 0;
    }

    clock_t start = clock();
    
    for (int tid = 0; tid < reductionSize; tid++) {
        h_reduction[tid] = (tid < voxelCount) ? h_flags[tid] : 0;
    }

    clock_t end = clock();
    double timeInSeconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", timeInSeconds * 1000);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_41_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", h_reduction[i]);
            }
            fclose(fp);
        }
    }

    free(h_flags);
    free(h_reduction);
    return 0;
}