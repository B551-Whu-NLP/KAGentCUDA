#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int arraySize = 100000000;
    int* h_canData = (int*)malloc(arraySize * sizeof(int));
    for (int i = 0; i < arraySize; i++) {
        h_canData[i] = i;
    }

    clock_t start = clock();
    for (int idx = 0; idx < arraySize; idx++) {
        h_canData[idx] += 1;
    }
    clock_t end = clock();

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", timeTaken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_23_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", h_canData[i]);
            }
            fclose(fp);
        }
    }

    free(h_canData);
    return 0;
}