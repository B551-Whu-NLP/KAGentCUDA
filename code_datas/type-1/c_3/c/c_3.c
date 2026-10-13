#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void square(int* array, int arrayCount) {
    for (int idx = 0; idx < arrayCount; idx++) {
        array[idx] *= array[idx];
    }
}

int main(int argc, char* argv[]) {
    int arrayCount = 10000000;
    int* h_array = (int*)malloc(arrayCount * sizeof(int));
    for (int i = 0; i < arrayCount; ++i) {
        h_array[i] = i;
    }

    clock_t start = clock();

    square(h_array, arrayCount);

    clock_t end = clock();

    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_3_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%d ", h_array[i]);
            }
            fclose(file);
        }
    }
    printf("\nTime: %.2fms\n", timeElapsed * 1000);

    free(h_array);
    return 0;
}