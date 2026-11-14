#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void countRangesGlobal(int size, int* A, int* B) {
    for (int i = 0; i < size; i++) {
        int x = A[i] / 100;
        B[x] += 1;
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 10000000; // 增大数组大小
    int* h_A = (int*)malloc(arraySize * sizeof(int));
    int* h_B = (int*)malloc(arraySize * sizeof(int));

    srand(42);
    for (int i = 0; i < arraySize; ++i) {
        h_A[i] = rand() % 10000;
        h_B[i] = 0;
    }

    clock_t start = clock();
    countRangesGlobal(arraySize, h_A, h_B);
    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeElapsed);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_27_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%d ", h_B[i]);
            }
            fclose(file);
        }
    }

    free(h_A);
    free(h_B);
    return 0;
}