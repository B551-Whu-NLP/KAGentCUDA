#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void memsetCpuInt(int* data, int val, int N) {
    for (int index = 0; index < N; index++) {
        data[index] = val;
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 100000000;  // 增大数组大小
    int initialValue = 42;
    int* h_data = (int*)malloc(arraySize * sizeof(int));
    
    clock_t start = clock();

    memsetCpuInt(h_data, initialValue, arraySize);

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", timeElapsed * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_7_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%d ", h_data[i]);
            }
            fclose(file);
        }
    }

    free(h_data);
    return 0;
}