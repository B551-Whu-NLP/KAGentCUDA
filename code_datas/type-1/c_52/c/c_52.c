#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void resetHeap(int *heap, int *heapPtr, int numBlock) {
    for (int index = 0; index < numBlock; index++) {
        if (index == 0)
            heapPtr[0] = numBlock - 1;
        heap[index] = numBlock - index - 1;
    }
}

int main(int argc, char *argv[]) {
    const int N = 100000000; // 增大数组大小
    int* h_heap = (int*)malloc(N * sizeof(int));
    int* h_heapPtr = (int*)malloc(N * sizeof(int));

    h_heap[0] = N;
    for (int i = 0; i < N; ++i) {
        h_heapPtr[i] = i;
    }

    clock_t start = clock();
    for(int iter = 0; iter < 100000; iter++) { // 增加迭代次数
        resetHeap(h_heap, h_heapPtr, 100); // 增大numBlock
    }
    clock_t end = clock();
    
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_52_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_heapPtr[%d]: %d\n", i, h_heapPtr[i]);
            }
            fclose(fp);
        }
    }
    free(h_heap);
    free(h_heapPtr);
    return 0;
}