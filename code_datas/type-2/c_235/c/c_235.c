#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void find_max(int* data, int blockSize, int nbBlocks) {
    for (int i = 0; i < nbBlocks; ++i) {
        if (data[0] < data[i * blockSize]) {
            data[0] = data[i * blockSize];
        }
    }
}

int main(int argc, char *argv[]) {
    int dataSize = 100000000;  // 增大数据规模
    int nbBlocks = 100000;     // 增加块数量
    int blockSize = dataSize / nbBlocks;
    int* h_data = (int*)malloc(dataSize * sizeof(int));

    for (int i = 0; i < dataSize; ++i) {
        h_data[i] = i;
    }

    clock_t start = clock();
    find_max(h_data, blockSize, nbBlocks);
    clock_t end = clock();
    double timeElapsed = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", timeElapsed);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_235_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            fprintf(fp, "%d\n", h_data[0]);
            fclose(fp);
        }
    } else {
        printf("%d\n", h_data[0]);
    }

    free(h_data);
    return 0;
}