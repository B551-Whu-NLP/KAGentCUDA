#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void SearchPosShmem1EQ(int key, int* devKey, int* devPos, int size) {
    for (int globalTx = 0; globalTx < size; globalTx++) {
        if (devKey[globalTx] == key) {
            devPos[0] = globalTx;
        }
    }
}

int main(int argc, char *argv[]) {
    int size = 100000000;  // 增大数组大小10倍
    int key = 50000000;    // 对应调整key值
    int* h_devKey = (int*)malloc(size * sizeof(int));
    int* h_devPos = (int*)malloc(sizeof(int));

    for (int i = 0; i < size; ++i) {
        h_devKey[i] = i;
    }

    clock_t start = clock();
    SearchPosShmem1EQ(key, h_devKey, h_devPos, size);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_246_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            fprintf(fp, "%d: %d\n", key, h_devPos[0]);
            fclose(fp);
        }
    }

    free(h_devKey);
    free(h_devPos);

    return 0;
}