#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void zeroIndices(long* vec_out, const long N) {
    for (int idx = 0; idx < N; idx++) {
        vec_out[idx] = vec_out[idx] - vec_out[0];
    }
}

int main(int argc, char* argv[]) {
    long arraySize = 100000000; // 进一步增大数组大小
    long* h_vec_out = (long*)malloc(arraySize * sizeof(long));
    for (long i = 0; i < arraySize; ++i) {
        h_vec_out[i] = (long)(i);
    }

    clock_t start = clock();

    zeroIndices(h_vec_out, arraySize);

    clock_t end = clock();
    double timeElapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", timeElapsed * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_21_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%ld ", h_vec_out[i]);
            }
            fclose(file);
        }
    }

    free(h_vec_out);
    return 0;
}