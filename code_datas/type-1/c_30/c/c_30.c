#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sum_arrays(int* a, int* b, int* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    int arraySize = 10000000;
    int* h_a = (int*)malloc(arraySize * sizeof(int));
    int* h_b = (int*)malloc(arraySize * sizeof(int));
    int* h_c = (int*)malloc(arraySize * sizeof(int));
    
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    clock_t start = clock();
    sum_arrays(h_a, h_b, h_c, arraySize);
    clock_t end = clock();

    double timeTaken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
    printf("Time: %.2f(ms)\n", timeTaken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_30_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", h_c[i]);
            }
            fclose(fp);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}