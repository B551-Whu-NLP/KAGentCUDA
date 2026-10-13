#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void multMat(int n, int* arrForce_d, int* arrDistance_d, int* arrAnswer_d) {
    for (int i = 0; i < n; i++) {
        arrAnswer_d[i] = arrForce_d[i] * arrDistance_d[i];
    }
}

int main(int argc, char* argv[]) {
    int n = 10000000;
    int* h_arrForce = (int*)malloc(n * sizeof(int));
    int* h_arrDistance = (int*)malloc(n * sizeof(int));
    int* h_arrAnswer = (int*)malloc(n * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < n; ++i) {
        h_arrForce[i] = i;
        h_arrDistance[i] = i * 2;
    }

    clock_t start = clock();

    multMat(n, h_arrForce, h_arrDistance, h_arrAnswer);

    clock_t end = clock();

    double time_taken = (double)(end - start) * 1000 / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_253_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%d ", h_arrAnswer[i]);
            }
            fclose(file);
        }
    }

    free(h_arrForce);
    free(h_arrDistance);
    free(h_arrAnswer);

    return 0;
}