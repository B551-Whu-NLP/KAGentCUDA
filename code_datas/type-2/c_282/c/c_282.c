#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void analysis(int D[], int L[], int R[], int N) {
    int id;
    for (id = 0; id < N; id++) {
        int label = L[id];
        int ref;
        if (label == id) {
            do {
                label = R[ref = label];
            } while (ref ^ label);
            R[id] = label;
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 10000000; // 进一步增大数组大小
    int* h_D = (int*)malloc(N * sizeof(int));
    int* h_L = (int*)malloc(N * sizeof(int));
    int* h_R = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        h_D[i] = i;
        h_L[i] = i;
        h_R[i] = i;
    }

    clock_t start = clock();

    analysis(h_D, h_L, h_R, N);

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_spent * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_282_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < N; ++i) {
                fprintf(fp, "%d ", h_R[i]);
            }
            fclose(fp);
        }
    }

    free(h_D);
    free(h_L);
    free(h_R);

    return 0;
}