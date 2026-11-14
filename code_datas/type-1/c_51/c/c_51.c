#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void delay_kernel(int* h_N_mobil, int* h_Tau, int iterations) {
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < *h_N_mobil; i++) {
            h_Tau[i] += 1;
        }
    }
}

int main(int argc, char *argv[]) {
    const int N = 500000;
    int* h_N_mobil = (int*)malloc(sizeof(int));
    int* h_Tau = (int*)malloc(N * sizeof(int));

    if (!h_N_mobil || !h_Tau) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    h_N_mobil[0] = N;
    for (int i = 0; i < N; ++i) {
        h_Tau[i] = i;
    }

    clock_t start = clock();
    delay_kernel(h_N_mobil, h_Tau, 1000);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_51_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Tau[%d]: %d\n", i, h_Tau[i]);
            }
            fclose(fp);
        }
    }

    free(h_N_mobil);
    free(h_Tau);
    return 0;
}