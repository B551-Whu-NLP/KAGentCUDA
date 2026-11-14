#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void upsweep_scan(int twod, int N, int *output) {
    int twod1 = twod * 2;
    int idx;
    for (idx = 0; idx + twod1 - 1 < N; idx += twod1) {
        output[idx + twod1 - 1] += output[idx + twod - 1];
    }
}

int main(int argc, char *argv[]) {
    const int N = 10000;
    const int twod = 32;
    int* h_output = (int*)malloc(N * sizeof(int));

    if (h_output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        h_output[i] = i * 2;
    }

    clock_t start = clock();

    upsweep_scan(twod, N, h_output);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_61_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_output[%d]: %d\n", i, h_output[i]);
            }
            fclose(fp);
        }
    }

    free(h_output);

    return 0;
}