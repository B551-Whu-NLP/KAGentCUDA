#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void boundaryCorrectIndexes(int *d_in, int *d_out, int length, int N) {
    clock_t start = clock();

    for (int idx = 0; idx < length; idx++) {
        if (d_in[idx] > N) {
            d_out[idx] = N;
        } else {
            d_out[idx] = d_in[idx];
        }
    }

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);
}

int main(int argc, char *argv[]) {
    const int length = 10000000;
    const int N = 500;
    int* h_in = (int*)malloc(length * sizeof(int));
    int* h_out = (int*)malloc(length * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < length; ++i) {
        h_in[i] = rand() % 1000;
    }

    boundaryCorrectIndexes(h_in, h_out, length, N);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", "c_60");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d\n", h_out[i]);
            }
            fclose(fp);
        }
    }

    free(h_in);
    free(h_out);

    return 0;
}