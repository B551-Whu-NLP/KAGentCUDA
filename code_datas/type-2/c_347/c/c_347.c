#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void accumulateSum(int *input, int *output, int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += input[i];
        output[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    const int n = 500000;
    int input[n], output[n];
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < n; ++i) {
        input[i] = i + 1;
    }

    clock_t start = clock();
    accumulateSum(input, output, n);
    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_filename[100];
        sprintf(output_filename, "%s_output.txt", argv[0]);
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", output[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    return 0;
}