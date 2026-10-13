#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void histo(const unsigned int* const vals, unsigned int* const histo, int numVals) {
    for (int i = 0; i < numVals; i++) {
        histo[vals[i]] = histo[vals[i]] + 1;
    }
}

int main(int argc, char *argv[]) {
    unsigned int numVals = 100000000; // Increased by 100x
    unsigned int* h_vals;
    unsigned int* h_histo;
    h_vals = (unsigned int*)malloc(numVals * sizeof(unsigned int));
    h_histo = (unsigned int*)calloc(numVals, sizeof(unsigned int));

    for (int i = 0; i < numVals; ++i) {
        h_vals[i] = i % (numVals/1000);
    }

    clock_t start = clock();
    histo(h_vals, h_histo, numVals);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_226_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%u ", h_histo[i]);
            }
            fclose(fp);
        }
    }

    free(h_vals);
    free(h_histo);

    return 0;
}