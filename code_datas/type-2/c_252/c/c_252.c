#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

void histogram(const int* x, int* bins, const int nbins, const int n) {
    for (int i = 0; i < nbins; ++i) {
        bins[i] = 0;
    }

    for (int i = 0; i < n; ++i) {
        if (x[i] >= 0 && x[i] < nbins) {
            bins[x[i]]++;
        }
    }
}

int main(int argc, char* argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int n = 100000000; // 增大数组大小
    const int nbins = 10000; // 增大bin数量
    int* h_x = (int*)malloc(n * sizeof(int));
    int* h_bins = (int*)calloc(nbins, sizeof(int));

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_x[i] = rand() % nbins;
    }

    clock_t start = clock();
    histogram(h_x, h_bins, nbins, n);
    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_spent * 1000);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "%s_output.txt", "c_252");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(f, "%d: %d\n", i, h_bins[i]);
            }
            fclose(f);
        }
    }

    free(h_x);
    free(h_bins);

    return 0;
}